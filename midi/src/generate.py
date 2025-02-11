import os
import numpy as np
import torch
import torch.nn.functional as F
import datetime
from pathlib import Path
import pretty_midi
import utils


def generate(model, helper, device, out_dir, conditioning, 
                penalty_coefficient=0.4, continuous_conditions=None,
                    max_input_len=1216, amp=True, batch_size=1,
                    gen_len=4096, temperatures=[1.3, 1.3], top_k=-1, 
                    top_p=0.6, write=False, varying_condition=None, 
                    verbose=False, primers=[["<START>"]],
                    threshold_n_instruments=3, less_instruments=False,
                    max_seconds_to_chord=8, chord_seconds=None, chord_sensitivity_seconds=1,
                    duration=None):
      
    with torch.no_grad():
        if write:
            os.makedirs(out_dir, exist_ok=True)
            log_path = Path(out_dir) / 'log.txt'
            with open(log_path, 'w') as f_out:
                pass
        
        model = model.to(device)
        model.eval()

        assert len(temperatures) in (1, 2)

        if varying_condition is not None:
            batch_size = varying_condition[0].size(0)
        else:
            try:
                continuous_conditions = torch.FloatTensor(continuous_conditions).to(device)
            except:
                continuous_conditions = None
            if conditioning == "none":
                batch_size = len(primers)
            else:
                batch_size = continuous_conditions.shape[0]

        timeshift_index = helper['event_to_index']['TIMESHIFT']
        chord_token = helper['pair_to_token'][(helper['event_to_index']['<CHORD>'], 0)]
                
        # will be used to penalize repeats
        repeat_counts = [0 for _ in range(batch_size)]

        # These are special symbols and shouldn't be created by the model
        exclude_symbols = ["<START>", '<PAD>']
        if threshold_n_instruments > 1:
            exclude_symbols += ['<LESS_INSTRUMENTS>', '<MORE_INSTRUMENTS>']
            if less_instruments:
                n_instruments_event = '<LESS_INSTRUMENTS>'
            else:
                n_instruments_event = '<MORE_INSTRUMENTS>'
            n_instruments_pair = (helper['event_to_index'][n_instruments_event], 0)
            n_instruments_token = torch.tensor(helper['pair_to_token'][n_instruments_pair], device=device).repeat(batch_size, 1)
            max_input_len -= 1
        else:
            n_instruments_token = None


        # Convert them to tokens
        exclude_tokens = torch.tensor([helper['pair_to_token'][(helper['event_to_index'][exclude_symbol], 0)] for exclude_symbol in exclude_symbols])
        null_conditions_tensor = torch.FloatTensor([np.nan, np.nan]).to(device)
        
        # will have generated symbols and indices

        if not isinstance(primers, list):
            primers = [[primers]]

        if len(primers) == 1:   # if one primer is used for multiple songs
            primers *= batch_size
            null_conditions_tensor = null_conditions_tensor.repeat(batch_size, 1)

        if len(chord_seconds) == 1:
            chord_seconds = chord_seconds * batch_size
            
        chord_seconds = [sublist + [float('inf')] for sublist in chord_seconds]
        chord_seconds_tensor = torch.FloatTensor(chord_seconds).to(device)

        primer_tokens = [[helper["pair_to_token"][(helper["event_to_index"][symbol], 0)] for symbol in primer] \
                        for primer in primers]

        gen_song_tensor = torch.LongTensor(primer_tokens).to(device)       
        gen_song_timed_events = [[(token, 0, 0.) for token in primer] for primer in primers]

        times_to_chord = torch.min(
            torch.ones((chord_seconds_tensor.shape[0]), device=device) * max_seconds_to_chord, 
            torch.min(chord_seconds_tensor, dim=1)[0])
        # Repeat to match the input size (primers)
        times_to_chord = times_to_chord.unsqueeze(-1).repeat(1, gen_song_tensor.shape[-1])

        max_input_len -= 2
        conditions_tensor = continuous_conditions
        # Latest generated index can be thought of the last element of primer
        gen_inds = gen_song_tensor[:, [-1]]

        i = 0
        time_cursors = torch.zeros((batch_size, 1), device=device)
        while (not bool(duration) and i < gen_len) or (bool(duration) and torch.min(time_cursors).item() < duration):
            i += 1
            if verbose:
                if duration:
                    print(f'\r{torch.min(time_cursors).item():4.1f}/{duration:4.1f} seconds', end=' ', flush=True)
                else:
                    print(f'\r{gen_len - i}', end=' ', flush=True)

            # Truncate input if needed
            input_ = gen_song_tensor
            times_to_chord_input = times_to_chord
            if gen_song_tensor.shape[-1] > max_input_len:
                input_ = input_[:, -max_input_len:]
                times_to_chord_input = times_to_chord_input[:, -max_input_len:]
                
            if gen_song_tensor.shape[-1] == max_input_len:
                print(utils.memory())

            if threshold_n_instruments > 1:      # Concatenate n_instruments token
                input_ = torch.cat((n_instruments_token, input_), 1)
                # Pad times_to_chord accordingly ("same" padding)
                times_to_chord_input = torch.cat((times_to_chord_input[:, [0]], times_to_chord_input), 1)

            # INTERPOLATED CONDITIONS
            if varying_condition is not None:
                valences = varying_condition[0][:, i-1]
                arousals = varying_condition[1][:, i-1]
                conditions_tensor = torch.cat([valences[:, None], arousals[:, None]], dim=-1)

            # Run model
            with torch.amp.autocast(device.type, enabled=amp):
                # input_ = input_
                output = model(input_, conditions_tensor, times_to_chord_input)
                # output = output.permute((1, 0, 2))

            # Process output, get predicted token
            output = output[:, -1, :]     # Select last timestep
            output[output != output] = 0    # zeroing nans
            
            if torch.all(output == 0) and verbose:
                # if everything becomes zero
                print("All predictions were NaN during generation")
                output = torch.ones(output.shape).to(device)

            # exclude certain tokens
            output[:, exclude_tokens] = -float("inf")
            
            effective_temps = []
            for j in range(batch_size):
                gen_idx = gen_inds[j, 0].item()
                gen_pair = helper["token_to_pair"][gen_idx]
                effective_temp = temperatures[1]
                if isinstance(gen_pair, tuple):
                    gen_event = helper["index_to_event"][gen_pair[0]]
                    if "TIMESHIFT" == gen_event:
                        # switch from rest temperature to note temperature
                        effective_temp = temperatures[0]
                effective_temps.append(effective_temp)

            temp_tensor = torch.Tensor(effective_temps).to(device)

            output = F.log_softmax(output, dim=-1)

            # Add repeat penalty to temperature
            if penalty_coefficient > 0:
                repeat_counts_array = torch.Tensor(repeat_counts).to(device)
                temp_multiplier = torch.maximum(torch.zeros_like(repeat_counts_array, device=device), 
                    torch.log((repeat_counts_array+1)/4)*penalty_coefficient)
                repeat_penalties = temp_multiplier * temp_tensor
                temp_tensor += repeat_penalties

            # Apply temperature
            output /= temp_tensor.unsqueeze(-1)
            
            # top-k
            if top_k <= 0 or top_k > output.size(-1): 
                top_k_eff = output.size(-1)
            else:
                top_k_eff = top_k
            output, top_inds = torch.topk(output, top_k_eff, dim=-1)

            # top-p
            if top_p > 0 and top_p < 1:
                cumulative_probs = torch.cumsum(F.softmax(output, dim=-1), dim=-1)
                remove_inds = cumulative_probs > top_p
                remove_inds[:, 0] = False   # at least keep top value
                output[remove_inds] = -float("inf")

            output = F.softmax(output, dim=-1)
        
            # Sample from probabilities
            inds_sampled = torch.multinomial(output, 1, replacement=True)
            gen_inds = top_inds.gather(1, inds_sampled)

            # Update repeat counts
            num_choices = torch.sum((output > 0).int(), -1)
            
            for j in range(batch_size):
                if num_choices[j] <= 2: repeat_counts[j] += 1
                else: repeat_counts[j] = repeat_counts[j] // 2

            # Update output song
            gen_song_tensor = torch.cat((gen_song_tensor, gen_inds), 1)

            # Update times_to_chord
            
            # Extend the time-to-chord tensor by observing timeshifts
            # Find out which indices in the batch are timeshift tokens
            
            gen_pairs = []
            for j, gen_ind in enumerate(gen_inds):
                gen_pair = helper['token_to_pair'][gen_ind.item()]
                gen_pairs.append(gen_pair)
                event = helper['index_to_event'][gen_pair[0]]
                value = gen_pair[1]
                if event == 'TIMESHIFT':
                    value = value * helper['tick_to_ms'] / 1000
                elif event[0] == '<':
                    pass
                else:
                    value = pretty_midi.note_number_to_name(value)

                gen_event = (helper['index_to_event'][gen_pair[0]], value, round(time_cursors[j].item(), 2))
                
                gen_song_timed_events[j].append(gen_event)
                if write and verbose:
                    with open(log_path, 'a') as f_out:
                        f_out.write(str(i) + ': ' + str(gen_event) + '\n')

            if write and verbose:
                with open(log_path, 'a') as f_out:
                    f_out.write('\n')
            gen_pairs = torch.tensor(gen_pairs, device=device)

            # If <CHORD> token is generated correctly, we will remove that from target chord locations
            is_chord = (gen_inds == chord_token)  # find which samples are <CHORD>
            # Find which target chord is generated, using a proximity of 1 second
            chord_generated = torch.abs(time_cursors - chord_seconds_tensor) < chord_sensitivity_seconds
            # Combine the boolean masks for direct indexing
            combined_mask = is_chord & chord_generated
            # Remove those chords
            chord_seconds_tensor[combined_mask] = -1

            is_timeshift = gen_pairs[:, 0] == timeshift_index
            # Get shift amounts, convert to seconds. If token isn't timeshift, use zero
            shift_amounts = torch.where(is_timeshift, gen_pairs[:, 1] * helper['tick_to_ms'] / 1000, 0).unsqueeze(-1).to(device)
            time_cursors += shift_amounts   # Update time cursors
            if chord_seconds is not None:
                # Find the remaining time until the next nearest chord
                time_to_chord = chord_seconds_tensor - time_cursors    # difference
                time_to_chord[time_to_chord < 0] = float('inf')   # cancel past chords, only looking at future
                time_to_chord = torch.min(time_to_chord, dim=1, keepdim=True)[0]    # find closest chord
                time_to_chord = torch.clamp(time_to_chord, 0, max_seconds_to_chord)   # clamp to max value
                times_to_chord = torch.cat((times_to_chord, time_to_chord), 1)
            else:
                times_to_chord = torch.tensor([])

        output_midis = []
        for i in range(gen_song_tensor.size(0)):
            now = datetime.datetime.now()
            out_file_name = now.strftime("%Y_%m_%d_%H_%M_%S")
            out_dir = Path(out_dir)
            out_mid_path = out_file_name + ".mid"
            out_mid_path = out_dir / out_mid_path

            if verbose:
                print("")
            mid = utils.ind_tensor_to_mid(gen_song_tensor[i, :], helper["token_to_pair"], helper["index_to_event"], 
                                    timeshift_index, helper['tick_to_ms'], verbose=False)
            output_midis.append(mid)
            if write:
                mid.write(str(out_mid_path))
                if verbose:
                    print(f"Saved to {out_mid_path}")
                f_out.close()

    return output_midis, None, None

