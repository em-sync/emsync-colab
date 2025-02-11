# from argparse import ArgumentParser
# import os
# import sys
# import torch
# from pathlib import Path
# from midi2audio import FluidSynth
# import ffmpeg
# import random
# import torch
# import numpy as np
# import pandas as pd
# from utils import download_gdrive
# from midi.src.generate import generate
# from midi.src.midi_model import build_model
# from video.src.classify import run_video

# def adjust_valence_arousal(df, max_a=1, min_a=-1, max_v=1, min_v=-1):
#     """
#     Adjusts valence and arousal means and standard deviations to fit within [-max_norm, max_norm].
    
#     Args:
#         df (pd.DataFrame): DataFrame containing columns 'V_mean', 'V_std', 'A_mean', 'A_std'.
#         max_norm (float): The maximum absolute value for the normalized range.
        
#     Returns:
#         pd.DataFrame: Adjusted DataFrame with normalized valence and arousal values.
#     """

#     df["V_mean_new"] = df["V_mean"]
#     df["V_std_new"] = df["V_std"]

#     # Adjust Valence
#     V_min, V_max = df["V_mean"].min(), df["V_mean"].max()
#     V_scaling = (max_v - min_v) / (V_max - V_min)
#     df["V_mean_new"] = (df["V_mean"] - V_min) * V_scaling + min_v
#     df["V_std_new"] = df["V_std"] * V_scaling

#     # Adjust Arousal
#     A_min, A_max = df["A_mean"].min(), df["A_mean"].max()
#     A_scaling = (max_a - min_a) / (A_max - A_min)
#     df["A_mean_new"] = (df["A_mean"] - A_min) * A_scaling + min_a
#     df["A_std_new"] = df["A_std"] * A_scaling

#     # Prepare adjusted DataFrame
#     adjusted_df = df[["word", "V_mean_new", "V_std_new", "A_mean_new", "A_std_new"]]
#     adjusted_df.rename(columns={
#         "V_mean_new": "V_mean", 
#         "V_std_new": "V_std", 
#         "A_mean_new": "A_mean", 
#         "A_std_new": "A_std"
#     }, inplace=True)
    
#     return adjusted_df


# if __name__ == '__main__':
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     code_model_dir = os.path.abspath(os.path.join(script_dir, 'model'))
#     code_utils_dir = os.path.join(code_model_dir, 'utils')
#     sys.path.extend([code_model_dir, code_utils_dir])

#     parser = ArgumentParser()


#     parser.add_argument('--input_path', type=str, help='Input video path')
#     parser.add_argument('--output_path', type=str, help='Output video path')
#     parser.add_argument('--youtube_url', type=str, help='Youtube URL')
#     parser.add_argument('--soundfont_path', type=str, help='Soundfont path',
#                         default='midi/midi_files/custom.sf2')
#     parser.add_argument('--midi_model_dir', type=str, help='Trained MIDI model folder', 
#                         default='midi/midi_files/midi_generator_weights')
#     parser.add_argument('--video_model_dir', type=str, help='Trained video model folder', 
#                         default='video/weights_and_labels/emotion_classifier_weights')
#     parser.add_argument('--keep_aux_files', action='store_true', 
#                         help="Keep auxilliary (WAV and MID) files")
#     parser.add_argument('--start_from_beginning', action='store_true', 
#                         help="Starts the song beginning (primer becomes [<START>, <BAR>] as opposed to [<BAR>])")
#     parser.add_argument('--min_scenecut_distance', type=float, default=4, 
#                         help='Get rid off some scenecuts to establish a minimum distance between them.')
#     parser.add_argument('--max_v', type=float, default=0.8, 
#                         help="Maximum arousal")
#     parser.add_argument('--min_v', type=float, default=-0.8, 
#                     help="Minimum arousal")
#     parser.add_argument('--max_a', type=float, default=0.8, 
#                         help="Maximum arousal")
#     parser.add_argument('--min_a', type=float, default=-0.8, 
#                     help="Minimum arousal")

#     parser.add_argument('--dont_adjust_va', action='store_true', help="Dont adjust valence-arousal")

#     parser.add_argument('--probabilistic_va', action='store_true', help="Sample valence-arousal instead of using means")
#     parser.add_argument('--no_cuda', action='store_true', help="Use CPU")

#     parser.add_argument('--max_input_len', type=int, help='Max input len', default=1216)
#     parser.add_argument('--temp', type=float, help='Generation temperature', default=1.35)
#     parser.add_argument('--topk', type=int, help='Top-k sampling', default=-1)
#     parser.add_argument('--topp', type=float, help='Top-p sampling', default=0.6)
#     parser.add_argument('--seed', type=int, default=0, help="Random seed")
#     parser.add_argument('--no_amp', action='store_true', help="Disable automatic mixed precision")
#     parser.add_argument('--penalty_coefficient', type=float, default=0.4,
#                         help="Coefficient for penalizing repeating notes")
#     parser.add_argument("--quiet", action='store_true', help="Not verbose")

#     parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
#     parser.add_argument('--min_n_instruments', type=int, help='Minimum number of instruments', default=0)
#     parser.add_argument('--less_instruments', action='store_true', help="Uses less instruments (1 or 2)")


#     args = parser.parse_args()

#     temp = [args.temp, args.temp]

#     assert bool(args.input_path) ^ bool(args.youtube_url), 'Please provide either video path or Youtube URL.'

#     if args.seed > 0:
#         random.seed(args.seed)
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         torch.cuda.manual_seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)  # if using multi-GPU

#     emotion_categories = ('anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise')

#     output = run_video(model_dir=args.video_model_dir, video_path=args.input_path, youtube_url=args.youtube_url, visualize=True)
#     predictions = output['predictions']
#     scenecuts = output['scenecuts']
#     duration = output['duration']

#     sorted_indices = np.argsort(predictions)[::-1]
#     top_prediction = emotion_categories[sorted_indices[0]]    # remove percentage and space for safe file naming


#     ''' Mapping emotions to valence-arousal values

#     The file "emotion_to_va.csv" contains the means and standard deviations
#     for Gaussian distributions of valence-arousal values of each discrete emotion.

#     Source: J. A. Russell and A. Mehrabian, “Evidence for a three-factor theory of emotions,”
#     Journal of research in Personality, vol. 11, no. 3, pp. 273-294, 1977.

#     The output of discrete emotion classifier is a probability vector, containing all emotions,
#     e.g. 44% joy, 23% surprise, ...

#     Using these probabilities as weights, we can create a mixture of Gaussians.

#     mixture_mean = np.sum(weights * means)
#     mixture_var = np.sum(weights * (stds ** 2 + means ** 2)) - (mixture_mean ** 2)
#     '''

#     stats = pd.read_csv(Path("video") / 'weights_and_labels' / 'emotion_to_va.csv')
#     if not args.dont_adjust_va:
#         stats = adjust_valence_arousal(stats, max_a=args.max_a, min_a=args.min_a, max_v=args.max_v, min_v=args.min_v)
#     weights = predictions / predictions.sum()
#     valence_mixture_mean = np.sum(weights * stats['V_mean'].values)
#     arousal_mixture_mean = np.sum(weights * stats['A_mean'].values)

#     if args.probabilistic_va:
#         # Sample from the mixture by first selecting the Gaussian based on weights
#         valence_gaussian_idx = np.random.choice(len(stats), p=weights)
#         arousal_gaussian_idx = np.random.choice(len(stats), p=weights)
        
#         # Get parameters of the selected Gaussians
#         valence_mean = stats['V_mean'].iloc[valence_gaussian_idx]
#         valence_std = stats['V_std'].iloc[valence_gaussian_idx]
        
#         arousal_mean = stats['A_mean'].iloc[arousal_gaussian_idx]
#         arousal_std = stats['A_std'].iloc[arousal_gaussian_idx]
        
#         # Sample from the selected Gaussian
#         valence = np.random.normal(valence_mean, valence_std)
#         arousal = np.random.normal(arousal_mean, arousal_std)
        
#     else:
#         valence = valence_mixture_mean
#         arousal = arousal_mixture_mean

#     # Truncate the distribution between -1 and 1
#     valence = np.clip(valence, -1, 1)
#     arousal = np.clip(arousal, -1, 1)
    
#     # Filter scenecuts to avoid overdensity
#     if scenecuts == []:
#         filtered_scenecuts = []
#     else:
#         filtered_scenecuts = [scenecuts[0]]
#         for cut in scenecuts[1:]:
#             if cut - filtered_scenecuts[-1] >= args.min_scenecut_distance:
#                 filtered_scenecuts.append(cut)


#     main_model_dir = Path(".")

#     model_dir_full = main_model_dir / args.midi_model_dir
#     experiment = model_dir_full.parents[0].name

#     # output_dir = Path(args.output_path).parent
#     output_dir = Path('input_output')

#     model_fp = model_dir_full / 'midi_model.pt'

#     if not model_fp.exists():
#         id = '1SWiA-vb4DNTxorX92K91Y31zMpJRANUz'
#         download_gdrive(id, model_fp)

#     mappings_fp = model_dir_full / 'mappings.pt'
#     config_fp = model_dir_full / 'model_config.pt'

#     maps = torch.load(mappings_fp, weights_only=False)

#     device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    
#     verbose = not args.quiet
#     if verbose:
#         if device == torch.device("cuda"):
#             # print("Using GPU " + utils.get_gpu_name())
#             print("Using GPU")
#         else:
#             print("Using CPU")

#     # Load model
#     config = torch.load(config_fp, weights_only=False)

#     model, _ = build_model(None, load_config_dict=config)
#     model = model.to(device)
#     if os.path.exists(model_fp):
#         model.load_state_dict(torch.load(model_fp, weights_only=True, map_location=device))
#     else:
#         raise ValueError("Model not found")

#     if args.start_from_beginning:
#         primers = [["<START>", "<BAR>"]]
#     else:
#         primers = [["<BAR>"]]
    
#     continuous_conditions = [[valence, arousal]]
    
#     chord_seconds = [filtered_scenecuts]
#     max_length = max(len(sublist) for sublist in chord_seconds)

#     # Extend each sublist with float('inf') to make their lengths equal
#     chord_seconds = [sublist + [float('inf')] * (max_length - len(sublist)) for sublist in chord_seconds]

#     output_midis, _, _ = generate(
#                 model, maps, device, 
#                 output_dir, config['conditioning'], 
#                 continuous_conditions=continuous_conditions,
#                 penalty_coefficient=args.penalty_coefficient, top_p=args.topp, 
#                 gen_len=float('inf'), max_input_len=config['context_len'],
#                 amp=not args.no_amp, primers=primers, temperatures=temp, top_k=args.topk, 
#                 write=False, verbose=True,
#                 threshold_n_instruments=config['threshold_n_instruments'],
#                 chord_seconds=chord_seconds, less_instruments=args.less_instruments,
#                 duration=duration, batch_size=args.batch_size, 
#                 )
    
#     # Write MIDIs
#     fade_out_seconds = 3
#     if fade_out_seconds >= duration:
#         fade_out_seconds = 0
#     fade_start_time = duration - fade_out_seconds
    
#     video = ffmpeg.input(args.input_path)
#     input_path = Path(args.input_path)
#     video_name = input_path.stem
#     video_extension = input_path.suffix

#     output_path = output_dir / f'{video_name}_output{video_extension}'

#     file_stem = f'{video_name}_output'

#     for i, mid in enumerate(output_midis):
        
#         # Write MIDI
#         output_midi_path = output_dir / f'{file_stem}.mid'
#         mid.write(str(output_midi_path))

#         # Synthesize MIDI to WAV
#         output_wav_path = output_dir / f'{file_stem}.wav'

#         if not os.path.exists(args.soundfont_path):
#             id = '1aimFDohXM8RbKSqyxAaADkPGkpBHszWG'
#             download_gdrive(id, args.soundfont_path)
#         fs = FluidSynth(args.soundfont_path)
#         fs.midi_to_audio(output_midi_path, output_wav_path)

#         # Get apply fade out to audio
#         audio = (
#             ffmpeg.input(output_wav_path)
#             .filter('atrim', duration=duration)
#             .filter('afade', type='out', start_time=fade_start_time, duration=fade_out_seconds)
#         )

#         (
#             ffmpeg.output(video.video, audio, str(output_path), vcodec='copy', acodec='libopus', strict='experimental')
#             .overwrite_output()
#             .run(quiet=True)
#         )
#         print(f"Output written to {str(output_path)}")
#         if not args.keep_aux_files:
#             os.remove(output_midi_path)
#             os.remove(output_wav_path)
