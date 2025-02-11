# from google.colab import files
from IPython.display import display, Video

from pathlib import Path
import os
import torch
from pathlib import Path
from midi2audio import FluidSynth
import ffmpeg
import random
import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace

from utils import download_gdrive, create_dropdown, adjust_valence_arousal
from midi.src.generate import generate
from midi.src.midi_model import build_model
from video.src.classify import VideoEmotionClassifier


# script_dir = os.path.dirname(os.path.abspath(__file__))
# code_model_dir = os.path.abspath(os.path.join(script_dir, 'model'))
# code_utils_dir = os.path.join(code_model_dir, 'utils')
# sys.path.extend([code_model_dir, code_utils_dir])


### CONFIG

args = SimpleNamespace(
    input_path=None,
    soundfont_path='../large_files/custom.sf2',
    midi_model_dir='midi/midi_files',
    video_model_dir='video/weights_and_labels',
    keep_aux_files=False,
    start_from_beginning=False,
    min_scenecut_distance=4.0,
    max_v=0.8,
    min_v=-0.8,
    max_a=0.8,
    min_a=-0.8,
    dont_adjust_va=False,
    probabilistic_va=False,
    no_cuda=False,
    max_input_len=1216,
    temp=1.35,
    topk=-1,
    topp=0.6,
    seed=0,
    no_amp=False,
    penalty_coefficient=0.4,
    quiet=False,
    batch_size=1,
    min_n_instruments=0,
    less_instruments=False
)

temp = [args.temp, args.temp]

if args.seed > 0:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if using multi-GPU

emotion_categories = ('anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise')
video_classifier = VideoEmotionClassifier(args.video_model_dir, visualize=True)

main_model_dir = Path(".")

model_dir_full = main_model_dir / args.midi_model_dir
experiment = model_dir_full.parents[0].name

output_dir = Path('input_output')

model_fp = Path('../large_files/midi_model.pt') 

if not model_fp.exists():
    id = '1SWiA-vb4DNTxorX92K91Y31zMpJRANUz'
    download_gdrive(id, model_fp)

mappings_fp = model_dir_full / 'mappings.pt'
config_fp = model_dir_full / 'midi_model_config.pt'

maps = torch.load(mappings_fp, weights_only=False)

device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')

verbose = not args.quiet
if verbose:
    if device == torch.device("cuda"):
        # print("Using GPU " + utils.get_gpu_name())
        print("Using GPU")
    else:
        print("Using CPU")

# Load model
config = torch.load(config_fp, weights_only=False)

model, _ = build_model(None, load_config_dict=config)
model = model.to(device)
if os.path.exists(model_fp):
    model.load_state_dict(torch.load(model_fp, weights_only=True, map_location=device))
else:
    raise ValueError("Model not found")

if args.start_from_beginning:
    primers = [["<START>", "<BAR>"]]
else:
    primers = [["<BAR>"]]

if not os.path.exists(args.soundfont_path):
    id = '1aimFDohXM8RbKSqyxAaADkPGkpBHszWG'
    download_gdrive(id, args.soundfont_path)

### RUN

input_path = 'input_output/sample2.mp4'

output = video_classifier.run(input_path)
predictions = output['predictions']
scenecuts = output['scenecuts']
duration = output['duration']

sorted_indices = np.argsort(predictions)[::-1]
top_prediction = emotion_categories[sorted_indices[0]]    # remove percentage and space for safe file naming


''' Mapping emotions to valence-arousal values

The file "emotion_to_va.csv" contains the means and standard deviations
for Gaussian distributions of valence-arousal values of each discrete emotion.

Source: J. A. Russell and A. Mehrabian, “Evidence for a three-factor theory of emotions,”
Journal of research in Personality, vol. 11, no. 3, pp. 273-294, 1977.

The output of discrete emotion classifier is a probability vector, containing all emotions,
e.g. 44% joy, 23% surprise, ...

Using these probabilities as weights, we can create a mixture of Gaussians.

mixture_mean = np.sum(weights * means)
mixture_var = np.sum(weights * (stds ** 2 + means ** 2)) - (mixture_mean ** 2)
'''

stats = pd.read_csv(Path("video") / 'weights_and_labels' / 'emotion_to_va.csv')
if not args.dont_adjust_va:
    stats = adjust_valence_arousal(stats, max_a=args.max_a, min_a=args.min_a, max_v=args.max_v, min_v=args.min_v)
weights = predictions / predictions.sum()
valence_mixture_mean = np.sum(weights * stats['V_mean'].values)
arousal_mixture_mean = np.sum(weights * stats['A_mean'].values)

if args.probabilistic_va:
    # Sample from the mixture by first selecting the Gaussian based on weights
    valence_gaussian_idx = np.random.choice(len(stats), p=weights)
    arousal_gaussian_idx = np.random.choice(len(stats), p=weights)
    
    # Get parameters of the selected Gaussians
    valence_mean = stats['V_mean'].iloc[valence_gaussian_idx]
    valence_std = stats['V_std'].iloc[valence_gaussian_idx]
    
    arousal_mean = stats['A_mean'].iloc[arousal_gaussian_idx]
    arousal_std = stats['A_std'].iloc[arousal_gaussian_idx]
    
    # Sample from the selected Gaussian
    valence = np.random.normal(valence_mean, valence_std)
    arousal = np.random.normal(arousal_mean, arousal_std)
    
else:
    valence = valence_mixture_mean
    arousal = arousal_mixture_mean

# Truncate the distribution between -1 and 1
valence = np.clip(valence, -1, 1)
arousal = np.clip(arousal, -1, 1)

# Filter scenecuts to avoid overdensity
if scenecuts == []:
    filtered_scenecuts = []
else:
    filtered_scenecuts = [scenecuts[0]]
    for cut in scenecuts[1:]:
        if cut - filtered_scenecuts[-1] >= args.min_scenecut_distance:
            filtered_scenecuts.append(cut)

continuous_conditions = [[valence, arousal]]

chord_seconds = [filtered_scenecuts]
max_length = max(len(sublist) for sublist in chord_seconds)

# Extend each sublist with float('inf') to make their lengths equal
chord_seconds = [sublist + [float('inf')] * (max_length - len(sublist)) for sublist in chord_seconds]

output_midis, _, _ = generate(
            model, maps, device, 
            output_dir, config['conditioning'], 
            continuous_conditions=continuous_conditions,
            penalty_coefficient=args.penalty_coefficient, top_p=args.topp, 
            gen_len=float('inf'), max_input_len=config['context_len'],
            amp=not args.no_amp, primers=primers, temperatures=temp, top_k=args.topk, 
            write=False, verbose=True,
            threshold_n_instruments=config['threshold_n_instruments'],
            chord_seconds=chord_seconds, less_instruments=args.less_instruments,
            duration=duration, batch_size=args.batch_size, 
            )

# Write MIDIs
fade_out_seconds = 3
if fade_out_seconds >= duration:
    fade_out_seconds = 0
fade_start_time = duration - fade_out_seconds

video = ffmpeg.input(input_path)
input_path = Path(input_path)
video_name = input_path.stem
video_extension = input_path.suffix

output_path = output_dir / f'{video_name}_output{video_extension}'

file_stem = f'{video_name}_output'

for i, mid in enumerate(output_midis):
    
    # Write MIDI
    output_midi_path = output_dir / f'{file_stem}.mid'
    mid.write(str(output_midi_path))

    # Synthesize MIDI to WAV
    output_wav_path = output_dir / f'{file_stem}.wav'


    fs = FluidSynth(args.soundfont_path)
    fs.midi_to_audio(output_midi_path, output_wav_path)

    # Get apply fade out to audio
    audio = (
        ffmpeg.input(output_wav_path)
        .filter('atrim', duration=duration)
        .filter('afade', type='out', start_time=fade_start_time, duration=fade_out_seconds)
    )

    (
        ffmpeg.output(
            video.video, 
            audio, 
            str(output_path), 
            vcodec='libx264', 
            preset='ultrafast', 
            acodec='libopus', 
            strict='experimental'
        )
        .overwrite_output()
        .run(quiet=True)
    )
    print(f"Output written to {str(output_path)}")
    if not args.keep_aux_files:
        os.remove(output_midi_path)
        os.remove(output_wav_path)

    if duration < 360:   # Show video if it won't use all the RAM
        print()
        print('Input video:')
        display(Video(str(output_path), embed=True, width=640, height=480))

    # files.download(str(output_path))
