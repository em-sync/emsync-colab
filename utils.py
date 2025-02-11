import pretty_midi
import numpy as np
import torch
import yt_dlp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import ffmpeg
import gdown
import subprocess
import re

# Groups are custom made
GROUP_TO_PROGRAM = {
    'DRUMS': 0,
    'PIANO': 0,  # acoustic grand piano
    'GUITAR': 25,   # steel guitar
    'BASS': 32,     # acoustic bass
    'STRINGS': 48,   # string ensemble 2
}

def ind_tensor_to_mid(x, idx2tuple, idx2event, timeshift_index, tick_to_ms, verbose=False):
    # Indices to midi
    x = ind_tensor_to_tuples(x, idx2tuple, timeshift_index, tick_to_ms)
    x = tuples_to_mid(x, idx2event, verbose=verbose)
    return x


def ind_tensor_to_tuples(x, ind2tuple, timeshift_index, tick_to_ms):
    # Indices to tuples
    tuples = []
    for el in x:
        tup = ind2tuple[el.item()]
        if isinstance(tup, tuple):
            event, value = tup
            if event == timeshift_index:
                value *= tick_to_ms
                tup = (event, value)
        tuples.append(tup)
    return tuples

def tuples_to_mid(x, idx2event, verbose=False, default_velocity=80, high_velocity=120):

    tracks = {}
    for instrument, program in GROUP_TO_PROGRAM.items():
        track = pretty_midi.Instrument(program=program, is_drum=instrument == 'DRUMS', name=instrument.lower())
        track.notes = []
        tracks.update({instrument: track})

    active_notes = {}
    chord_active = False
    # velocity = default_velocity
    time_cursor = 0
    for el in x:
        el = (idx2event[el[0]], el[1])
        event = el[0]
        if event == '<CHORD>':
            chord_active = True
        if not event.startswith("<"):     # if not special token
            # event = idx2event[el[0]]       
            if "TIMESHIFT" == event:
                timeshift = float(el[1])
                time_cursor += timeshift / 1000.0
                chord_active = False
            else:
                on_off, instrument = event.split("_")
                pitch = int(el[1])
                if on_off == "ON":
                    velocity = high_velocity if chord_active and instrument in ('GUITAR', 'PIANO') else default_velocity
                    active_notes.update({(instrument, pitch): (time_cursor, velocity)})
                elif (instrument, pitch) in active_notes:
                    start, note_velocity = active_notes[(instrument, pitch)]
                    end = time_cursor
                    tracks[instrument].notes.append(pretty_midi.Note(note_velocity, pitch, start, end))
                elif verbose:  
                    print("Ignoring {:>15s} {:4} because there was no previos ""ON"" event".format(event, pitch))

    mid = pretty_midi.PrettyMIDI(initial_tempo=240.0)
    mid.instruments += tracks.values()
    return mid


def equidistant_indices(source_length, target_length):
    assert source_length > target_length, 'Source length is less than target length.'
    inds = np.round(np.linspace(0, source_length - 1, num=target_length)).astype(np.int32).tolist()
    assert np.array_equal(inds, np.unique(inds))
    return inds


def ffmpeg_scene_detect(input_, threshold=0.27):
    # Returns timestamps of scenecuts
    commands_flat = f"ffmpeg -i {input_} -vsync vfr -vf select=scene -loglevel debug -f null /dev/null 2>&1 | grep scene:" 
    output = subprocess.run(commands_flat, shell=True, capture_output=True, text=True).stdout
    output = output.split("\n")
    scenes_times = []
    for line in output:
        if "scene" in line:
            line = line.split(" ")
            line = [item for item in line if ":" in item]
            line_dict = {}
            for item in line:
                key, val = item.split(":")[:2]
                if key in ("t", "scene"):
                    try:
                        val_found = float(val)
                    except:     # remove non-numeric characters
                        val_found = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", val)
                        if val_found == []:
                            continue
                        else:
                            val_found = float(val[0])

                    line_dict[key] = val_found

            if "t" in line_dict.keys() and "scene" in line_dict.keys():
                if line_dict["scene"] > threshold:
                    # scenes_times.append((line_dict["t"], line_dict["scene"]))
                    scenes_times.append(line_dict["t"])

    return scenes_times


def wrap_text(text, linewidth):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 > linewidth:
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += " "
            current_line += word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)

def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def find_largest_box(boxes):
    max_area = 0
    max_index = 0
    for i, frame in enumerate(boxes):
        for box in frame:
            area = box_area(box)
            if area > max_area:
                max_area = area
                max_index = i
    return max_index


def draw_boxes_on_image(image, shapes, labels=None, output_file=None, title=None):
    # To visualize the face emotion and OCR results
    if labels == None:
        labels = [None] * len(shapes)
    if labels and len(labels) != len(shapes):
        raise ValueError("Number of labels must match the number of shapes.")

    fig, ax = plt.subplots()
    ax.imshow(image)
    
    for i, shape in enumerate(shapes):
        if len(shape) != 4:
            raise ValueError("Each shape must contain exactly 4 points.")
        
        box_color = 'yellow'
        # If a label is provided, add it below each shape with lime color and black outline
        if labels and labels[i] is not None:
            box_color = 'lime'
            centroid_x = np.mean([point[0] for point in shape])
            centroid_y = np.max([point[1] for point in shape]) + 10  # Place label below the shape
            
            # Draw the text with a black outline
            text = ax.text(centroid_x, centroid_y, labels[i], fontsize=6, color='lime',
                           ha='center', va='top', fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                                   path_effects.Normal()])
            
        # Create a polygon patch for each shape with a yellow outline and black stroke
        # shape = np.array(shape) + 0.2 * (np.array(shape) - np.mean(shape, axis=0))
        polygon = patches.Polygon(shape, closed=True, edgecolor=box_color, linewidth=1, fill=None)
        polygon.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                  path_effects.Normal()])
        ax.add_patch(polygon)
    
    ax.axis('off')
    if title != None:
        ax.set_title(title, fontdict={'fontsize': 8})
    if output_file != None:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.2)
    else:
        plt.show(block=False)


def convert_to_points(coords):
    # Converts points of corners to coordinates
    if len(coords) != 4:
        return []
    x1, y1, x2, y2 = coords
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def get_video_duration(video_path):
    # Probe the video to get its duration
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='format=duration', format='json')
        duration = float(probe['format']['duration'])
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise
    return duration


def extract_frames_and_audio(video_path, output_fps=None, size=None):
    # Probe the video to get original width, height, fps, and audio sampling rate
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    audio_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    
    original_width = int(video_info['width'])
    original_height = int(video_info['height'])
    original_fps = eval(video_info['r_frame_rate'])  # Get original FPS
    pix_fmt = video_info.get('pix_fmt', 'rgb24')

    if audio_info:
        audio_sample_rate = int(audio_info['sample_rate'])  # Get original audio sampling rate
        num_audio_channels = int(audio_info.get('channels', 1))
    else:
        audio_sample_rate = None
        num_audio_channels = None

    # --- GET FRAMES ---

    # Determine the number of channels based on the pixel format
    if 'gray' in pix_fmt:  # Black and white video
        num_channels = 1
        pix_fmt = 'gray'
    else:  # Color video
        num_channels = 3
        pix_fmt = 'rgb24'

    # Calculate new dimensions if resizing
    if size:
        if original_height < original_width:
            new_height = size
            new_width = int((original_width / original_height) * size)
        else:
            new_width = size
            new_height = int((original_height / original_width) * size)
    else:
        new_width = original_width
        new_height = original_height

    if output_fps == None:
        output_fps = original_fps
    
    out, _ = (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=output_fps)
        .filter('scale', new_width, new_height)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )

    # Calculate the number of frames
    num_frames = len(out) // (new_width * new_height * num_channels)

    # Convert the video frames to a numpy array
    video_array = np.frombuffer(out, np.uint8).reshape((num_frames, new_height, new_width, num_channels))

    # --- GET AUDIO ---

    if audio_info:      # if available
        bits = 32
        dtype_ = eval(f"np.int{bits}")
        max_ = np.iinfo(dtype_).max
        format = f"s{bits}le"

        out_audio, _ = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format=format)
            .run(capture_stdout=True, quiet=True)
        )
        # Convert the audio to a numpy array
        audio_array = np.frombuffer(out_audio, dtype=dtype_).astype(np.float32)
        audio_array = audio_array.reshape((-1, num_audio_channels)).T / max_
    else:
        audio_array = None

    return video_array, audio_array, original_fps, audio_sample_rate

def download_gdrive(id, target_path):
    url = f'https://drive.google.com/uc?id={id}&export=download'
    gdown.download(url, str(target_path), quiet=False)


def download_youtube_video(youtube_id, target_path="./youtube", size=None):
    """
    Downloads the best available format of a YouTube video and saves it to the specified path with the inferred extension.

    Args:
        youtube_id (str): The YouTube video ID.
        target_path (str): The target path without an extension.

    Returns:
        str: The full file path including the inferred extension.
    """
    ydl_opts = {
        'outtmpl': f'{target_path}.%(ext)s',
        'format': 'best',
        'noprogress': True,
        'overwrites': True
    }
    if size:
        ydl_opts['format'] = f'best[height<={size}]'

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f'https://www.youtube.com/watch?v={youtube_id}', download=True)
            ext = ydl.prepare_filename(info).split('.')[-1]  # Infer the extension
    except:
        print("Requested size is unavailable. Trying the best available format.")
        del ydl_opts['format']
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f'https://www.youtube.com/watch?v={youtube_id}', download=True)
                ext = ydl.prepare_filename(info).split('.')[-1]  # Infer the extension
        except:
            return None

    full_path = f"{target_path}.{ext}"
    return full_path


def normalize(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)

def detach_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        x.requires_grad = False
        x = x.cpu()
        x = x.numpy()
    return x
