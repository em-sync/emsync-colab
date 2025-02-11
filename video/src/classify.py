from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
import utils as u
from . import video_models

def run_video(model_dir, video_path=None, visualize=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    size = 360    # Use 360p

    scenecuts = u.ffmpeg_scene_detect(video_path)
    duration = u.get_video_duration(video_path)

    save_memory = False

    labels = ('anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise')

    video_project_dir = Path(__file__).resolve().parents[1]

    model_dir = Path(model_dir)
    config = torch.load(model_dir / 'model_config.pt', weights_only=False)

    classifier = video_models.AttentionClassifier(
            feature_dims=config['feature_dims'],
            # n_layers=config['n_layers'],
            d_model=config['d_model'],
            d_output=config['n_labels'],
            dropout=config['dropout'],
            )
    weights_fp = model_dir / 'video_model.pt'

    if not weights_fp.exists():
        id = '11aaWCbhcccW_lTaUqKqux2h-nNAfIYCi'
        u.download_gdrive(id, weights_fp)
    weights = torch.load(weights_fp, map_location=lambda storage, weights_only=True, loc: storage)
    classifier.load_state_dict(weights)
    del weights
    classifier = classifier.to(device)
    classifier.eval()

    feature_names = config['features']

    n_frames = config['n_frames']

    video_project_dir = Path(__file__).resolve().parents[1]
    stats = torch.load(video_project_dir / 'weights_and_labels' / 'stats.pt', weights_only=False)

    extractors = {}

    # Colab doesn't have large RAMs for system and GPU.
    # So store some of the models in the GPU to open up some system RAM.
    # Store the rest on the CPU and move to GPU when needed.
    features_in_gpu = ('asr_sentiment', 'beats', 'clip', 'face_emotion', 'ocr_sentiment')
    if save_memory:
        features_in_gpu.remove('ocr_sentiment')

    for feature_name in feature_names:
        if feature_name == 'clip':
            extractors[feature_name] = video_models.CLIPRunner()
        elif feature_name == 'beats':
            extractors[feature_name] = video_models.BEATSRunner()
        elif feature_name == 'asr_sentiment':
            extractors[feature_name] = video_models.ASRSentiment()
        elif feature_name == 'ocr_sentiment':
            extractors[feature_name] = video_models.OCRPipeline()
        elif feature_name == 'face_emotion':
            extractors[feature_name] = video_models.FaceExtractAndClassify()

        if feature_name in features_in_gpu:
            extractors[feature_name].to_device(device)

    if visualize:
        caption_model = video_models.CaptionRunner()

    print('Extracting frames and audio...', end=' ')

    # The visual models use a fixed number of frames (16).
    # So adjust FPS to extract only that many frames.
    video_duration = u.get_video_duration(video_path)
    input_frame_sequence_length = config['feature_lengths']['clip']
    output_fps = input_frame_sequence_length / video_duration

    input_frames, input_audio, input_fps, sr = u.extract_frames_and_audio(str(video_path), output_fps=output_fps, size=size)

    print('Done.')

    sample_features = {}
    video_outputs = {}
    clip_features = None    # We will use it later for captioning

    # PART II - FEATURE EXTRACTION
    with torch.no_grad():
        for feature_name in tqdm(feature_names, desc='Extracting features'):
            if feature_name == 'clip':
                input_tensor = input_frames
            elif feature_name == 'beats':
                input_tensor = input_audio
            elif feature_name == 'asr_sentiment':
                input_tensor = input_audio
            elif feature_name == 'ocr_sentiment':
                input_tensor = input_frames
            elif feature_name == 'face_emotion':
                input_tensor = input_frames

            if feature_name not in features_in_gpu:
                extractors[feature_name].to_device(device)     # Move to GPU if available

            video_output = extractors[feature_name].process_video(input_tensor=input_tensor, n_frames=n_frames, sr=sr)
            
            if save_memory:
                if feature_name not in features_in_gpu:
                    extractors[feature_name].to_device('cpu')     # Move back to CPU to open up space

            extracted_feature = video_output['features']

            if feature_name == 'clip':
                clip_features = extracted_feature

            # Keep other output for visualization
            video_outputs[feature_name] = video_output

            if extracted_feature == []:     # Feature unavailable, use zeros
                extracted_feature = torch.zeros((config['feature_lengths'][feature_name],
                    config['feature_dims'][feature_name]))
            elif config['normalize_data']:      # Normalization
                extracted_feature = u.normalize(extracted_feature, stats[feature_name]['min'], stats[feature_name]['max'])

            # Fix lengths
            source_length = extracted_feature.shape[0]
            target_length = config['feature_lengths'][feature_name]

            if source_length > target_length:     # Take equidistant frames
                inds = u.equidistant_indices(source_length, target_length)
                extracted_feature = extracted_feature[inds, :]
            elif source_length < target_length:
                extracted_feature = torch.nn.functional.pad(extracted_feature, (0, 0, 0, target_length - source_length))

            # Add batch dimension and move to device
            extracted_feature = extracted_feature.unsqueeze(0).to(device)

            sample_features[feature_name] = extracted_feature

        # PART III - RUN MODEL
        with torch.cuda.amp.autocast(enabled=config['amp']):
            output = classifier(sample_features).squeeze()
            del sample_features
            output = torch.nn.functional.softmax(output)
            output = u.detach_tensor(output)


    # PART IV - VISUALIZATION (OPTIONAL)
    if visualize:

        # # *** FACE and OCR
        # Display a sample frame with OCR and facial boxes (later)
        if 'ocr_sentiment' in video_outputs.keys():
            ocr_boxes = video_outputs['ocr_sentiment']['boxes']
        else:
            ocr_boxes = []
        
        face_boxes = video_outputs['face_emotion']['coordinates']

         # Find the frame largest face
        i = u.find_largest_box(face_boxes)

        if ocr_boxes != []:
            if ocr_boxes == None or ocr_boxes[i] == None: 
                ocr_boxes = []
            else:
                # Get OCR boxes for that frame
                ocr_boxes = [pred[0] for pred in ocr_boxes[i]]

        # We don't need predictions per frame, we will print OCR from the entire video
        ocr_predictions = [None] * len(ocr_boxes)

        if face_boxes[i] == None:
            face_boxes[i] = []

        face_boxes = [u.convert_to_points(box) for box in face_boxes[i]]
        face_predictions = video_outputs['face_emotion']['predictions'][i]
        if face_predictions == None:
            face_predictions = []

        face_predictions = [f'{round(pred[1] * 100):2}%  {pred[0].upper()}' for pred in face_predictions]

        boxes = face_boxes + ocr_boxes
        box_labels = face_predictions + ocr_predictions
        selected_frame = input_frames[i]

        # *** CLIP caption
        # Using the same sample frame, create a caption using CLIP
        clip_feature = clip_features[i:i+1, ...].to(device)
        with torch.no_grad():
            caption_model.to_device(device)   # Move to GPU if exists
            caption = caption_model(clip_feature).capitalize()
            if save_memory:
                caption_model.to_device('cpu')

        # *** Optical Character Recognition (OCR) analysis
        if 'ocr_sentiment' in video_outputs.keys():
            ocr_text = video_outputs['ocr_sentiment']['ocr_processed']
            if ocr_text != []:
                ocr_text = '. '.join(video_outputs['ocr_sentiment']['ocr_processed'])
                sentiment_prediction = video_outputs['ocr_sentiment']['predictions'][0].capitalize()
                sentiment_percentage = round(video_outputs['ocr_sentiment']['predictions'][1] * 100)
                print()
                print('Optical character recognition (OCR):')
                print(u.wrap_text(ocr_text, 72))
                print('Sentiment analysis:')
                print(f"{sentiment_percentage}%  {sentiment_prediction}")

        # *** Automatic Speech Recognition (ASR) analysis
        asr_text = video_outputs['asr_sentiment']['asr']
        asr_language = video_outputs['asr_sentiment']['language']
        if asr_text != "":
            sentiment_prediction = video_outputs['asr_sentiment']['predictions'][0].capitalize()
            sentiment_percentage = round(video_outputs['asr_sentiment']['predictions'][1] * 100)
            print()
            print('Automatic speech recognition (ASR):')
            if asr_language not in ('english', None):
                print(f'(Translated from {asr_language.capitalize()}.)')
            print(u.wrap_text(asr_text, 72))
            print('Sentiment analysis:')
            print(f"{sentiment_percentage}%  {sentiment_prediction}")

        # *** BEATS classification
        # Using the entire video, print BEATS (audio event) probabilities
        beats_predictions = video_outputs['beats']['predictions']

        if beats_predictions != []:
            print()
            print('Audio classification:')
            best_prediction = beats_predictions[0]
            print(f"{round(best_prediction[1] * 100)}%  {best_prediction[0]}")
            for prediction in beats_predictions[1:]:
                print(f'{round(prediction[1] * 100)}%  {prediction[0]}')
            print()

        # *** FACE and OCR
        # Display a sample frame with OCR and facial boxes
        no_display = matplotlib.get_backend() == 'agg'
        title = 'Predicted caption: ' + caption
        if no_display:
            output_file = 'input_output/sample_frame.png'
        else:
            output_file = None
            print('Sample frame')
        u.draw_boxes_on_image(selected_frame, boxes, labels=box_labels, output_file=output_file, title=title)

    # Print emotion prediction results
    sorted_indices = np.argsort(output)[::-1]

    print()
    print('Emotion prediction: ')
    for idx in sorted_indices:
        print(f"{round(output[idx] * 100)}%  {labels[idx].capitalize()}")

    return {'predictions': output, 'duration': duration, 'scenecuts': scenecuts}

