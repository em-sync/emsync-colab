from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
import utils as u
from . import video_models

class VideoProcessor:
    def __init__(self, model_dir):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.size = 360  # Use 360p
        self.model_dir = Path(model_dir)
        self.labels = ('anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise')
        self.video_project_dir = Path(__file__).resolve().parents[1]
        
        # Load model config
        self.config = torch.load(self.model_dir / 'model_config.pt', weights_only=False)
        
        # Initialize classifier
        self.classifier = video_models.AttentionClassifier(
            feature_dims=self.config['feature_dims'],
            d_model=self.config['d_model'],
            d_output=self.config['n_labels'],
            dropout=self.config['dropout']
        )
        
        # Load classifier weights
        weights_fp = self.model_dir / 'video_model.pt'
        if not weights_fp.exists():
            id = '11aaWCbhcccW_lTaUqKqux2h-nNAfIYCi'
            u.download_gdrive(id, weights_fp)
        weights = torch.load(weights_fp, map_location='cpu', weights_only=True)
        self.classifier.load_state_dict(weights)
        del weights
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        # Load feature information
        self.feature_names = self.config['features']
        self.n_frames = self.config['n_frames']
        self.stats = torch.load(self.video_project_dir / 'weights_and_labels' / 'stats.pt', weights_only=False)
        
        # Initialize feature extractors
        self.extractors = {}
        features_in_gpu = ('asr_sentiment', 'beats', 'clip', 'face_emotion', 'ocr_sentiment')
        for feature_name in self.feature_names:
            if feature_name == 'clip':
                self.extractors[feature_name] = video_models.CLIPRunner()
            elif feature_name == 'beats':
                self.extractors[feature_name] = video_models.BEATSRunner()
            elif feature_name == 'asr_sentiment':
                self.extractors[feature_name] = video_models.ASRSentiment()
            elif feature_name == 'ocr_sentiment':
                self.extractors[feature_name] = video_models.OCRPipeline()
            elif feature_name == 'face_emotion':
                self.extractors[feature_name] = video_models.FaceExtractAndClassify()
            
            if feature_name in features_in_gpu:
                self.extractors[feature_name].to_device(self.device)
        
        print('Extracting frames and audio...', end=' ')
    
    def run(self, video_path, visualize=False):
        scenecuts = u.ffmpeg_scene_detect(video_path)
        duration = u.get_video_duration(video_path)
        save_memory = False
        
        video_duration = u.get_video_duration(video_path)
        input_frame_sequence_length = self.config['feature_lengths']['clip']
        output_fps = input_frame_sequence_length / video_duration
        
        input_frames, input_audio, input_fps, sr = u.extract_frames_and_audio(
            str(video_path), output_fps=output_fps, size=self.size
        )
        
        print('Done.')
        
        # Feature extraction
        sample_features = {}
        video_outputs = {}
        clip_features = None
        
        with torch.no_grad():
            for feature_name in tqdm(self.feature_names, desc='Extracting features'):
                if feature_name in ('clip', 'ocr_sentiment', 'face_emotion'):
                    input_tensor = input_frames
                else:
                    input_tensor = input_audio
                
                video_output = self.extractors[feature_name].process_video(
                    input_tensor=input_tensor, n_frames=self.n_frames, sr=sr
                )
                extracted_feature = video_output['features']
                
                if extracted_feature == []:
                    extracted_feature = torch.zeros((
                        self.config['feature_lengths'][feature_name],
                        self.config['feature_dims'][feature_name]
                    ))
                elif self.config['normalize_data']:
                    extracted_feature = u.normalize(
                        extracted_feature, self.stats[feature_name]['min'], self.stats[feature_name]['max']
                    )
                
                extracted_feature = extracted_feature.unsqueeze(0).to(self.device)
                sample_features[feature_name] = extracted_feature
        
        # Run classifier
        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            output = self.classifier(sample_features).squeeze()
            output = torch.nn.functional.softmax(output)
            output = u.detach_tensor(output)
        
        sorted_indices = np.argsort(output)[::-1]
        print('\nEmotion prediction: ')
        for idx in sorted_indices:
            print(f"{round(output[idx] * 100)}%  {self.labels[idx].capitalize()}")
        
        return {'predictions': output, 'duration': duration, 'scenecuts': scenecuts}
