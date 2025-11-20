from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import sys
import os
import glob

sys.path.append('/Users/sydneythiessen/ast/src/models')
from ast_models import ASTModel



class DysarthriaDataset(Dataset):
    """
    Custom Dataset for loading mel spectrograms for dysarthria detection.
    
    Classes:
    5 - Healthy
    4 - ALS without Dysarthria
    3 - Mild Dysarthria
    2 - Moderate Dysarthria
    1 - Severe Dysarthria
    """

    def __init__(self, spectrogram_dir, label_file, target_time = 1024, target_freq = 128, use_phonation = 'all'):
        """ 
        Parameters:
        spectrogram_dir (str): Directory containing mel spectrogram .npy files.
        label_file : str
            Path to CSV file mapping speaker/file to severity label
            Format: filename,severity_label or speaker_id,severity_label
        use_phonation: str or list: Specifies which phonation types to include.
        """
        self.target_time = target_time
        self.target_freq = target_freq
        self.data = []
        self.labels = []

        all_phonations = ['phonationA', 'phonationE', 'phonationI', 
                         'phonationO', 'phonationU', 'rhythmKA', 
                         'rhythmPA', 'rhythmTA']
        
        if use_phonation == 'all':
            self.phonations_to_use = all_phonations
        elif isinstance(use_phonation, list):
            self.phonations_to_use = use_phonation

        self.label_map = self._load_labels(label_file)

        #load all spectrograms from phonation directories
        for phonation_type in self.phonations_to_use:
            phonation_dir = os.path.join(spectrogram_dir, phonation_type)

            spec_files = glob.glob(os.path.join(phonation_dir, '*.npy'))
            print(f"Found {len(spec_files)} files in {phonation_dir}")

            for spec_file in spec_files:
                identifier = self._get_identifier(spec_file)
                if identifier in self.label_map:
                    self.data.append(spec_file)
                    self.labels.append(self.label_map[identifier])
                else:
                    print(f"Warning: No label found for {identifier}, skipping.")

        print(f"\n=== Dataset Summary ===")
        print(f"Total samples: {len(self.data)}")
        print(f"Class distribution:")
        for severity in range(1, 6):
            count = self.labels.count(severity)
            severity_name = self._get_severity_name(severity)
            print(f"  Class {severity} ({severity_name}): {count} samples")

    def __len__(self):
        return len(self.data)

    def _load_labels(self, label_file):
        label_map = {}
        with open(label_file, 'r') as f:
            header = f.readline().strip()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 2:
                    continue
                identifier, label = parts
                label_map[identifier] = int(label)
        return label_map
    
    def _get_identifier(self, filepath):
        filename = os.path.basename(filepath)
        filename = filename.replace('.npy', '')

        if '_' in filename:
            identifier = filename.rsplit('_', 1)[0]

        else:  
            identifier = filename
        return identifier

    def _get_severity_name(self, label):
        """Convert numeric label to severity name"""
        severity_names = {
            5: "Healthy",
            4: "ALS without dysarthria",
            3: "Mild dysarthria",
            2: "Moderate dysarthria",
            1: "Severe dysarthria"
        }
        return severity_names.get(label, "Unknown")
    
    def __getitem__(self, idx):
        # Load spectrogram
        spec = np.load(self.data[idx])
        
        # Transpose from (freq, time) to (time, freq)
        # Your saved format: (128, 1024) -> need (1024, 128)
        spec = spec.T
        
        # Adjust shape to match AST requirements
        spec = self._adjust_shape(spec)
        
        # Convert to tensor
        spec_tensor = torch.from_numpy(spec).float()
        
        # Get label (subtract 1 to make it 0-indexed if needed)
        label = self.labels[idx] - 1  # Convert 1-5 to 0-4 for PyTorch
        
        return spec_tensor, label
    
    def _adjust_shape(self, spec):
        """Pad or trim spectrogram to target shape"""
        # Handle time dimension
        if spec.shape[0] < self.target_time:
            pad_amount = self.target_time - spec.shape[0]
            spec = np.pad(spec, ((0, pad_amount), (0, 0)), mode='constant')
        else:
            spec = spec[:self.target_time, :]
        
        # Handle frequency dimension  
        if spec.shape[1] < self.target_freq:
            pad_amount = self.target_freq - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_amount)), mode='constant')
        else:
            spec = spec[:, :self.target_freq]
        
        return spec

if __name__ == "__main__":

    spectrogram_path = '/Users/sydneythiessen/Documents/GitHub/MacewanVoices/task1_test/mel_spectrograms_test'
    label_file = '/Users/sydneythiessen/Documents/GitHub/MacewanVoices/AST_Model/task1_test/file_labels_test.csv'

    dataset = DysarthriaDataset(spectrogram_path, label_file, use_phonation='all')

        # Test loading one sample
    spec, label = dataset[0]
    print(f"Spectrogram shape: {spec.shape}")
    print(f"Label: {label}")

# train_loader = DataLoader(dataset, 
#                           batch_size=8, 
#                           shuffle=True, 
#                           num_workers=0)

# ast_model = ASTModel(
#     label_dim=5,
#     fstride=10,
#     tstride=10,
#     input_fdim=128,
#     input_tdim=1024,
#     imagenet_pretrain=True,
#     audioset_pretrain=False,
#     model_size='base384'
# )

# print("\nTesting data loader and model forward pass...")
# for batch_specs, batch_labels in train_loader:
#     print(f"Batch spectrograms shape: {batch_specs.shape}")  # Expect (batch_size, time, freq)
#     print(f"Batch labels shape: {batch_labels.shape}")        # Expect (batch_size, 1024, 128)
    
#     ast_model.eval()
#     with torch.no_grad():
#         outputs = ast_model(batch_specs)
#         print(f"Model outputs shape: {outputs.shape}")  # Expect (batch_size, 5)

#         #get predictions
#         predictions = torch.softmax(outputs, dim=1)
#         predicted_classes = torch.argmax(predictions, dim=1)
#         print(f"Predicted classes: {predicted_classes}")
#         print(f"Prediction probabilities: {predictions}")
    
#     break  # Just test one batch