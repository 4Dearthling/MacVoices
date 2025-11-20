"Built from the tutorial: https://www.youtube.com/watch?v=O04v3cgHNeM"

import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

class Loader:
    #Load an audio file and return the audio time series and sampling rate.

    def __init__(self, sample_rate, duration, mono):
        """
        Parameters
        ----------
        sample_rate : int
            Target sampling rate.
            duration : float
            Duration (in seconds) to load.
            mono : bool
            Convert signal to mono.
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono


    def load(self, file_path):
        """
        Parameters
        ----------
        file_path : str
            Path to the input file.

        Returns
        -------
        y : np.ndarray [shape=(n,)]
            Audio time series.
        sr : number > 0 [scalar]
            Sampling rate of `y`.
        """
        signal = librosa.load(file_path, sr=self.sample_rate, duration=self.duration, mono=self.mono)[0]
        return signal
    

class Padder:
    """
    Responsible to apply padding to an audio signal to ensure a fixed length.
    """
    def __init__(self, mode = "constant"):
        """
        Parameters
        ----------
        mode : str
            Padding mode to use. Default is "constant".
        """
        self.mode = mode


    def left_pad(self, array, num_missing_items):
        """
        Parameters
        ----------
        array : np.ndarray
            Input array to pad.
        num_missing_items : int
            Number of items to pad.

        Returns
        -------
        padded_array : np.ndarray
            Padded array.
        """
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)
        return padded_array
    
    def right_pad(self, array, num_missing_items):
        """
        Parameters
        ----------
        array : np.ndarray
            Input array to pad.
        num_missing_items : int
            Number of items to pad.

        Returns
        -------
        padded_array : np.ndarray
            Padded array.
        """
        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
        return padded_array
    
class LogSpectrogramExtractor:
    "Extracts log spectrograms (in DB) from a time-series signal"

    def __init__(self, frame_size, hop_length, max_freq_bins=None):
        """
        Parameters
        ----------
        frame_size : int
            Length of the FFT window.
        hop_length : int
            Number of samples between successive frames.
        max_freq_bins : int or None
            Maximum number of frequency bins to retain. If None, retain all.
        """
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.max_freq_bins = max_freq_bins

    def extract(self, signal):
        """
        Parameters
        ----------
        signal : np.ndarray
            Input audio time series.

        Returns
        -------
        log_spectrogram : np.ndarray
            Log spectrogram in decibels.
        """
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length) [:-1]

               # Apply frequency bin truncation
        if self.max_freq_bins is not None:
            stft = stft[:self.max_freq_bins]
        else:
            stft = stft[:-1]  # Original behavior - remove last bin

        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram

class MelSpectrogramExtractor:
    """Extracts mel spectrograms from a time-series signal"""

    def __init__(self, sample_rate, n_fft, hop_length, n_mels=128):
        """
        Parameters
        ----------
        sample_rate : int
            Audio sampling rate
        n_fft : int
            Length of the FFT window.
        hop_length : int
            Number of samples between successive frames.
        n_mels : int
            Number of mel frequency bins (AST uses 128 by default)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def extract(self, signal):
        """
        Parameters
        ----------
        signal : np.ndarray
            Input audio time series.

        Returns
        -------
        mel_spectrogram : np.ndarray
            Mel spectrogram in decibels, shape (n_mels, time_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=signal, 
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            window = 'hamming',
            center = True,
            pad_mode = 'reflect'
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    

class MinMaxNormalizer:
    """Applies Min-Max normalization to an array"""

    def __init__(self, min_val, max_val):
        """
        Parameters
        ----------
        min_val : float
            Minimum value for normalization.
        max_val : float
            Maximum value for normalization.
        """
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, array):
        """
        Parameters
        ----------
        array : np.ndarray
            Input array to normalize.
        Returns
        -------
        normalized_array : np.ndarray
            Normalized array.
        """
        array_min = np.min(array)
        array_max = np.max(array)

        normalized_array = (array - array_min) / (array_max - array_min) # Scale to [0, 1]
        normalized_array = normalized_array * (self.max_val - self.min_val) + self.min_val  # Scale to [min_val, max_val]
        return normalized_array
    
    def denormalize(self, normalized_array, original_min, original_max):
        """
        Parameters
        ----------
        normalized_array : np.ndarray
            Normalized array to denormalize.
        original_min : float
            Original minimum value before normalization.
        original_max : float
            Original maximum value before normalization.
        Returns
        -------
        denormalized_array : np.ndarray
            Denormalized array.
        """
        denormalized_array = (normalized_array - self.min_val) / (self.max_val - self.min_val)
        denormalized_array = denormalized_array * (original_max - original_min) + original_min
        return denormalized_array

class ASTNormalizer:
    """Applies AST-style normalization (mean/std)"""

    def __init__(self, use_audioset_stats=True):
        """
        Parameters
        ----------
        use_audioset_stats : bool
            If True, use AudioSet mean/std. Otherwise compute from data.
        """
        self.use_audioset_stats = use_audioset_stats
        self.mean = None
        self.std = None

    def normalize(self, spectrogram):
        """
        Parameters
        ----------
        spectrogram : np.ndarray
            Input spectrogram to normalize.
        
        Returns
        -------
        normalized_spectrogram : np.ndarray
            Normalized spectrogram.
        """
        if self.use_audioset_stats:
            # AudioSet normalization from AST repo
            normalized = (spectrogram + 4.26) / (4.57 * 2)
        else:
            # Compute mean/std from the spectrogram
            if self.mean is None or self.std is None:
                self.mean = np.mean(spectrogram)
                self.std = np.std(spectrogram)
            normalized = (spectrogram - self.mean) / (self.std + 1e-8)
        
        return normalized

class Saver:
    """
    Responsible to save features, and the min max values
    """

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def _generate_save_path(self, file_path):
        directory = os.path.basename(os.path.dirname(file_path)) 
        file_name = os.path.split(file_path)[1]
        save_directory = os.path.join(self.feature_save_dir, directory)

        os.makedirs(save_directory, exist_ok=True)

        save_path = os.path.join(save_directory, file_name[:-4] + ".npy")
        return save_path
    
    def save_min_max_values(self, min_max_values):
        os.makedirs(self.min_max_values_save_dir, exist_ok=True)
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")

        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

class Viewer:
    """
    Responsible to visualize spectrograms.
    """
    def __init__(self, viewer_dir):
        self.viewer_dir = viewer_dir

    def save_spectrogram_image(self, file_path):
        save_path = self._generate_save_path(file_path)

        file_name = os.path.split(save_path)[1]
        spectrogram = np.load(file_path)

        plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label="Log Amplitude")
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency Bins")
        plt.title(f"Log Spectrogram for {file_name[:-4]}")
        plt.savefig(save_path)
        plt.clf()

    def _generate_save_path(self, file_path):
        directory = os.path.basename(os.path.dirname(file_path)) 
        file_name = os.path.split(file_path)[1]
        save_directory = os.path.join(self.viewer_dir, directory)

        os.makedirs(save_directory, exist_ok=True)

        save_path = os.path.join(save_directory, file_name[:-4] + ".png")
        return save_path


class PreprocessingPipeline:
    """
    Processes audio files in a directory, applying the following steps to each file:
    
        1 - load a file
        2 - pad the signal (if necessary)
        3 - extract log spectrogram from signal
        4 - normalize spectrogram
        5 - save the normalized spectrogram
    
    Storing the min max values for all the log spectrograms. (used to reconstruct the signal)
    """

    def __init__(self):
        self._loader = None
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.viewer = None
        self.min_max_values = {}
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader
    

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)


    def process(self, audio_files_dir):
        for root, dirs, files in os.walk(audio_files_dir):
            if root == audio_files_dir:
                continue

            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")
        
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self.viewer.save_spectrogram_image(save_path)
        self._store_min_max_value(save_path, feature.min, feature.max)

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False
    
    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    
    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 160
    DURATION = 10.24 # in secs
    SAMPLE_RATE = 16000
    MONO = True
    N_MELS = 128

    script_dir = os.path.dirname(os.path.abspath(__file__))

    SPECTROGRAMS_SAVE_DIR = os.path.join(script_dir, "task1_test/mel_spectrograms/")
    MIN_MAX_VALUES_SAVE_DIR = os.path.join(script_dir, "task1_test/min_max_values/")
    VIEWER_DIR = os.path.join(script_dir, "task1_test/spectrogram_images/")
    FILES_DIR = os.path.join(script_dir, "task1_test/test/")

    #instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    mel_spectrogram_extractor = MelSpectrogramExtractor(SAMPLE_RATE, FRAME_SIZE, HOP_LENGTH, N_MELS)
    ast_normalizer = ASTNormalizer(use_audioset_stats=True)
    #min_max_normalizer = MinMaxNormalizer(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
    viewer = Viewer(VIEWER_DIR)
    
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = mel_spectrogram_extractor
    preprocessing_pipeline.normalizer = ast_normalizer
    #preprocessing_pipeline.extractor = log_spectrogram_extractor
    #preprocessing_pipeline.normalizer = min_max_normalizer
    preprocessing_pipeline.saver = saver
    preprocessing_pipeline.viewer = viewer

    preprocessing_pipeline.process(FILES_DIR)

