import librosa
import numpy
import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd
import time
SAVED_MODEL_PATH = "home_model.h5"
SAMPLES_TO_CONSIDER = 22050
fs = 44100  # Sample rate
seconds = 3  # Duration of recording
AUDIO_FILE_PATH = "test/voice.wav"
class _Keyword_Spotting_Service:

    """Singleton class for keyword spotting inference with trained models.

    :param model: Trained model
    """
    model = None

    _mapping = [
                   "all off",
                   "all on",
                   "bathroom light off",
                   "bathroom light on",
                   "bedroom light off",
                   "bedroom light on",
                   "kitchenroom light off",
                   "kitchenroom light on",
                   "noise",
                   "restroom light off",
                   "restroom light on",
                   "salonroom light off",
                   "salonroom light on"
    ]

    _instance = None


    def predict_keyword(self, file_path):
        """

        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.

        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples

        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.

    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()

        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance




if __name__ == "__main__":
    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()
    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1
    text = kss1.predict_keyword(AUDIO_FILE_PATH)
    print("You said:{}".format(text))
    # Take the input from the microphone

    """try:
        while True:
            print("Say Something......")
            myrecording = sd.rec( int( seconds * fs ), samplerate=fs, channels=2 )
            sd.wait()  # Wait until recording is finished
            write( AUDIO_FILE_PATH, fs, myrecording )  # Save as WAV file

            time.sleep(3)

            text = kss.predict_keyword(AUDIO_FILE_PATH)

            print("You said: {}".format(text))
    except KeyboardInterrupt:
        pass"""












