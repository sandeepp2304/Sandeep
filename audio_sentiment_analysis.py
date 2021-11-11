import pandas as pd
import numpy as np
import os
import wave
import re
import pickle
from pydub import AudioSegment
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy import signal
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import enums
from google.cloud.speech_v1p1beta1 import types
from google.cloud import storage
from config import Config
import time


# credentials/key to activate google cloud and google speech recognition api's
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "*/webmine-91e3757ad733.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/python5/PycharmProjects/audio_sentiment_analysis/modules/webmine-dd1c30ba9324.json"


class AudioSentimentAnalysis:

    def remove_mp3(self):
        '''
        Remove mp3 from the folder
        '''
        if self.file_name.endswith(".mp3"):
            os.remove(self.file_name)

    def remove_wav(self):
        '''
        Remove wav from the folder
        '''
        if self.file_name.endswith(".wav"):
            os.remove(self.file_name)

    def mp3_to_wav(self):
        '''
        Convert mp3 to wav format
        '''
        name, ext = os.path.splitext(self.file_name)
        if ext == ".mp3":
            mp3_sound = AudioSegment.from_mp3(self.file_name)
            # rename them using the old name + ".wav"
            mp3_sound.export("{0}.wav".format(name), format="wav")
        self.remove_mp3()

    def frame_rate_channel(self):
        '''
        Get the frame rate and the number of channels
        '''
        with wave.open(self.file_name, "rb") as wave_file:
            frame_rate = wave_file.getframerate()
            channels = wave_file.getnchannels()
            return frame_rate, channels

    def stereo_to_mono(self):
        '''
        Converting stereo audio to mono,because google speech recognition api can't understand stereo files
        '''
        sound = AudioSegment.from_wav(self.file_name)
        sound = sound.set_channels(1)
        sound.export(self.file_name, format="wav")

    def high_low_pass(self):
        '''
        Audio_preprocessing using High pass and Low pass band filters and filtering the noice
        '''
        (Frequency, array) = read(self.file_name)  # Reading the sound file.
        b, a = signal.butter(5, 1000 / (Frequency / 2), btype='highpass')  # ButterWorth filter 4350
        filtered_signal = signal.lfilter(b, a, array)
        c, d = signal.butter(5, 380 / (Frequency / 2), btype='lowpass')  # ButterWorth low-filter
        new_filtered_signal = signal.lfilter(c, d, filtered_signal)  # Applying the filter to the signal
        write(self.file_name, Frequency, np.int16(new_filtered_signal / np.max(np.abs(new_filtered_signal)) * 44100))

    def upload_blob(self, bucket_name, source_file_name, destination_blob_name):
        '''
        Uploads a file to the bucket/google cloud.
        '''
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

    def delete_blob(self, bucket_name, blob_name):
        '''
        Deletes a blob from the bucket/google cloud .
        '''
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()

    def google_transcribe(self):
        '''
        Transfering audio file from local disc to google cloud and then using google speech recognition api
        converting audio file to text data
        '''
        # file_name = self.file_path + audio_file_name
        #         print("Pass only wav format audio files in a folder,NO MP3...etc")
        #         self.mp3_to_wav(file_name)
        #         print("Audio_file_Name:",audio_file_name)

        # The name of the audio file to transcribe

        frame_rate, channels = self.frame_rate_channel()

        if channels == 1:
            pass
        else:
            self.stereo_to_mono()
        self.high_low_pass()
        bucket_name = 'mlaudiofiles12'

        source_file_name = self.file_name
        destination_blob_name = source_file_name.split('/')[-1]

        self.upload_blob(bucket_name, source_file_name, destination_blob_name)

        gcs_uri = 'gs://mlaudiofiles12/' + destination_blob_name
        transcript = ''

        client = speech.SpeechClient()
        audio = types.RecognitionAudio(uri=gcs_uri)

        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=frame_rate,
            language_code='hi_In')  # specifing the language

        # Detects speech in the audio file
        operation = client.long_running_recognize(config, audio)
        response = operation.result(timeout=10000)

        for result in response.results:
            transcript += result.alternatives[0].transcript

        self.delete_blob(bucket_name, destination_blob_name)
        #         self.remove_wav(audio_file_name)
        #         print("Speech_Text_Done")
        return transcript

    def load_model_text(self):
        '''
        loading the model.
        :param path: path to your clf model.
        '''
        self.loaded_model_text = pickle.load(
            open("/home/python5/PycharmProjects/audio_sentiment_analysis/modules/Hindi_Sentiment_Analysis_BOG_NB_1.clf",
                 'rb'))
        return self.loaded_model_text

    def makepredictions_text(self, preprocessed_tweet):
        '''
        Process the files and get the Sentiment and their Probability.
        '''
        loaded_model_text = self.load_model_text()
        if len(preprocessed_tweet) == 0:
            print("No_Voice")
        else:
            predictions = loaded_model_text.predict([preprocessed_tweet])
            probability = loaded_model_text.predict_proba([preprocessed_tweet])
            #             print("Prediction/Sentiment :", " ", predictions)
            # print(predictions)
            predictions = {"prediction":predictions[0]}
            return predictions

    #             print("Probability Ratio for Negative,Neutral,Positive Respectively :"," ",probability)

    def preprocessing_tweet(self, transcript):
        '''
        preprocessing the string
        '''
        self.transcript = transcript
        stopword_list = []

        # get custom stopwords from a file (pt-br). You can create your own database of stopwords on a text file
        df = pd.read_fwf('/home/python5/PycharmProjects/audio_sentiment_analysis/modules/hindi_stopwords.txt', header=None)

        # list of array
        custom_stopwords = df.values.tolist()

        # transform list of array to list
        custom_stopwords = [s[0] for s in custom_stopwords]

        # You can also add stopwords manually instead of loading from the database.
        # Generally, we add stopwords that belong to this context.
        stopword_list.append('...')
        stopword_list.append('«')
        stopword_list.append('➔')
        stopword_list.append('|')
        stopword_list.append('»')

        # join all stopwords
        stopword_list.extend(custom_stopwords)

        # remove duplicate stopwords (unique list)
        stopword_list = list(set(stopword_list))
        #             print("Conversation_Begins & Gives Sentiment For Entire Conversation ")

        # remove the special characters
        tweet = re.sub(r'[?|!|\'|"|#]', r'', str(transcript))
        tweet = re.sub(r'[.|,|)|(|\|/]', r' ', tweet)
        # removing the digits
        tweet = re.sub(" \d+", " ", tweet)
        # remove urls from text
        tweet = re.sub(r'http\S+', r' ', tweet)
        # replace multiple spaces with single space
        tweet = re.sub('\s+', ' ', tweet)
        tweet = tweet.strip()
        tweet = tweet.split()
        # #lower the words
        # tweet = tweet.lower()
        #remove the stop words
        tweet = [word for word in tweet if not word in stopword_list]
        tweet= ' '.join(tweet)
        # print("Total_Conversation:\n",self.transcript)
        # print(tweet)
        try:
            response = self.makepredictions_text(tweet)
        # response = json.dumps(response)
            print(response)
            print(type(response))
            return response
        except:
            return {'message': 'file not found'}
        # self.remove_wav(audio_file_name)

    def main(self, filename):
        # localFilename = '/tmp/{}'.format(os.path.basename(key))
        # s3.download_file(Bucket=bucket, Key=key, Filename=localFilename)
        # inFile = open(localFilename, "r")
        start_time = time.time()
        self.file_name = Config.File_path+filename
        '''
        main loop from where all methods are calling
        '''
        # self.mp3_to_wav(filename)
        transcript = self.google_transcribe()
        response = self.preprocessing_tweet(transcript)
        print("--- %s seconds ---" % (time.time() - start_time))
        return response


if __name__ == "__main__":
    obj = AudioSentimentAnalysis()
    obj.main("Learn Hindi.wav")
