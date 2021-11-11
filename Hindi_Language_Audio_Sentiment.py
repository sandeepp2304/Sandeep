#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:02:31 2019

@author: ml
"""
# Import the packages

import pandas as pd
import numpy as np
from pydub import AudioSegment
import io
import os
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import enums
from google.cloud.speech_v1p1beta1 import types
import wave
from google.cloud import storage
import subprocess
import librosa
import soundfile as sf
import wave
import re
import keras
import librosa
import pickle
import json
from pydub import AudioSegment
import nltk
from scipy.io.wavfile import read
from scipy.io.wavfile import write     # Imported libaries such as numpy, scipy(read, write), matplotlib.pyplot
from scipy import signal
# credentials/key to activate google cloud and google speech recognition api's
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ml/GOOGLE_SD_KEY/webmine-91e3757ad733.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ml/GOOGLE_CL_KEY/webmine-dd1c30ba9324.json"

class AI:
                
    def remove_mp3(self, filepath):
        '''
        Remove mp3 from the folder
        '''
        location = os.listdir(filepath)
        for item in location:
            if item.endswith(".mp3"):
                os.remove(os.path.join(filepath, item))
        print("removed mp3 from the folder")
            
    def remove_wav(self,audio_file_name):
        '''
        Remove wav from the folder
        '''
#         location = os.listdir(filepath)
#         for item in location:
        if audio_file_name.endswith(".wav"):
#             os.remove(os.path.join(filepath, item))
            os.remove(os.path.join(self.file_path, audio_file_name))
        print("remove wav file from the folder:",audio_file_name)
        
    def mp3_to_wav(self, audio_file_name): 
        '''
        Convert mp3 to wav format
        '''
        audio_files = os.listdir()
        # You dont need the number of files in the folder, just iterate over them directly using:
        for file in audio_files:
            #spliting the file into the name and the extension
            name, ext = os.path.splitext(file)
            if ext == ".mp3":
               mp3_sound = AudioSegment.from_mp3(file)
               #rename them using the old name + ".wav"
               mp3_sound.export("{0}.wav".format(name), format="wav")
        self.remove_mp3(self.file_path)
        
    def frame_rate_channel(self, audio_file_name):
        '''
        Get the frame rate and the number of channels
        '''
        with wave.open(audio_file_name, "rb") as wave_file:
            frame_rate = wave_file.getframerate()
            channels = wave_file.getnchannels()
            return frame_rate,channels
        
    def stereo_to_mono(self, audio_file_name):
        '''
        Converting stereo audio to mono,because google speech recognition api can't understand stereo files
        '''
        sound = AudioSegment.from_wav(audio_file_name)
        sound = sound.set_channels(1)
        sound.export(audio_file_name, format="wav")
        print("stereo to mono conversion Done")
        
    def high_low_pass(self, audio_file_name):
        '''
        Audio_preprocessing using High pass and Low pass band filters and filtering the noice
        '''
        (Frequency, array) = read(audio_file_name) # Reading the sound file. 
        b,a = signal.butter(5, 1000/(Frequency/2), btype='highpass') # ButterWorth filter 4350
        filteredSignal = signal.lfilter(b,a,array)
        c,d = signal.butter(5, 380/(Frequency/2), btype='lowpass') # ButterWorth low-filter
        newFilteredSignal = signal.lfilter(c,d,filteredSignal) # Applying the filter to the signal
        write(audio_file_name, Frequency, np.int16(newFilteredSignal/np.max(np.abs(newFilteredSignal)) * 44100))
        print("Audio_Preprocess_Done") #     os.system("/home/ml/folder_2/"+item)
        
    def upload_blob(self, bucket_name, source_file_name, destination_blob_name):
        '''
        Uploads a file to the bucket/google cloud.
        '''
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)
        print("upload_blob Done")
        
    def delete_blob(self, bucket_name, blob_name):
        '''
        Deletes a blob from the bucket/google cloud .
        '''
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        print("delete_blob Done")
        
    def google_transcribe(self, audio_file_name):

        '''
        Transfering audio file from local disc to google cloud and then using google speech recognition api 
        converting audio file to text data
        '''
        file_name = self.file_path + audio_file_name
        print("Pass only wav format audio files in a folder,NO MP3...etc")
#         self.mp3_to_wav(file_name)
        print("Audio_file_Name:",audio_file_name)

        # The name of the audio file to transcribe

        frame_rate, channels = self.frame_rate_channel(file_name)

        if channels == 1:
            print("It's mono_file")
        else:
            self.stereo_to_mono(file_name)
        self.high_low_pass(file_name)
        bucket_name = 'mlaudiofiles12'
        source_file_name = self.file_path + audio_file_name
        destination_blob_name = audio_file_name

        self.upload_blob(bucket_name, source_file_name, destination_blob_name)

        gcs_uri = 'gs://mlaudiofiles12/' + audio_file_name
        transcript = ''

        client = speech.SpeechClient()
        audio = types.RecognitionAudio(uri=gcs_uri)

        config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        language_code='hi_In')#specifing the language

        # Detects speech in the audio file
        operation = client.long_running_recognize(config, audio)
        response = operation.result(timeout=10000)

        for result in response.results:
            transcript += result.alternatives[0].transcript

        self.delete_blob(bucket_name, destination_blob_name)
#         self.remove_wav(audio_file_name)
        print("Speech_Text_Done")
        return transcript
    
    def load_model_text(self):
        '''
        loading the model.
        :param path: path to your clf model.
        '''
        self.loaded_model_text = pickle.load(open("/home/ml/Downloads/LANGUAGES_DATA/HINDI_DATA/model/Hindi_Sentiment_Analysis_BOG_NB.clf", 'rb'))
        return self.loaded_model_text

    def makepredictions_text(self, preprocessed_tweet):
        '''
        Process the files and get the Sentiment and their Probability.
        '''
        loaded_model_text = self.load_model_text()
        if len(preprocessed_tweet) ==0 :
            print("No_Voice")
        else:
            predictions = loaded_model_text.predict([preprocessed_tweet])
            probability = loaded_model_text.predict_proba([preprocessed_tweet])
            print("Prediction/Sentiment :", " ", predictions)
            print("Probability Ratio for Negative,Neutral,Positive Respectively :"," ",probability)
                  
    def preprocessing_tweet(self, transcript,audio_file_name):
            '''
            preprocessing the string
            '''
            self.transcript  = transcript
            stopword_list = []
            # get custom stopwords from a file (pt-br). You can create your own database of stopwords on a text file
            df = pd.read_fwf('/home/ml/Downloads/LANGUAGES_DATA/HINDI_DATA/hindi_stopwords.txt', header = None)
            # list of array
            custom_stopwords = df.values.tolist()
            # transform list of array to list
            custom_stopwords = [s[0] for s in custom_stopwords]
            # You can also add stopwords manually instead of loading from the database. Generally, we add stopwords that belong to this context.
            stopword_list.append('...')
            stopword_list.append('«')
            stopword_list.append('➔')
            stopword_list.append('|')
            stopword_list.append('»')
            # join all stopwords
            stopword_list.extend(custom_stopwords)
            # remove duplicate stopwords (unique list)
            stopword_list = list(set(stopword_list))
            print("Conversation_Begins & Gives Sentiment For Entire Conversation ")
             #remove the special characters
            tweet = re.sub(r'[?|!|\'|"|#]',r'',str(transcript))
            tweet = re.sub(r'[.|,|)|(|\|/]',r' ',tweet)
            # removing the digits
            tweet = re.sub(" \d+", " ", tweet)
            #remove urls from text
            tweet = re.sub(r'http\S+', r' ', tweet)
            #replace multiple spaces with single space
            tweet = re.sub('\s+',' ', tweet)
            tweet = tweet.strip()
        #     tweet = tweet.split()
        #     # #lower the words
        #     # tweet = tweet.lower()
        #     #remove the stop words
        #     tweet = [word for word in tweet if not word in stopword_list]
        #     tweet= ' '.join(tweet)
            print("Total_Conversation:\n",self.transcript)
#             print(tweet)
            self.makepredictions_text(tweet)
            self.remove_wav(audio_file_name)
                  
    def main_loop(self, filepath):
        self.file_path = filepath
        '''
        main loop from where all methods are calling
        '''
        self.mp3_to_wav(filepath)
        for audio_file_name in os.listdir (filepath):
            transcript = self.google_transcribe(audio_file_name)
            self.preprocessing_tweet(transcript,audio_file_name)

if __name__ == "__main__":
    obj = AI() 
    result  = obj.main_loop("/home/ml/Downloads/sample_1/")
