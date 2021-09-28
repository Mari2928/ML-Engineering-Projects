# Import necessary libraries.
from deepspeech import Model, version
import librosa as lr
import numpy as np
import os 
import sys
from scipy.ndimage.filters import uniform_filter1d
from scipy import signal
import noisereduce as nr
from jiwer import wer

# ask user to input the language selection
language = input ("Enter language (English, Italian, or Spanish):")

# Printing the selected language
print ("%s is selected. Evaluating the files...\n" % language)

# Define process options

# hypothesize the original file
def option1_original(audio, ground_truth):
    # scale the audio from -1 to 1 to +/-32767  
    result = (audio * 32767).astype(np.int16)    
    hypothesis = ds.stt(result)  
    wer1 = wer(ground_truth, hypothesis)
    return wer1, hypothesis

# hypothesize after low-pass filter
def option2_lowpass(audio, ground_truth):
    result = uniform_filter1d(audio, size=3)
    result = (result * 32767).astype(np.int16) 
    hypothesis = ds.stt(result)
    wer2 = wer(ground_truth, hypothesis)
    return wer2, hypothesis

# hypothesize after noise reduction 
def option3_noise_reduction(audio, ground_truth):
    result = nr.reduce_noise(audio_clip=audio, noise_clip=audio, verbose=False)
    result = (result * 32767).astype(np.int16) 
    hypothesis = ds.stt(result)
    wer3 = wer(ground_truth, hypothesis)
    return wer3, hypothesis

# try 3 options and select the best one
def get_best_wer(audio, ground_truth):       

    # option 1: original
    wer1, hypothesis1 =  option1_original(audio, ground_truth)
    bestwer = [wer1, hypothesis1, "1. original"]

    # option 2: with low-pass filter
    wer2, hypothesis2 =  option2_lowpass(audio, ground_truth)
    if wer2 < bestwer[0]: bestwer = [wer2, hypothesis2, "2. lowpass filter"]

    # option 3: with noise reduction
    wer3, hypothesis3 = option3_noise_reduction(audio, ground_truth)
    if wer3 < bestwer[0]: bestwer = [wer3, hypothesis3, "3. noise reduction"]
    
    return bestwer


if language == 'English':
    
    # Set the deepspeech models for English
    scorer = "Models/deepspeech-0.9.3-models.scorer"
    model = "Models/deepspeech-0.9.3-models.pbmm"

    # Instanciate the deepspeech model, enable external scorer, and set the desired sample rate.
    ds = Model(model)
    ds.enableExternalScorer(scorer)
    desired_sample_rate = ds.sampleRate()
    
    # audio samples to be recognized
    audio_files = [ "Audio_Files/EN/checkin.wav",
                   "Audio_Files/EN/checkin_child.wav",
                   "Audio_Files/EN/parents.wav",
                   "Audio_Files/EN/parents_child.wav",
                   "Audio_Files/EN/suitcase.wav",
                   "Audio_Files/EN/suitcase_child.wav",
                   "Audio_Files/EN/what_time.wav",
                   "Audio_Files/EN/what_time_child.wav",
                   "Audio_Files/EN/where.wav",
                   "Audio_Files/EN/where_child.wav",
                   "Audio_Files/EN/my_sentence1.wav",
                   "Audio_Files/EN/my_sentence2.wav"]
    
    # ground truth to be evaluated
    ground_truth = ["where is the checkin desk",
                   "where is the checkin desk",
                   "i have lost my parents",
                   "i have lost my parents",
                   "please i have lost my suitcase",
                   "please i have lost my suitcase",
                   "what time is my plane",
                   "what time is my plane",
                   "where are the restaurants and shops",
                   "where are the restaurants and shops",
                   "where can i get on the taxi",
                   "how do i get to london from here"]
    
    for i in range(0, len(audio_files)):          
        
        # load audio file
        audio, sr = lr.load(audio_files[i], sr=desired_sample_rate)
        
        # try 3 options and select the best one
        bestwer = get_best_wer(audio, ground_truth[i])
        
        print(bestwer) 
        
elif language == 'Italian':
    
    # Set the deepspeech models for Italian
    scorer = "Models/kenlm_it.scorer"
    model = "Models/output_graph_it.pbmm"

    # Instanciate the deepspeech model, enable external scorer, and set the desired sample rate.
    ds = Model(model)
    ds.enableExternalScorer(scorer)
    desired_sample_rate = ds.sampleRate()
    
    # audio samples to be recognized
    audio_files = [ "Audio_Files/IT/checkin_it.wav",
                  "Audio_Files/IT/parents_it.wav",
                  "Audio_Files/IT/suitcase_it.wav",
                  "Audio_Files/IT/what_time_it.wav",
                  "Audio_Files/IT/where_it.wav",]
    
    # ground truth to be evaluated
    ground_truth = ["dove e il bancone",
                   "ho perso i miei genitori",
                   "per favore ho perso la mia valigia",
                   "a che ora e il mio aereo",
                   "dove sono i ristoranti e i negozi"]
    
    for i in range(0, len(audio_files)):          
        
        # load audio file
        audio, sr = lr.load(audio_files[i], sr=desired_sample_rate)
        
        # try 3 options and select the best one
        bestwer = get_best_wer(audio, ground_truth[i])       
        
        print(bestwer)
        
elif language == 'Spanish':
    
    # Set the deepspeech models for Spanish
    scorer = "Models/kenlm_es.scorer"
    model = "Models/output_graph_es.pbmm"

    # Instanciate the deepspeech model, enable external scorer, and set the desired sample rate.
    ds = Model(model)
    ds.enableExternalScorer(scorer)
    desired_sample_rate = ds.sampleRate()
    
    # audio samples to be recognized
    audio_files = [ "Audio_Files/ES/checkin_es.wav",
                  "Audio_Files/ES/parents_es.wav",
                  "Audio_Files/ES/suitcase_es.wav",
                  "Audio_Files/ES/what_time_es.wav",
                  "Audio_Files/ES/where_es.wav",]
    
    # ground truth to be evaluated
    ground_truth = ["Dónde están los mostradores",
                   "he perdido a mis padres",
                   "por favor, he perdido mi maleta",
                   "a qué hora es mi avión",
                   "dónde están los restaurantes y las tiendas"]
    
    for i in range(0, len(audio_files)):          
        
        # load audio file
        audio, sr = lr.load(audio_files[i], sr=desired_sample_rate)
        
        # try 3 options and select the best one
        bestwer = get_best_wer(audio, ground_truth[i])       
        
        print(bestwer) 
    
else:
    print("Please enter the supported language (English, Italian, or Spanish).")
    sys.exit(1)
    
    
