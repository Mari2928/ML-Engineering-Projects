## minimal vosk mic input example
## is a stipped out version of this:
## ## https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py
## which has an Apache 2.0 license
## https://github.com/alphacep/vosk-api/blob/master/COPYING

#import argparse
import os
import queue
import vosk
import sys
import json
import wave 

model_dir = "vosk-model-small-en-us-0.15"

if not os.path.exists(model_dir):
    print ("Please download a model for your language from https://alphacephei.com/vosk/models")
    print ("and unpack as "+model_dir+"' in the current folder.")
    sys.exit(0)

model = vosk.Model(model_dir) # load from a folder

# sample audios to be recognized
audio_files = [ "../EN/checkin.wav",                
                "../EN/checkin_child.wav",
                "../EN/parents.wav",
                "../EN/parents_child.wav",
                "../EN/suitcase.wav",
                "../EN/suitcase_child.wav",
                "../EN/what_time.wav",
                "../EN/what_time_child.wav",
                "../EN/where.wav",
                "../EN/where_child.wav"]


# go through each audio file and print its result
for i in range(0, len(audio_files)):

    with wave.open(audio_files[i]) as wf:
        assert wf.getnchannels() == 1, "must be a mono wav"
        assert wf.getsampwidth() == 2, "must be a 16bit wav"
        assert wf.getcomptype() == "NONE", "must be PCM data"

        rec = vosk.KaldiRecognizer(model, wf.getframerate())

        # wait until finish talking
        while True:	
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                #print(res["text"])
                break

        # print the final result
        print(rec.FinalResult())


