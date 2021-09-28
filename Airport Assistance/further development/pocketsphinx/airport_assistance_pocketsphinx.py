import os
from pocketsphinx import AudioFile, get_model_path, get_data_path, Pocketsphinx
import librosa as lr
import soundfile as sf

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
                "../EN/where_child.wav",
                "../EN/my_sentence1.wav",
                "../EN/my_sentence2.wav"]

# ground truth to be comapred
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
    
    audio, sr =  lr.load(audio_files[i], sr=None)

    # write out to a file to be used by pocketsphinx
    sf.write('test.wav', audio, sr, 'PCM_24')

    audio_file = "test.wav"

    assert os.path.exists(audio_file), audio_file + "does not exist"

    ps = Pocketsphinx()
    ps.decode(audio_file=audio_file)
    #ps.hypothesis()
    #ps.confidence()

    # print the ground truth and best 4 hypothesis
    print(ground_truth[i], ps.best(count=4))
