''' 
    Extract text from speech using IBM's Watson Text-to-speech engine

    Run as:
    python extract_detailed_text_watson.py --dir [directory] --cat [category]

    Juan Terven
    Diana Cordova
    Oct 2018
'''
import os
import pandas as pd
import subprocess
from watson_developer_cloud import SpeechToTextV1
import json
from natsort import natsorted
import argparse

directory = '/datasets/Our_dataset'
selected_cat = 'CNN'

IBM_USERNAME = ""
IBM_PASSWORD = ""

def main(args):
    directory = args.videos_dir
    selected_cat = args.category

    speech_to_text = SpeechToTextV1(username=IBM_USERNAME, password=IBM_PASSWORD)

    # Read wave file names in videos directory
    audio_names = []
    for video_file in os.listdir(os.path.join(directory, selected_cat)):
        if video_file.endswith(".wav") and not os.path.isfile(os.path.join(directory, selected_cat, video_file[0:-4] + '.json')):
            audio_names.append(video_file)
    audio_names = natsorted(audio_names)

    num_files = len(audio_names)
    print('found', num_files, 'files')
    #print(audio_names)

    # Read spreadsheet
    df = pd.read_excel(os.path.join(directory, selected_cat +'.xlsx'))

    for audio_name in audio_names:
        # For each video file, check if the link is available
        data = df[df['Video'].str.contains(audio_name[:-4])==True]

        link = ''

        if data.shape[0] == 0:
            print('Not found in spredsheet:', audio_name)
        else:
            link = data.iloc[0]['Link']

            # Extract text using Watson
            print('Extracting detailed text using Watson for', audio_name)
            audio_path = os.path.join(directory, selected_cat, audio_name)

            with open(audio_path, "rb") as audio_file:
                result = speech_to_text.recognize(audio_file, content_type="audio/wav",
                                model='es-ES_BroadbandModel', timestamps=True,
                                word_confidence=True, ).get_result()
                
                # add the link to the results
                result['link'] = link

                # save json file
                out_json_path = audio_path[:-4] + '.json'
                with open(out_json_path, 'w') as outfile:
                    json.dump(result, outfile)

if __name__== "__main__":
    
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Extract subvideos')
    parser.add_argument('--dir', dest='videos_dir',
                        help='Directory with videos', type=str)
    parser.add_argument('--cat', dest='category',
                        help='Video category', type=str)
    args = parser.parse_args()

    main(args)
