''' 
    Extract wav files from videos

    Run as:
    python extract_wav_files.py --dir [directory] --cat [category]

    Juan Terven
    Diana Cordova
    Oct 2018
'''
import os
import pandas as pd
import subprocess
import argparse

def main(args):
    
    directory = args.videos_dir
    selected_cat = args.category

    # Read file names in videos directory
    video_names = []
    for video_file in os.listdir(os.path.join(directory, selected_cat)):
        if video_file.endswith(".mp4") and not os.path.isfile(os.path.join(directory, selected_cat, video_file[0:-4] + '.wav')):
            video_names.append(video_file)

    num_files = len(video_names)
    print('found', num_files, 'files')
    #print(video_names)

    # Read spreadsheet
    df = pd.read_excel(os.path.join(directory, selected_cat +'.xlsx'))

    for video_name in video_names:
        print('video name:', video_name)
        # For each video file, check if the link is available
        data = df[df['Video'].str.contains(video_name[:-4])==True]

        if data.shape[0] == 0:
            print('Not found in spredsheet:', video_name)

        print('Extracting wav file from ', video_name)
        # extract the wav file
        video_path = os.path.join(directory, selected_cat, video_name)
        out_wave_path = video_path[:-4] + '.wav'
        #ffmpeg -i video.mp4 -acodec pcm_s16le -ac 1 -ar 16000 out.wav
        subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'panic',
                        '-i', video_path, '-acodec', 'pcm_s16le', '-ac', '1',
                        '-ar', '16000', out_wave_path])


if __name__== "__main__":
    
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Extract subvideos')
    parser.add_argument('--dir', dest='videos_dir',
                        help='Directory with videos', type=str)
    parser.add_argument('--cat', dest='category',
                        help='Video category', type=str)
    args = parser.parse_args()

    main(args)        

    