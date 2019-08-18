''' 
    Generate entries for the dataset by extracting bounding boxes
    and time stamps from videos.

    Run as:
    python extract_subvideos.py --dir [directorio] --cat [categoria]
            --vids_log [archivo_log] --results_dir [directorio_de_resultados]
            --ann_file [archivo_de_anotaciones]

    Juan Terven
    Diana Cordova
    Oct 2018
'''
from face_alignment_class import FaceAlignment
import numpy as np
import cv2
import re
import os
import math
import datetime
import json
from natsort import natsorted
import subprocess
import math
import csv
import argparse
import unicodedata

scale = 0.5    # downscale factor for input images to increase processing speed
max_bad_frames = 10  # maximum number of frames without face
min_area = 2500      # minimum face size (area in pixels)
csv_columns = ['link', 'text', 'conf', 'start', 'end', 'mouth3d', 'angle']

def main(args):
    
    fa = FaceAlignment()

    videos_directory = args.videos_dir
    results_dir = args.results_dir
    vids_name = args.category
    vid_proc_name = args.log_file
    dataset_annotation_file = args.ann_file
    if args.save_videos == 'True':
        save_videos = True
    else:
        save_videos = False

    # Create video window
    cv2.namedWindow('Vid')

    # load or create list with processed files
    processed_files = []
    videos_processed_exists = os.path.isfile(
        os.path.join(results_dir, vid_proc_name))
    if not videos_processed_exists:
        with open(os.path.join(results_dir, vid_proc_name), "w") as fp:
            for pfiles in processed_files:
                print(pfiles, file=fp)
    else:
        with open(os.path.join(results_dir, vid_proc_name)) as fp:
            processed_files = fp.read().splitlines()

    # Create annotation file the first time
    annotation_exists = os.path.isfile(os.path.join(
                        results_dir, dataset_annotation_file))
    if not annotation_exists:
        
        try:
            with open(os.path.join(
                results_dir, dataset_annotation_file), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
        except IOError:
            print("Error creating annotaton file. I/O error") 
    
    # Get json files list names in videos directory
    files_list = []
    for ann_file in os.listdir(os.path.join(videos_directory, vids_name)):
        if ann_file.endswith(".json"):
            files_list.append(ann_file[0:-5])
    files_list = natsorted(files_list)

    num_files = len(files_list)
    print('found', num_files, 'files')

    # traverse all the files
    stop_videos = False
    for file in files_list:
        if stop_videos:
            break

        # check if current video is not in alredy processed 
        if file in processed_files:
            print(file, 'has already been processed. Skipping it.')
            continue

        num_output_video = 0
        
        # Search for the video files in videos_directory
        video_name = file + '.mp4'
        print('Processing video:', video_name)

        if save_videos:
            # create output directory
            output_dir = os.path.join(results_dir, vids_name, file)

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

        # Load watson results
        with open(os.path.join(
            videos_directory, vids_name, file + '.json')) as f:
            stt_results = json.load(f)

        # Extract all the words with confidence >90
        words_data = extract_words_from_watson_results(stt_results, max_words=5)

        # Start the video capture
        cap = cv2.VideoCapture(os.path.join(
            videos_directory, vids_name, video_name))

        # Extract video metadata
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('video resolution:', width, ' x ', height)
        print('video framerate:', fps)

        frame_count = 0
        fps_processing = 30.0  # fps holder
        t = cv2.getTickCount() # initiate the tickCounter
        count = 0

        for entry in words_data:
            # Extract speech to text data
            print('entry:', type(entry), entry)
            s_sec, s_millisec = divmod(float(entry['start']), 1)
            e_sec, e_millisec = divmod(float(entry['end']), 1)
            s_min = 0
            e_min = 0
            s_millisec = s_millisec * 1000
            e_millisec = e_millisec * 1000
            
            print('s_sec, s_millisec:', s_sec, s_millisec)

            if s_sec >= 60:
                s_min = math.floor(s_sec / 60.0)
                s_sec = s_sec % 60
            if e_sec >= 60:
                e_min = math.floor(e_sec / 60.0)
                e_sec = e_sec % 60

            # Determine video frames involved in stt entry
            min_frame = s_min*fps*60 + (s_sec*fps)
            max_frame = e_min*fps*60 + (e_sec*fps)

            # go to min_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

            frame_count = min_frame
            # read frames from min_frame to max_frame
            num_people = 0
            
            valid_video = True
            landmarks = []
            angles = []
            consecutive_frames_no_people = 0
            while frame_count < max_frame:    
                if count == 0:
                    t = cv2.getTickCount()

                # capture next frame
                ret, frame = cap.read()

                if not ret:
                    continue

                frame_count += 1

                # resize frame for faster processing
                if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                    continue
                    
                frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                                         interpolation=cv2.INTER_LINEAR)

                # detect faces and landmarjs
                fa.update_features(frame_small)

                landmarks.append(fa.get_mouth_features(scale=scale))
                num_people = fa.get_num_people()
                angles.append(fa.get_yaw())

                # if it detects less than or more than 1 person
                # go to next subtitle
                if num_people != 1:                    
                    consecutive_frames_no_people += 1
                    
                if consecutive_frames_no_people >= max_bad_frames:
                    print(consecutive_frames_no_people,
                        ' frames without 1 person. Skiping to next subtitle')
                    valid_video = False
                    break
                
                # if only one person in the scene
                if num_people == 1:
                    consecutive_frames_no_people = 0

                    fa.renderMouth(frame_small)

                    # Put fps at which we are processing camera feed on frame
                    cv2.putText(frame_small, "{0:.2f}-fps".format(fps_processing),
                                (50, height-50), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 255), 2)

                # Display the image
                cv2.imshow('Vid',frame_small)
            
                # Read keyboard and exit if ESC was pressed
                k = cv2.waitKey(1) & 0xFF
                if k ==27:
                    exit()
                elif k == ord('q'):
                    stop_videos = True

                # increment frame counter
                count = count + 1
                # calculate fps at an interval of 100 frames
                if (count == 30):
                    t = (cv2.getTickCount() - t)/cv2.getTickFrequency()
                    fps_processing = 30.0/t
                    count = 0

            # if this was a valid video
            if valid_video and len(landmarks) > 0:
                num_output_video += 1

                entry['mouth3d'] = landmarks
                entry['angle'] = angles

                if save_videos:
                    s_hr = 0
                    e_hr = 0
                    if s_min >= 60:
                        s_hr = math.floor(s_min / 60)
                        s_min = s_min % 60
                    if e_min >= 60:
                        e_hr = math.floor(e_min / 60)
                        e_min = e_min % 60

                    # cut and crop video
                    # ffmpeg -i input.mp4 -ss hh:mm:ss -filter:v crop=w:h:x:y -c:a copy -to hh:mm:ss output.mp4
                    ss = "{0:02d}:{1:02d}:{2:02d}.{3:03d}".format(
                        s_hr, s_min, int(s_sec), math.ceil(s_millisec))
                    es = "{0:02d}:{1:02d}:{2:02d}.{3:03d}".format(
                        e_hr, e_min, int(e_sec), math.ceil(e_millisec))
                    crop = "crop={0:1d}:{1:1d}:{2:1d}:{3:1d}".format(
                        bbw, bbh, bbx1, bby1)

                    out_name = os.path.join(output_dir, str(num_output_video))

                    subprocess.call(['ffmpeg', #'-hide_banner', '-loglevel', 'panic',
                                    '-i', os.path.join(
                                    videos_directory, vids_name, video_name),
                                    '-ss', ss,
                                    '-filter:v', crop, '-c:a', 'copy',
                                    '-to', es, out_name +'.mp4'])
                                            # save recognized speech
                    text_file = open(out_name +'.txt', "w")
                    text_file.write(entry['text'] + '\n')
                    text_file.write(str(entry['conf']))
                    text_file.close()

        # append results to annotation file
        append_annotation_file(os.path.join(
            results_dir, dataset_annotation_file), words_data)

        # save name of processed file
        processed_files.append(file)
        with open(os.path.join(results_dir, vid_proc_name), "w") as fp:
            for p_file in processed_files:
                print(p_file, file=fp)

        
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def extract_text_conf_ts(s_idx, max_words, num_words, timestamps, conf, link):
    text = ''
    avg_conf = 0
    start = timestamps[int(s_idx * max_words)][1]
    end = timestamps[int(s_idx * max_words + num_words-1)][2]
    
    for w_idx in range(num_words):
        text = text + ' ' + timestamps[int(s_idx*max_words + w_idx)][0]
        avg_conf += conf[int(s_idx*max_words + w_idx)][1]

    avg_conf = round(avg_conf/num_words, 2)
    if len(text.strip()) >= 4:
        out_entry = {'link': link, 'text': text.strip(), 'conf': avg_conf,
                     'start':start, 'end': end, 'mouth3d': [],
                     'angle': [] }
    else:
        out_entry = {}
    return out_entry
    
def extract_words_from_watson_results(stt_results, max_words=5):
    data = stt_results['results']
    link = stt_results['link']
    link = link.rsplit('/', 1)[-1]
    out_data = []
    for sentence_idx, ann in enumerate(data):
        data_ann = ann['alternatives'][0]
        text = data_ann['transcript']
        conf = data_ann['word_confidence']
        timestamps = data_ann['timestamps']
        num_words = len(timestamps)
        num_splits = num_words//max_words
        rest = num_words%max_words

        if num_words < max_words:
            maxx_words = num_words
        else:
            maxx_words = max_words
        
        for s_idx in range(num_splits):
            out_entry = extract_text_conf_ts(s_idx, maxx_words, maxx_words,
                                             timestamps, conf, link)
            out_data.append(out_entry)
        
        if rest > 0:
            out_entry = extract_text_conf_ts(num_splits, maxx_words, rest,
                                         timestamps, conf, link)
            if out_entry:
                out_data.append(out_entry)
            
    return out_data

def append_annotation_file(csv_file, data):
    try:
        with open(csv_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            for entry in data:
                writer.writerow(entry)
    except IOError:
        print("I/O error") 

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii

if __name__== "__main__":
    
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Extract subvideos')
    parser.add_argument('--dir', dest='videos_dir',
                        help='Directory with videos', type=str)
    parser.add_argument('--cat', dest='category',
                        help='Video category', type=str)
    parser.add_argument('--vids_log', dest='log_file',
                        help='Name of log file', type=str)
    parser.add_argument('--results_dir', dest='results_dir',
                        help='Directory with results', type=str)
    parser.add_argument('--ann_file', dest='ann_file',
                        help='Annotations file (csv)', type=str)
    parser.add_argument('--save_videos', dest='save_videos',
                        help='Save videos', type=str, default='False')
    
    args = parser.parse_args()

    main(args)
