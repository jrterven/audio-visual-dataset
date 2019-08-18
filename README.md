# Audio-visual dataset for speech recognition systems
This repository contains the source code to generate a database that can be used to train speech recognition systems from visual information. Text and video alignment are accurate in milliseconds thanks to the IBM Audio-to-Text engine.
The procedure can be used for any language simply by using videos in the desired language.
The generated output consists of a CSV file with the following fields:

| Anotación        | Descripción  | 
| ------------- |:-------------:|
| Link          | YouTube video link without the string: *www.youtube.com/*		 |
| Text          | Spoken text in the video sample						         |
| Confidence    | Average speech recognition score in the range [0, 1]		     |
| Start         | Sample start timestamp in seconds				                 |
| End			| Sample end timestamp in seconds								 |
| Mouth features| Eight 3D mouth features covering the inner lips		         |
| Difficulty     | Easy, medium or hard, according to the faces angles            |


# Procedure

## 1. Download the videos and place them in a directory.
You can generate several directories for different categories. E.g. news, blogs, etc.
Create a spreadsheet file whose name matches the category that contains the video link
followed by the name of the downloaded video. The first row of this file must contain * Link * in the first column and * Video * in the second column. This file is used to control the link of each video
and is used to generate the database with the link.


## 2. Extract audio from the videos
To extract the audio from the video file run **extract_wav_files.py**
```
python extract_wav_files.py --dir [directory] --cat [category]
```
where:
- *directory* is the path to the main directory in step 1   
- *category* is the category or subdirectory. 

For example, if the videos are inside *c:/videos/news* 
```
python extract_wav_files.py --dir c:/videos --cat news
```

## 3. Extract text from the videos
To extract text from the videos, we use IBM's **text-to-speech** engine.
In order to execute this part, you need an account in [IBM Cloud](https://idaas.iam.ibm.com/idaas/mtfim/sps/authsvc?PolicyId=urn:ibm:security:authentication:asf:basicldapuser). 
Then, you need to create a resource type **Speech to Text** to get the user name and password.
Once you have those credentials, open the file **extract_detailed_text_watson.py** and edit the fields *IBM_USERNAME** and **IBM_PASSWORD**. Then execute the script as follows.
```
python extract_detailed_text_watson.py --dir [directory] --cat [category]
```

## 4. Extract subvideos
Once you have the subtitles for each video, execute **extract_subvideos.py**
```
python extract_subvideos.py --dir [directory] --cat [category] --vids_log [log_file] --results_dir [results_directory] --ann_file [annotations_file]
```
where:
- *directory* is the path to the main directory in step 1   
- *category* is the category or subdirectory. 
- *log_file* is the file where the names of the processed videos are saved. This file is created if does not exists and it is saved inside *results_directory*.
- *results_directory* output directory
- *annotations_file* name of the final annotations file.

**Publication**  
Audio-visual database for Spanish-based speech recognition systems.   
Córdova Diana, Terven Juan, Romero Alejandro and Herrera Ana.
MICAI 2019


Licencia
----

MIT
