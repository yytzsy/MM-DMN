#!/bin/bash
# get all filename in specified path
path="/mnt/data/yuanyitian/videoQA/MSRVTT-QA/video/train-video/"
new_path="/mnt/data/yuanyitian/videoQA/MSRVTT-QA/video/train-video-wav/"
new_subfix=".wav"
files=$(ls $path)
for filename in $files
do
	ffmpeg -i $path$filename $new_path${filename%.*}$new_subfix
	echo $path$filename
	echo $new_path${filename%.*}$new_subfix
	# echo $filename >> filename.txt
done