#!/bin/bash
# Modified from the original here: https://github.com/lupantech/ScienceQA/blob/main/tools/download.sh

cd images

if [ -d "train" ];
then
  echo "Already downloaded train"
else
  ls -alF
  wget https://scienceqa.s3.us-west-1.amazonaws.com/images/train.zip
  unzip -q train.zip
  rm train.zip
fi

if [ -d "val" ];
then
  echo "Already downloaded val"
else
  ls -alF
  wget https://scienceqa.s3.us-west-1.amazonaws.com/images/val.zip
  unzip -q val.zip
  rm val.zip
fi

if [ -d "test" ];
then
  echo "Already downloaded test"
else
  ls -alF
  wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip
  unzip -q test.zip
  rm test.zip
fi

echo "Completed downloads!"