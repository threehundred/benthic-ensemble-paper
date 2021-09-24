#!/bin/bash
#for i in 0 1 2 3 4 5 6 7 8
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
  for j in 1 2 3 4 5 6 7 8 9 10
  do
     python3 inference.py \
              "./logs/Trip1-GROUP_DESC-ensemble-$i/checkpoints-Trip1-GROUP_DESC-ensemble-$i/weights.best.hdf5" \
              "./logs/Trip1-GROUP_DESC-ensemble-data/mean_image.jpg" \
              "./logs/Trip1-GROUP_DESC-ensemble-data/labels.txt" \
              "./logs/Trip1-GROUP_DESC-ensemble-data/X_test.p" \
              "./logs/Trip1-GROUP_DESC-ensemble-data/y_test.p" \
              "" \
              "" \
              "colour" \
              $j \
              "./inferenceoutputs/Trip1-inference-test-GROUP_DESC-colour$j-$i.p" \
              "./inferenceoutputs/Trip1-inferencey-test-GROUP_DESC-colour$j-$i.p"
  done
done