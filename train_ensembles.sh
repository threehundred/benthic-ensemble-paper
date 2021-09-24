#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
   python3 train-ensemble.py Trip1-GROUP_DESC-ensemble-$i ./logs/Trip1-GROUP_DESC-ensemble-data
done