#!/bin/bash
printf "Analisys started by launch_all.sh\n\n"

t1=$(date +"%s")
for file in $(find $2 -name "*.json")
do
    printf "\n\nAnalisys on $file\n"
    ./DataAnalisys.py $file -j $1
done

cd ../
find results/ -name "*.json" > results/results_json_list.txt

t2=$(date +"%s")

tot_time=$(($t2 - $t1))

mins=$(($tot_time/60))
secs=$(($tot_time - $mins*60))

printf "\n\nlaunch_all.sh: EXECUTION TIME: $mins[min]$secs[s]\n\n"
