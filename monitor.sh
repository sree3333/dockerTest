#!/bin/bash
while :
do
	if [[ ! $(pgrep -f app.py) ]]; then
    		nohup python3 app.py &
	fi
done
