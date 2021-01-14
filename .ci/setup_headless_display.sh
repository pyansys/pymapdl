#!/bin/sh
set -x

sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx xvfb
which Xvfb
Xvfb $DISPLAY -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3
set +x
