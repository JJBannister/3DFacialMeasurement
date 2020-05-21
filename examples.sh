#! /bin/sh

source activate Morf
cd ./src/examples

echo "Landmark Example"
python LandmarkScans.py

echo "Registration Example"
python RegisterScans.py
