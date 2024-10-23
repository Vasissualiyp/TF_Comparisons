#!/usr/bin/env bash

# This script is supposed to run MUSIC, copy the resulting field, and run HMF comparison automatically

# Define runs and directories
PWD=$(pwd)
INTERFACE_DIR="../../pp-music-interface/"
RUN_DIR="../music-interface-run2/"
FIELDNAME="Fvec_640Mpc_MUSIC"

# Run MUSIC
cd $INTERFACE_DIR
./run_music.sh

# Move the fields file to the desired location
cd ..
cp -f "fields/$FIELDNAME" "$RUN_DIR/$FIELDNAME"

# Run HMF (and PS) comparisons
cd $PWD
python compare_HMFs.py


