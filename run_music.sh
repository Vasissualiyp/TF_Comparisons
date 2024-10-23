#!/usr/bin/env bash

# This script is supposed to run MUSIC, copy the resulting field, and run HMF comparison automatically
# It takes ~11 min to run this script, start-to-finish, on mussel, levelmin=10.

# Define runs and directories
SCRIPT_DIR=$(pwd)
INTERFACE_DIR="$SCRIPT_DIR/../../pp-music-interface/"
RUN_DIR="$SCRIPT_DIR/../music-interface-run2/"
FIELDNAME="Fvec_640Mpc_MUSIC"

# Run MUSIC
cd $INTERFACE_DIR
#./run_music.sh

# Move the fields file to the desired location
cd ..
#cp -f "fields/$FIELDNAME" "$RUN_DIR"fields/"$FIELDNAME"

# Run HMF (and PS) comparisons
cd $SCRIPT_DIR
python compare_HMFs.py


