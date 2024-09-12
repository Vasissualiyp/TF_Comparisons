#!/usr/bin/env bash

PWD=$(pwd)
HPDIR="../interface_music_run1"

mkdir $HPDIR; cd $HPDIR

rm -rf fields output param music-based-out.txt

cp -r ../../fields ./
cp -r ../../output ./
cp -r ../../pp-music-interface/param ./
cp ../../pp-music-interface/music-based-out.txt ./

cd $PWD
