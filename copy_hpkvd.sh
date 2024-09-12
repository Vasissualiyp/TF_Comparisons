#!/usr/bin/env bash

PWD=$(pwd)
HPDIR="../interface_hpkvd_run1"

mkdir $HPDIR; cd $HPDIR

rm -rf fields output param hpkvd-based-out.txt

cp -r ../../fields ./
cp -r ../../output ./
cp -r ../../pp-music-interface/param ./
cp ../../pp-music-interface/hpkvd-based-out.txt ./

cd $PWD
