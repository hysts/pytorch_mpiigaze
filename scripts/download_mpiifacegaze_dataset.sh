#!/usr/bin/env bash

set -Ceu

mkdir -p datasets
cd datasets
wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip
unzip MPIIFaceGaze_normalized.zip
mv MPIIFaceGaze_normalizad MPIIFaceGaze_normalized
