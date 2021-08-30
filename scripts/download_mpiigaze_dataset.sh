#!/usr/bin/env bash

set -Ceu

mkdir -p datasets
cd datasets
wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz
tar xzvf MPIIGaze.tar.gz
