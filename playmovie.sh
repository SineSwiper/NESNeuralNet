#!/bin/bash
TOPDIR=$PWD
PREFIX=$PWD/torch

export LD_LIBRARY_PATH=$PREFIX/lib

$PREFIX/bin/fceux --sound 0 --nogui --playmov $1 roms/Super*
