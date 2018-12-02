#!/bin/bash
TOPDIR=$PWD
PREFIX=$PWD/torch

export LD_LIBRARY_PATH=$PREFIX/lib
export LUA_PATH="./?.lua;./?/init.lua;;"

# Copy stdout and stderr to a logfile
LOGFILE="logs/dqn_log_`/bin/date +\"%FT%R\"`"
exec > >(tee -i ${LOGFILE})
exec 2>&1

#$PREFIX/bin/fceux --sound 0 --nogui --loadlua $TOPDIR/dqn/train_agent.lua roms/Super*
xvfb-run $PREFIX/bin/fceux --sound 0 --nogui --loadlua $TOPDIR/dqn/train_agent.lua roms/Super*
