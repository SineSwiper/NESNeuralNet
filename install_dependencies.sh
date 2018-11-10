#!/usr/bin/env bash

######################################################################
# Torch install
######################################################################

# Die on any error
set -e

TOPDIR=$PWD

# Prefix:
PREFIX=$PWD/torch
echo "Installing Torch into: $PREFIX"

if [[ `uname` != 'Linux' ]]; then
  echo 'Platform unsupported, only available for Linux'
  exit
fi
if [[ `which apt-get` == '' ]]; then
    echo 'apt-get not found, platform not supported'
    exit
fi

# Install dependencies for Torch:
sudo apt-get update
sudo apt-get install -qqy build-essential
sudo apt-get install -qqy gcc g++
sudo apt-get install -qqy cmake
sudo apt-get install -qqy curl
sudo apt-get install -qqy libreadline-dev
sudo apt-get install -qqy git-core
sudo apt-get install -qqy libjpeg-dev
sudo apt-get install -qqy libpng-dev
sudo apt-get install -qqy ncurses-dev
sudo apt-get install -qqy imagemagick
sudo apt-get install -qqy unzip
sudo apt-get install -qqy libqt4-dev
sudo apt-get install -qqy liblua5.1-0-dev
sudo apt-get install -qqy libgd-dev
sudo apt-get install -qqy scons
sudo apt-get install -qqy libgtk2.0-dev
sudo apt-get install -qqy libsdl-dev
sudo apt-get update


echo "==> Torch7's dependencies have been installed"

# Create directory for sources
mkdir -p $TOPDIR/src

# Build and install Torch7
cd $TOPDIR/src
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir -p build
cd build
git checkout master; git pull
rm -f CMakeCache.txt
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release
make
make install

# Install base packages:
$PREFIX/bin/luarocks install cwrap
$PREFIX/bin/luarocks install paths
$PREFIX/bin/luarocks install torch
$PREFIX/bin/luarocks install nn

# Install GPU packages:
path_to_nvcc=$(which nvcc)
if [ -x "$path_to_nvcc" ]
then
    $PREFIX/bin/luarocks install cutorch
    $PREFIX/bin/luarocks install cunn
fi

$PREFIX/bin/luarocks install luafilesystem
$PREFIX/bin/luarocks install penlight
$PREFIX/bin/luarocks install sys
$PREFIX/bin/luarocks install xlua
$PREFIX/bin/luarocks install image
$PREFIX/bin/luarocks install env
#$PREFIX/bin/luarocks install qtlua
#$PREFIX/bin/luarocks install qttorch
$PREFIX/bin/luarocks install nngraph

echo ""
echo "=> Torch7 has been installed successfully"
echo ""

echo "Installing FCEUX ... "
cd $TOPDIR/src
git clone https://github.com/TASVideos/fceux
cd fceux
LUA_LINKFLAGS='-L'$PREFIX'/lib -lluajit' LUA_INCDIR=$PREFIX'/include' scons SYSTEM_LUA=1 --prefix $PREFIX install
echo "FCEUX installation completed"

echo "Installing Lua-GD ... "
cd $TOPDIR/src
git clone https://github.com/ittner/lua-gd.git
cd lua-gd
sed -i "s/LUABIN=lua5.1/LUABIN=..\/..\/bin\/luajit/" Makefile
$PREFIX/bin/luarocks make
echo "Lua-GD installation completed"

echo
echo "All done!"

