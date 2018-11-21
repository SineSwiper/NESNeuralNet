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

# Install dependencies for Torch
echo -e "\n==> Installing OS dependencies\n"

sudo apt-get update
sudo apt-get install -y \
    build-essential gcc g++ cmake curl libreadline-dev git-core libjpeg-dev libpng-dev libncurses5-dev imagemagick unzip xvfb \
    libqt4-dev liblua5.1-0-dev libgd-dev scons libgtk2.0-dev libsdl-dev libmkl-full-dev

echo -e "\n==> OS dependencies install finished\n"

# Create directory for sources
mkdir -p $TOPDIR/src

# Build and install Torch7
echo -e "\n==> Installing Torch7\n"

cd $TOPDIR/src
[ -d 'luajit-rocks' ] || git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir -p build
cd build
git checkout master; git pull
rm -f CMakeCache.txt
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release
make
make install

echo -e "\n==> Torch7 install finished\n"

# Install manifest in home directory to get rid of warnings

mkdir -p $HOME/.luarocks/lib/luarocks/rocks
$PREFIX/bin/luarocks-admin make-manifest --local-tree --tree=$HOME/.luarocks

# Install base+GPU packages
echo -e "\n==> Installing Lua dependencies\n"

gpu_rocks=''
path_to_nvcc=$(which nvcc)
[ -x "$path_to_nvcc" ] && gpu_rocks='cutorch cunn'

LUAROCKS=$PREFIX/bin/luarocks
for rock in cwrap paths torch nn luafilesystem penlight sys xlua image env nngraph $gpu_rocks; do
    # Don't install already-installed rocks
    [[ `$LUAROCKS list | grep "^$rock\$"` ]] || $LUAROCKS install $rock
done

#$PREFIX/bin/luarocks install qtlua
#$PREFIX/bin/luarocks install qttorch

echo -e "\n==> Lua dependencies install finished\n"

echo -e "\n==> Installing FCEUX\n"

cd $TOPDIR/src
[ -d 'fceux' ] || git clone https://github.com/SineSwiper/fceux
cd fceux
git checkout feature/lua_enhancements
LUA_LINKFLAGS='-L'$PREFIX'/lib -lluajit' LUA_INCDIR=$PREFIX'/include' scons SYSTEM_LUA=1 --prefix $PREFIX install

echo -e "\n==> FCEUX install finished\n"

#echo "Installing Lua-GD ... "
#cd $TOPDIR/src
#git clone https://github.com/ittner/lua-gd.git
#cd lua-gd
#sed -i "s/LUABIN=lua5.1/LUABIN=..\/..\/bin\/luajit/" Makefile
#$PREFIX/bin/luarocks make
#echo "Lua-GD installation completed"

echo -e "\n==== All done! ====\n"

