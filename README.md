# CHONK

Repository accompanying the publication in GMD [see paper](https://gmd.copernicus.org/articles/17/71/2024/). It contains the code used to run the simulations and generate the data behind the figures. 

Update 2025 : I am still using this code here and there, but I am re-implementing the main feature in my new code base [`scabbard`](https://github.com/bgailleton/scabbard/tree/main)

#### What IS this code?
 A sandbox experimental Landscape Evolution Model developed to test a method crossing cellular automata and graph theory in given scenarios described in the companion paper.

#### What is it NOT?
 A stable and efficient framework to run LEMs simulations or develop new ones. While usable, it is more a proof-of-concept than anything else.


#### Why?
 It required a lot of trial-and-errors to get all the features working (especially the lake solver). The code is slow, require a lot of memory and is easily breakable.


#### But what if I want to use the method?
 So do we, that's why we are working on two other exciting projects:

- First, a stable, efficient and production-ready version of CHONK - now we learnt from all these errors. While not offering (yet) all the aspects of CHONK, the new code is already indescribably faster, cleaner and more flexible while requiring (way) less memory. It should be available in the coming months. 

- Then, a framework dedicated to building your own LEM following the philosophy described in the paper.

In any case, see (here for updates about the projects)[https://bgailleton.github.io/chonk/] or feel free to contact me if you have more questions.

## Installation 

Right, let's say you still want to use this version to verify/reproduce the results from the manuscript.

### Compiler

You first need to install a `c++` compiler (if you already have one able to compile with the standard `c++14` you are good to go), it can be quite heavy on `MacOS` and `Windows`, so I tried to keep the minimum requirements: 


- On Windows, look for "Build Tools for Visual Studio 2022" [here](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022). It will install the minimum tools required to build `C++` projects on windows.

- On MacOS, open a `Terminal` and run `xcode-select --install` to only install the command line tools to compile `c++` (already quite big...) and bypass the full installation of `Xcode`.

- On linux, make sure you have `gcc/g++` > 8. 

Note that on both Windows and MacOS you can also install some versions of `gcc/g++` through various methods, but they are quite difficult to get to work properly.

### Anaconda

You then need an `anaconda` environment manager. If you don't know what it is, let's say it creates small boxes in your computer and put all the code needed for a given application in the box so that it can find everything it needs, in hte right version, without interfering with the rest of the system. `Anaconda` is a company but license-free versions of their tool exist. I recommend `mambaforge` - you can find it [there](https://github.com/conda-forge/miniforge#mambaforge). 

Follow the installation instructions and start a new terminal: 

- First you need to create a box (ONLY NEEDED ONCE): `mamba create -n CHONK`
- Then you need to "enter" the box (NEEDED AT EACH NEW SESSION): `mamba activate CHONK`
- Install the dependencies (ONLY NEEDED ONCE, the last 2 package are only recommended to load/save `DEM`): `mamba install matplotlib git numpy scipy jupyterlab ipympl pybind11 cmake rasterio gdal xtensor-python ipyfastscape xarray-simlab`
- Now, you need to clone or download the current repository. `cd` wherever you wanna place it and run `git clone https://github.com/bgailleton/CHONK`
- Finally, `cd` to `CHONK` and run `python setup.py install` (only needed once).
- Done!

## Usage

See the `notebooks` folder for some examples.

# Credits

This model was primarily developed by Boris Gailleton (boris.gailleton@univ-rennes1.fr) at the GFZ institute (Potsdam, Germany) with the help and advice of Luca Malatesta, Guillaume Cordonnier and Jean Braun.

