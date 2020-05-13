//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#ifndef CHONK_HPP
#define CHONK_HPP

// STL imports
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <fstream>
#include <queue>
#include <iostream>
#include <numeric>
#include <cmath>

// All the xtensor requirements
#include "xtensor-python/pyarray.hpp" // manage the I/O of numpy array
#include "xtensor-python/pytensor.hpp" // same
#include "xtensor-python/pyvectorize.hpp" // Contain some algorithm for vectorised calculation (TODO)
#include "xtensor/xadapt.hpp" // the function adapt is nice to convert vectors to numpy arrays
#include "xtensor/xmath.hpp" // Array-wise math functions
#include "xtensor/xarray.hpp"// manages the xtensor array (lower level than the numpy one)
#include "xtensor/xtensor.hpp" // same



class chonk
{
  public:
    // Default constructor
    chonk() { create(); }

    // Merge function
    void merge(std::vector<chonk> other_chonks);

    // move and split functions
    // TODO

    // Functions that apply and calculate fluxes
    // TODO

    // Accessors and modifyers
    // # Water flux
    double get_water_flux(){return water_flux;}
    void set_water_flux(double value){water_flux = value;}
    // # Erosion flux
    double get_erosion_flux(){return erosion_flux;}
    void set_erosion_flux(double value){erosion_flux = value;}

  protected:
    // The ID of the chonk
    int chonkID;
    // Current location on the graph
    int current_node;
    // Current flux of water in the CHONK (in L^3/T)
    double water_flux;
    // Current erosion flux in H/T
    double erosion_flux;

  private:
    void create();
};

#endif