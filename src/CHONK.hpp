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
    chonk() { create(); }

  protected:
    int current_id;

  private:
    void create();
};


// void compute_receivers_d8(std::vector<int>& receivers, std::vector<float>& dist2receivers, std::vector<float>& elevation, int nx, int ny, float dx, float dy);
// void compute_donors(std::vector<int>& ndonors, std::vector<int>&  donors, std::vector<int>&  receivers, int nnodes);
// int _add2stack(int inode, std::vector<int>& ndonors, std::vector<int>& donors, std::vector<int>& stack, int istack);
// void compute_stack(std::vector<int>& stack, std::vector<int>& ndonors, std::vector<int>& donors, std::vector<int>& receivers, int& nnodes);



#endif