#ifndef CHONK_CPP
#define CHONK_CPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <ctime>
#include <fstream>
#include <functional>
#include <queue>
#include <limits>
#include <chrono>

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor/xadapt.hpp"


#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include <iostream>
#include <numeric>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
// #include <pair>

#include <queue>

#include "CHONK.hpp"


// This empty constructor is just there to have a default one.
void chonk::create()
{
  std::string yo = "I am an empty constructor yo!";

}


void chonk::merge(std::vector<chonk> other_chonks)
{
  // The merging function will need to be updated thouroughly!
  for(auto& tchonk:other_chonks)
  {
    this->water_flux += tchonk.get_water_flux();
  }
  // Probs add some destroyers here
}










#endif