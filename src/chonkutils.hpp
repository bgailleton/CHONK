//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#ifndef chonkutils_HPP
#define chonkutils_HPP

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
#include <initializer_list>


// All the xtensor requirements
#include "xtensor-python/pyarray.hpp" // manage the I/O of numpy array
#include "xtensor-python/pytensor.hpp" // same
#include "xtensor-python/pyvectorize.hpp" // Contain some algorithm for vectorised calculation (TODO)
#include "xtensor/xadapt.hpp" // the function adapt is nice to convert vectors to numpy arrays
#include "xtensor/xmath.hpp" // Array-wise math functions
#include "xtensor/xarray.hpp"// manages the xtensor array (lower level than the numpy one)
#include "xtensor/xtensor.hpp" // same





// #####################################################
// ############# Internal Node objects #################
// #####################################################

// the class nodium is just used for the priority queue struture when solving lakes.
// it is a very small class that combine a node index and its elevation when I insert it within the PQ
// The operators are defined in the cpp file.
class nodium
{
  public:
    // empty constructor
    nodium(){};
    // Constructor by default
    nodium(int node,double elevation){this->node = node; this->elevation = elevation;};
    // Elevation data
    double elevation;
    // Node index
    int node;
};

template<class T, class U>
class PQ_helper
{
  public:
    // empty constructor
    PQ_helper(){};
    // Constructor by default
    PQ_helper(T node,U score){this->node = node; this->score = score;};
    // Elevation data
    U score;
    // Node index
    T node;
};


// Hack the container behind
template <class T, class S, class C>
    S& Container(std::priority_queue<T, S, C>& q) {
        struct HackedQueue : private std::priority_queue<T, S, C> {
            static S& Container(std::priority_queue<T, S, C>& q) {
                return q.*&HackedQueue::c;
            }
        };
    return HackedQueue::Container(q);
};

// nodiums are sorted by elevations for the depression filler
inline bool operator>( const nodium& lhs, const nodium& rhs )
{
  return lhs.elevation > rhs.elevation;
}
inline bool operator<( const nodium& lhs, const nodium& rhs )
{
  return lhs.elevation < rhs.elevation;
}

template<class T, class U>
inline bool operator>( const PQ_helper<T,U>& lhs, const PQ_helper<T,U>& rhs )
{
  return lhs.score > rhs.score;
}
template<class T, class U>
inline bool operator<( const PQ_helper<T,U>& lhs, const PQ_helper<T,U>& rhs )
{
  return lhs.score < rhs.score;
}






inline double sumvecdouble(xt::pytensor<double,1>& vec)
{
  double sum = 0;
  for(auto v:vec)
    sum +=v;
  return sum;;
}


































#endif