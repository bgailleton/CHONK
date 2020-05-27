//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#ifndef cppintail_HPP
#define cppintail_HPP

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



// this class organises the DEM nodes in order to solve all equations in the right order. This is the first step of each iteration of the model 
class NodeGraph
{
  public:
    NodeGraph() { create(); }
    NodeGraph(xt::pytensor<int,1>& pre_stack,xt::pytensor<int,1>& pre_rec, xt::pytensor<int,1>& post_rec,
  xt::pytensor<int,1>& tMF_stack, xt::pytensor<int,2>& tMF_rec, xt::pytensor<int,2>& tMF_don, xt::pytensor<double,1>& elevation, xt::pytensor<double,2>& tMF_length,
  float XMIN, float XMAX, float YMIN, float YMAX, float XRES, float YRES, int NROWS, int NCOLS, float NODATAVALUE)
    {create( pre_stack, pre_rec, post_rec, tMF_stack, tMF_rec, tMF_don, elevation, tMF_length, XMIN,  XMAX,  YMIN,  YMAX,  XRES, YRES, NROWS, NCOLS, NODATAVALUE);}

    inline int row_col_to_node(int& row, int& col){return row * NCOLS + col;};
    inline int row_col_to_node(size_t& row, size_t& col){return int(row * NCOLS + col);};
    // This function transform the linearised node indice to row/col
    inline void node_to_row_col(int& node, int& row, int& col)
    {
        col = node % NCOLS;
        row = int((node - col)/NCOLS);
    };

    void calculate_inherited_water_from_previous_lakes(xt::pytensor<double,1>& previous_lake_depth, xt::pytensor<int,1>& post_rec);



    // Accessors/modifiers
    //# Stacks and receivers
    int get_MF_stack_at_i(int i){return MF_stack[i];};
    std::vector<int> get_MF_receivers_at_node(int node){std::vector<int>output(8);for(size_t i=0;i<8;i++){output[i] = MF_receivers(node,i);};return output;};
    std::vector<int> get_MF_donors_at_node(int node){std::vector<int>output(8);for(size_t i=0;i<8;i++){output[i] = MF_donors(node,i);};return output;};
    std::vector<double> get_MF_lengths_at_node(int node){std::vector<double>output(8);for(size_t i=0;i<8;i++){output[i] = MF_lengths(node,i);};return output;};

    
    //# pits
    int get_pits_ID_at_node(int node){return pits_ID[node];};
    int get_pits_bottom_at_pit_ID(int ID){return pits_bottom[ID];};
    int get_pits_outlet_at_pit_ID(int ID){return pits_outlet[ID];};
    double get_pits_volume_at_pit_ID(int ID){return pits_volume[ID];};
    std::vector<int> get_pits_pixels_at_pit_ID(int ID){return pits_pixels[ID];};
    std::vector<int> get_sub_pits_at_pit_ID(int ID){return sub_depressions[ID];};
    double get_erosion_flux_at_node(int node){return register_erosion_flux[node];}
    void add_erosion_flux_at_node(int node, double val){register_erosion_flux[node] += val;}
    double get_deposition_flux_at_node(int node){return register_deposition_flux[node];}
    void add_deposition_flux_at_node(int node, double val){register_deposition_flux[node] += val;}
    double get_excess_water_at_pit_ID(int pID){return pits_inherited_water_volume[pID];}
    bool does_this_node_has_inhereted_water(int node) {return has_excess_water_from_lake[node];};
    double get_node_excess_at_node(int node){return node_to_excess_of_water[node];};
    xt::pytensor<int,1> get_all_nodes_in_depression();

    //# DEBUG
    xt::pytensor<int,1> DEBUG_get_preacc(){return preacc;}



  protected:
    // Geometrical/geographical features, their name should be self-explanatory
    float XMIN;
    float XMAX;
    float YMIN;
    float YMAX;
    float XRES;
    float YRES;
    int NROWS;
    int NCOLS;
    float NODATAVALUE;

    // These are the stacks ingested from fastscaplib_fortran
    xt::pytensor<int,1> MF_stack;
    xt::pytensor<int,1> preacc;
    xt::pytensor<int,1> basin_label;
    xt::pytensor<int,2> MF_receivers;
    xt::pytensor<double,2> MF_lengths;
    xt::pytensor<double,2> MF_donors;

    // Number of depressions
    int n_pits;
    // length=N_nodes, -1 if not in a pit, pit_ID otherwise
    std::vector<int> pits_ID;
    // length = n_pits, pit_ID to bottom_nodes
    std::vector<int> pits_bottom;
    // length = n_pits, pit_ID to outlet node. If equal to bottom: fluxes can escape the model
    std::vector<int> pits_outlet; 
    // legth = n_pits, pit_ID to number of pixels in the pit.
    std::vector<int> pits_npix; 
    // list of pixels in each pits
    std::vector<std::vector<int> > pits_pixels;
    // length =  N_pits, value = list of subdepressions IDs 
    std::vector<std::vector<int> > sub_depressions; 
    // length = n_pits, pit_ID to volume in L^3
    std::vector<double> pits_volume;
    // length = n_pits, pit_ID to inherited volume in L^3
    std::vector<double> pits_inherited_water_volume;
    // 

    // These two maps record for each pit node the erosion and deposition that has happened there
    // This is usefull to inverse the process when filling a pit (or not)
    std::map<int,double> register_deposition_flux;
    std::map<int,double> register_erosion_flux;

    // Dealing with excess of water
    // length = N_nodes, value: true if it has an excess of water
    std::vector<bool> has_excess_water_from_lake;
    // key: node ID, val: excess volume of water due to previous lake (recalculated to be diverted to outlets)
    std::map<int,double> node_to_excess_of_water;


  private:
    void create();
    void create(xt::pytensor<int,1>& pre_stack,xt::pytensor<int,1>& pre_rec,xt::pytensor<int,1>& post_rec, 
  xt::pytensor<int,1>& tMF_stack, xt::pytensor<int,2>& tMF_rec,xt::pytensor<int,2>& tMF_don, xt::pytensor<double,1>& elevation, xt::pytensor<double,2>& tMF_length,
  float XMIN, float XMAX, float YMIN, float YMAX, float XRES, float YRES, int NROWS, int NCOLS, float NODATAVALUE);



};





// Older tests


class cppintail
{
  public:
  
    cppintail() { create(); }
    cppintail(float tXMIN, float tXMAX, float tYMIN, float tYMAX, float tXRES, float tYRES, int tNROWS, int tNCOLS, float tNODATAVALUE) { create(tXMIN, tXMAX, tYMIN, tYMAX, tXRES, tYRES, tNROWS, tNCOLS, tNODATAVALUE); }

    void compute_neighbors(xt::pytensor<float,2>& DEM);
    
    void flowdir_to_receiver_indices(int nodeID, std::vector<int>& receiver_nodes);
    void flowdir_to_receiver_indices(int row, int col, std::vector<int>& receiver_rows, std::vector<int>& receiver_cols);

    void Initialise_MF_stacks(xt::pytensor<float,2>& DEM);

    void compute_DA_slope_exp( double slexponent, xt::pytensor<float,2>& DEM);

    void find_nodes_with_no_donors( xt::pytensor<float,2>& DEM);



    // This function transform the linearised node indice to row/col
    inline void node_to_row_col(int& node, int& row, int& col)
    {
        col = node % NCOLS;
        row = int((node - col)/NCOLS);
    };

    inline int row_col_to_node(int& row, int& col){return row * NCOLS + col;};
    inline int row_col_to_node(size_t& row, size_t& col){return int(row * NCOLS + col);};


    //Getter
    std::vector<int> get_label_from_nodonode(){return label_from_nodonodes;};
    xt::pytensor<int,2> get_flowdir(){return FLOWDIR;};
  

  protected:

    // Geometrical/geographical features
    float XMIN;
    float XMAX;
    float YMIN;
    float YMAX;
    float XRES;
    float YRES;
    int NROWS;
    int NCOLS;
    float NODATAVALUE;

    // flow - directions
    // Binary system to detect where the flow goes
    // each number is 0 for block and 1 for goes
    // position is around the pixel:
    // XXXXXXXX:
    // 1,2,3
    // 8,0,4
    // 7,6,5
    // For example:
    // 10010111 means flows in 5 directions
    xt::pytensor<int,2> FLOWDIR;

    // Drainage Area
    xt::pytensor<float,2> Drainage_area;

    // Node with no donor
    std::vector<int> no_donor_nodes, label_from_nodonodes;
    

    // MF stacks: to deal with multiple flow direction
    std::vector<int> MF_stack;





  private:
    void create();
    void create(float tXMIN, float tXMAX, float tYMIN, float tYMAX, float tXRES, float tYRES, int tNROWS, int tNCOLS, float tNODATAVALUE);

};







#endif