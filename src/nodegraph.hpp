//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#ifndef nodegraph_HPP
#define nodegraph_HPP

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
#include "chonkutils.hpp"
#include "depressiontree.hpp"


// All the xtensor requirements
#include "xtensor-python/pyarray.hpp" // manage the I/O of numpy array
#include "xtensor-python/pytensor.hpp" // same
#include "xtensor-python/pyvectorize.hpp" // Contain some algorithm for vectorised calculation (TODO)
#include "xtensor/xadapt.hpp" // the function adapt is nice to convert vectors to numpy arrays
#include "xtensor/xmath.hpp" // Array-wise math functions
#include "xtensor/xarray.hpp"// manages the xtensor array (lower level than the numpy one)
#include "xtensor/xtensor.hpp" // same


// #include "fastscapelib/basin_graph.hpp"
// #include "fastscapelib/Profile.h"
// #include "fastscapelib/union_find.hpp"
// #include "fastscapelib/utils.hpp"

void set_DEBUG_switch_nodegraph(std::vector<std::string> params, std::vector<bool> values );


// Vertx class: A class that manage one vertex: a node, its ID, receivers, length, donors, ...
// Anything useful to generate a graph
class Vertex 
{
public:
  // Empty constructor
  Vertex()
  {
    donors.reserve(8);
    Sdonors.reserve(8);
    receivers.reserve(8);
    length2rec.reserve(8);
    length2don.reserve(8);
  };
  // Useful constructor
  Vertex(
    std::vector<int> donors,  // child vertex in the donors direction
    std::vector<int> receivers, // child vertex in the receiver direction
    std::vector<double> length2rec, // length to each child vertex in the receiver direction
    std::vector<double> length2don // length to each child vertex in the receiver direction
    ){this->donors = donors; this->receivers = receivers; this->length2rec = length2rec;this->length2don = length2don;}

  // bool visiting; // bool for DFS
  // bool visited; // bool for DFS
  std::vector<int> donors; // list of child nodes in teh donor direction
  std::vector<int> Sdonors; // list of child nodes in teh donor direction
  std::vector<int> receivers; // list of child nodes in the receivers direction
  int Sreceivers; // list of child nodes in the receivers direction
  std::vector<double> length2rec; // list of length to receiver node
  double length2Srec; // length to steepest receiver node
  std::vector<double> length2don; // list of length to receiver node
};

// This class is managing the depression 
class Depression
{
public:

  Depression(){return;};
  Depression(int index, int parent, int level, int pit){this->index = index;this->parent = parent;this->level = level;
   this->volume = 0; this->hw_max = 0; this->pit = pit;};//
  // ID in the depression tree
  int index;
  // parent depression (-1 is none)
  int parent;
  // Children depressions (direct receivers in the tree)
  std::pair<int,int> children = {-111,-111};

  int twin = -1;

  bool has_children = false;
  // Depression level (see the different shades of grey in Figure 3 of Barnes et al., 2020 https://doi.org/10.5194/esurf-8-431-2020) 
  int level;
  // Connections to other basins
  int outlet;
  int external_connection;
  int internal_connection;
  // nodes in the depressions
  std::vector<int> nodes;
  // total Volume
  double volume;
  double min_volume_to_start = 0;

  // maximum height
  double hw_max;
  //pit
  int pit;

  bool processed = false;
  bool final = false;

  std::vector<double> label_prop;

  double volume_water = 0;
  double volume_sed = 0;
  double hw = 0;

};



//Deprecated

// // Depth first search algorithms for generic graph traversal (I do not use it anymore, but it is here just in case)
// bool dfs(
//   Vertex& vertex, 
//   std::vector<Vertex>& stack, 
//   std::vector<int>& next_vertexes, 
//   int& index_of_reading, 
//   int& index_of_pushing, 
//   std::vector<bool>& is_in_queue, 
//   std::vector<Vertex>& graph,
//   std::string& direction
//   ) ;
// bool dfs(Vertex& vertex,
// std::vector<Vertex>& stack,
// std::vector<Vertex>& graph,
// std::string& direction
// ) ;
// // Generic topological sort using DFS algorithm
// std::vector<int> topological_sort_by_dfs(std::vector<Vertex>& graph, int starting_node, std::string& direction) ;
// std::vector<int> topological_sort_by_dfs(std::vector<Vertex>& graph, std::string& direction) ;




class NodeGraphV2
{
public:
  // Empty Constructor
  NodeGraphV2() { };

  // Contructor for grid-type DEM from fastscape
  NodeGraphV2(
xt::pytensor<double,1>& elevation, // vectorised elevation
xt::pytensor<bool,1>& active_nodes, // array of active node - ie true where the node has erosion and stuff, false when it allows fluxes to escape
double dx, // resolution in x
double dy, // resolution in y
int nrows, // number of rows
int ncols, // number of cols
bool lake_solver // implicit if false, explicit if true
  );

std::vector<char> is_border;


// Multiple flow receivers can have some duplicates in fastscapelib-fortran. I somehow need to correct it
void initial_correction_of_MF_receivers_and_donors(xt::pytensor<int,2>& tMF_rec, xt::pytensor<int,2>& tMF_don, xt::pytensor<double,1>& elevation);

// More general utilities functions here:
//# Returns the address of the full stack
xt::pytensor<int,1>& get_MF_stack_full_adress(){return Mstack;}
//# Returns a copy of the full stack
xt::pytensor<int,1> get_MF_stack_full(){return Mstack;}
//# Returns a vector of the receivers at a certain node
std::vector<int>& get_MF_receivers_at_node(int node){return graph[node].receivers;};
//# Returns a vector of receivers without any rerouted pit (pits won't have any receivers)
std::vector<int>& get_MF_receivers_at_node_no_rerouting(int node){if(is_depression(node)){return empty_vector;} else {return graph[node].receivers;} };
//# Returns a vector of the donors at a certain node
std::vector<int>& get_MF_donors_at_node(int node){return graph[node].donors;};
//# Returns a vector of the lengths to receivers at a certain node
std::vector<double>& get_MF_lengths_at_node(int node){return graph[node].length2rec;};
//# Returns node at index i in the MF stack
int get_MF_stack_at_i(int i){return Mstack[i];}
// TODO
int get_index_MF_stack_at_i(int i){return index_in_Mstack[i];}
//# Check if the pit is to be rerouted
bool is_depression(int i){return pits_to_reroute[i];}
//# Update the receivers at node
void update_receivers_at_node(int node, std::vector<int>& new_receivers);
//# Update the donors at a node
void update_donors_at_node(int node, std::vector<int>& new_donors);

void compute_receveivers_and_donors(xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation);
void compute_receveivers_and_donors(xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation, std::vector<int>& nodes_to_compute);

void get_D8_neighbors(int i, xt::pytensor<bool,1>& active_nodes, std::vector<int>& neightbouring_nodes, std::vector<double>& length2neigh);
void get_D8_neighbors(int i, xt::pytensor<int,1>& active_nodes, std::vector<int>& neightbouring_nodes, std::vector<double>& length2neigh);
std::vector<int> get_all_flat_from_node(int i, xt::pytensor<double,1>& topography,  xt::pytensor<bool,1>& active_nodes);
bool are_there_flat_pits(xt::pytensor<double,1>& elevation, xt::pytensor<bool,1>& active_nodes);


void get_D4_neighbors(int i, xt::pytensor<bool,1>& active_nodes, std::vector<int>& neightbouring_nodes, std::vector<double>& length2neigh);

int get_Srec(int i) {return this->graph[i].Sreceivers;}
double get_length2Srec(int i) {return this->graph[i].length2Srec;}

std::vector<int> get_broken_nodes(){return not_in_stack;}
bool is_flat_draining(int node, xt::pytensor<double,1>& elevation, xt::pytensor<bool,1>& active_nodes);

// Single flow stack
// # Main function
void compute_stack();
// # recursive function
int _add2stack(int& inode, int& istack);

// Compute the Single flow basin labels and the pits
void compute_basins(xt::pytensor<bool,1>& active_nodes);
void compute_pits(xt::pytensor<bool,1>& active_nodes);
void correct_flowrouting(xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation);
void _connect_basins(xt::pytensor<int,2>& conn_basins, xt::pytensor<int,2>& conn_nodes, xt::pytensor<double,1>& conn_weights,          
                   xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation, int& nconn, int& basin0);
xt::xtensor<int,1> _compute_mst_kruskal(xt::pytensor<int,2>& conn_basins, xt::pytensor<double,1>& conn_weights);
void _orient_basin_tree(xt::pytensor<int,2>& conn_basins, xt::pytensor<int,2>& conn_nodes, int& basin0, xt::xtensor<int,1>& tree);
void _update_pits_receivers(xt::pytensor<int,2>& conn_basins,xt::pytensor<int,2>& conn_nodes, xt::xtensor<int,1>& mstree, xt::pytensor<double,1>& elevation);

std::vector<std::vector<int> > get_DEBUG_connbas() {return  DEBUG_connbas;}
std::vector<std::vector<int> > get_DEBUG_connode() {return  DEBUG_connode;}
std::vector<int> get_mstree(){std::vector<int> output; for(auto abs: mstree){output.push_back(SBasinOutlets[abs]);}; return output;}
std::vector<std::vector<int> > get_mstree_translated(){return this->mstree_translated;}

std::vector<double> get_distance_to_receivers_custom(int node, std::vector<int> list_of_receivers);

void virtual_filling(xt::pytensor<double,1>& elevation, xt::pytensor<bool,1>& active_nodes, int depression_ID, int starting_node);
void build_depression_tree(xt::pytensor<double,1>& elevation, xt::pytensor<bool,1>& active_nodes);
std::vector<int> get_all_childrens(int dep);
void update_fake_topography(xt::pytensor<double,1>& topography);
std::vector<int> get_next_building_round(xt::pytensor<double,1>& topography);

void collapse_depression_tree(xt::pytensor<int,2>& conn_basins, xt::pytensor<int,2>& conn_nodes, 
                                           xt::pytensor<double,1>& conn_weights, xt::pytensor<double,1>& elevation, int& basin0);
void update_topdep();
std::vector<int> update_receivers_explicit();


void recompute_multi_receveivers_and_donors(xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation, std::vector<int>& nodes_to_compute);

// // Flat surface resolver
// std::vector<int> Barnes2014_identify_flat(int starting_node, xt::pytensor<double,1>& elevation,xt::pytensor<bool,1>& active_nodes, int checker,  
//   std::queue<int>& HighEdge, std::queue<int>& LowEdge, std::vector<char>& is_high_edge, std::vector<char>& is_low_edge, std::map<int,int>&  this_flat_surface_node_index);

// void Barnes2014_AwayFromHigh(std::vector<int>& flat_mask, std::vector<int>& this_flat_surface_node, std::map<int,int>& this_flat_surface_node_index,
//  int checker, std::queue<int>& HighEdge, xt::pytensor<double,1>& elevation, double elev_check, std::vector<char>& is_high_edge, int& max_lab);

// void Barnes2014_TowardsLower(std::vector<int>& flat_mask, std::vector<int>& this_flat_surface_node, std::map<int,int>& this_flat_surface_node_index,
//  int checker, std::queue<int>& LowEdge, xt::pytensor<double,1>& elevation, double elev_check, std::vector<char>& is_low_edge, int max_lab);


xt::pytensor<int,1> get_flat_mask(){return flat_mask;};

int get_checker(int i, bool is_active);


// Return the upstream to downstream order of depressionns to solve according to Cordonnier et al., 2019
std::vector<int> get_Cordonnier_order();

double get_potential_depression_volume_at_node(int i){return this->depression_tree.potential_volume[i];};

void grow_depression_tree_v2(xt::pytensor<double,1>& elevation, xt::pytensor<bool,1>& active_nodes);
void raise_dep_to_new_node(int dep, int node, xt::pytensor<double,1>& elevation, xt::pytensor<bool,1>& active_nodes, bool integrate_node, int n_nodes_in_children);
void fill_the_depressions(std::vector<int>& next_to_check, xt::pytensor<double,1>& elevation, xt::pytensor<bool,1>& active_nodes);
void depression_initialisation(xt::pytensor<double,1>& elevation);



DepressionTree depression_tree;

// std::vector<int> top_depression;
// std::vector<int> bottom_depression;
// std::vector<double> potential_volume;

protected:

  // Node graph: vector of all vertexes in the DEM
  std::vector<Vertex> graph;
  // Integer number of elements
  int n_element;
  int nrows;
  int ncols;
  // Unsigned integer of number of element
  size_t un_element;
  // (deprecated?) cell area
  double cellarea;
  // X resolution
  double dx;
  // Y resolution
  double dy;

  //lake solving method
  bool lake_solver;

  // length is n_element and return true if the pit is to be rerouted
  std::vector<bool> pits_to_reroute;
  // The topological order of from top to bottom
  xt::pytensor<int,1> Mstack;
  xt::pytensor<int,1> index_in_Mstack;
  xt::pytensor<int,1> Sstack;
  xt::pytensor<int,1> SBasinID;
  xt::xtensor<int,1> mstree;
  std::vector<std::vector<int> > mstree_translated;


  std::vector<std::vector<int> > DEBUG_connbas;
  std::vector<std::vector<int> > DEBUG_connode;

  std::vector<int> SBasinOutlets;
  std::vector<int> pits;
  int nbasins;
  int npits;
  std::vector<int> not_in_stack;

  std::vector<int> empty_vector;

  // helpers
  std::vector<std::vector<int> > neightbourer;
  std::vector<double> lengthener;

  // Depression management: modified from Barnes et al



  // DEBUGGUER
  xt::pytensor<int,1> flat_mask;

};

// _unionfind_spec = [
//     ('_parent', nb.intp[:]),
//     ('_rank', nb.intp[:]),
// ]


// @nb.jitclass(_unionfind_spec)
class UnionFind
{
  public:
    UnionFind(int size)
    {
      this->_parent = xt::arange(0, size) ;
      this->_rank = xt::zeros<int>({size}) ;
    };

    void Union(int& x, int& y)
    {
      int xroot = this->Find(x);
      int yroot = this->Find(y);

      if (xroot != yroot)
      {
        if(this->_rank[xroot] < this->_rank[yroot])
            this->_parent[xroot] = yroot;
        else
        {
          this->_parent[yroot] = xroot;
          if(this->_rank[xroot] == this->_rank[yroot])
            this->_rank[xroot] ++;
        }
      }
    }

    int Find(int& x)
    {
      int xp = x,xc;
      while (true)
      {
        xc = xp;
        xp = this->_parent[xc];
        if (xp == xc)
          break;
      }
      this->_parent[x] = xc;
      return xc;
    }

    xt::xtensor<int,1> _parent;
    xt::xtensor<int,1> _rank;

};

// Topological order algorithm for multiple receivers adapted from FORTRAN
// Original author: Jean Braun And Guillaume Cordonnier
std::vector<int> multiple_stack_fastscape(int n_element, std::vector<Vertex>& graph, std::vector<int>& not_in_stack, bool& has_failed);


#endif