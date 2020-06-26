#ifndef Environment_CPP
#define Environment_CPP

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

#include "Environment.hpp"
#include "CHONK.hpp"
#include "nodegraph.hpp"

#include <boost/timer/timer.hpp>

// ######################################################################
// ######################################################################
// ######################################################################
// ###################### Model Runner ##################################
// ######################################################################
// ######################################################################

// Initialises the model object, actually Does not do much but is required.
void ModelRunner::create(double ttimestep,double tstart_time,std::vector<std::string> tordered_flux_methods, std::string tmove_method)
{
  // Getting all the attributes
  this->timestep = ttimestep;
  this->start_time = tstart_time;
  this->ordered_flux_methods = tordered_flux_methods;
  this->move_method = tmove_method;
}

// initialising the node graph and the chonk network
void ModelRunner::initiate_nodegraph()
{
  // creating the nodegraph and preprocessing the depression nodes
  this->graph = NodeGraph(this->io_int_array["pre_stack"],this->io_int_array["pre_rec"],this->io_int_array["post_rec"],this->io_int_array["post_stack"] , this->io_int_array["m_stack"], this->io_int_array2d["m_rec"],this->io_int_array2d["m_don"], 
    this->io_double_array["surface_elevation"], this->io_double_array2d["length"], this->io_double["x_min"], this->io_double["x_max"], this->io_double["y_min"], 
    this->io_double["y_max"], this->io_double["x_res"], this->io_double["y_res"], this->io_int["n_rows"], this->io_int["n_cols"], this->io_int["no_data"]);

  // Chonkification
  if(this->chonk_network.size()>0)
  {
    this->chonk_network.clear();
  }
  this->chonk_network = std::vector<chonk>();
  this->chonk_network.reserve(size_t(this->io_int["n_elements"]));

  // filling my network with empty chonks
  for(size_t i=0; i<size_t(this->io_int["n_elements"]); i++)
  {
    this->chonk_network.emplace_back(chonk(int(i), int(i), false));
  }

  

  // This add previous inherited water from previous lakes
  // Note that it "empties" the lake and reinitialise the depth. If there is still a reason to for the lake, it will form it
  // std::cout << "wat" << std::endl;
  this->process_inherited_water();
  // std::cout << "er" << std::endl;
}

void ModelRunner::run()
{

  // Alright now I need to loop from top to bottom
  for(int i=0; i<io_int["n_elements"]; i++)
  {
 
    int node = this->graph.get_MF_stack_at_i(i);
    // std::cout << "1:" << node << std::endl;

    // std::cout << "2" << std::endl;
    this->manage_fluxes_before_moving_prep(this->chonk_network[node]);

    if(this->graph.is_depression(node))
    {
      // std::cout << "A" << std::endl;
      int outlet = -1;
      outlet = this->solve_depression(node);
      // std::cout << "B: outlet:" << outlet << std::endl;
      if(outlet<0)
        continue;
      else
        node = outlet;
    }

    // first step is to apply the right move method, to prepare the chonk to move
    // std::cout << "3" << std::endl;
    this->manage_move_prep(this->chonk_network[node]);
    // std::cout << "4" << std::endl;
    this->manage_fluxes_after_moving_prep(this->chonk_network[node]);
    // std::cout << "5" << std::endl;
    this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->io_double_array["surface_elevation_tp1"], io_double_array["sed_height_tp1"], this->timestep);
    
  }
  this->finalise(); //TODO
  // std::cout << "GURG" << std::endl;

  // POTENTIAL OPTIMISATION HERE
  // turns out I need to copy these back into the map to get them out of the model, there should be a way to get that sorted
  this->io_int_array["m_stack"] = this->graph.get_MF_stack_full();
  this->io_int_array2d["m_rec"]= this->graph.get_MF_rec_full();
  this->io_int_array2d["m_don"]= this->graph.get_MF_don_full();


}


void ModelRunner::finalise()
{
  xt::pytensor<double,1>& surface_elevation_tp1 = this->io_double_array["surface_elevation_tp1"];
  xt::pytensor<double,1>& sed_height_tp1 = this->io_double_array["sed_height_tp1"];

  for(int i=0; i< this->io_int["n_elements"]; i++)
  {
    chonk& tchonk = this->chonk_network[i];
    surface_elevation_tp1[i] -= tchonk.get_erosion_flux() * timestep;
    surface_elevation_tp1[i] += tchonk.get_deposition_flux() * timestep;
    sed_height_tp1[i] -= tchonk.get_erosion_flux() * timestep;
    if(sed_height_tp1[i]<0)
      sed_height_tp1[i] = 0;
    sed_height_tp1[i] += tchonk.get_deposition_flux() * timestep;
    // if(isnan(surface_elevation_tp1[i]) )
    // {
    //   std::cout << "NAN BECAUSE::" << tchonk.get_deposition_flux() << "||" << tchonk.get_erosion_flux()  << "||" << this->io_double_array["surface_elevation_tp1"][i] << std::endl;
    // }
  }
}



void ModelRunner::manage_fluxes_before_moving_prep(chonk& this_chonk)
{

  std::map<std::string,int> intcorrespondance;
  intcorrespondance["drainage_area"] = 1;
  intcorrespondance["precipitation_discharge"] = 2;
  intcorrespondance["infiltration_discharge"] = 3;

  for(auto method:this->ordered_flux_methods)
  {
    if(method == "move")
      break;
    int this_case = intcorrespondance[method];

    switch(this_case)
    {
      case 1:
        this_chonk.inplace_only_drainage_area(this->io_double["x_res"], this->io_double["y_res"]);
        break;
      case 2:
        this_chonk.inplace_precipitation_discharge(this->io_double["x_res"], this->io_double["y_res"],this->io_double_array["precipitation"]);
        break;
      case 3:
        this_chonk.inplace_infiltration(this->io_double["x_res"], this->io_double["y_res"], this->io_double_array["infiltration"]);
        break;
    }

  }
}

void ModelRunner::manage_move_prep(chonk& this_chonk)
{

  std::map<std::string,int> intcorrespondance;
  intcorrespondance["D8"] = 1;
  intcorrespondance["D8_nodeps"] = 2;
  intcorrespondance["MF_fastscapelib"] = 3;
  int this_case = intcorrespondance[this->move_method];

  switch(this_case)
  {
    case 1:
      this_chonk.move_to_steepest_descent(this->graph, this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["surface_elevation"],  this->io_double_array["surface_elevation_tp1"], this->io_double["x_res"], this->io_double["y_res"], chonk_network);
      break;
    case 2:
      this_chonk.move_to_steepest_descent_nodepression(this->graph, this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["surface_elevation"],  this->io_double_array["surface_elevation_tp1"], this->io_double["x_res"], this->io_double["y_res"], chonk_network);
      break;

    case 3:
      this_chonk.move_MF_from_fastscapelib(this->graph, this->io_double_array2d["external_weigths_water"], this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["surface_elevation"],  this->io_double_array["surface_elevation_tp1"], this->io_double["x_res"], this->io_double["y_res"], chonk_network);
      break;

      
    default:
      std::cout << "WARNING::move method name unrecognised, not sure what will happen now, probably crash" << std::endl;
  }
}

void ModelRunner::manage_fluxes_after_moving_prep(chonk& this_chonk)
{
  bool has_moved = false;
  std::map<std::string,int> intcorrespondance;
  intcorrespondance["basic_SPIL"] = 1;
  for(auto method:this->ordered_flux_methods)
  {
    
    if(method == "move")
    {
       has_moved = true;
       continue;
    }

    if(has_moved == false)
      continue;

    int this_case = intcorrespondance[method];
    switch(this_case)
    {
      case 1:
        this_chonk.active_simple_SPL(this->io_double["SPIL_n"], this->io_double["SPIL_m"], this->io_double_array["erodibility_K"], this->timestep, this->io_double["x_res"], this->io_double["y_res"]);
        break;
    }
  }
  return;
}


//#################################################
//#################################################
//#################################################
//######### Solving depression ####################
//#################################################
//#################################################
//#################################################

// This is an internal c++ structure I use to sort my nodes into a priority queue
struct nodium
{
  /// @brief Elevation data.
  double elevation;
  /// @brief Row index value.
  int node;
};

// Managing the comparison operators for the fill node, to let the queue know I wanna compare it by elevation values
// lhs/rhs : left hand side, right hand side
bool operator>( const nodium& lhs, const nodium& rhs )
{
  return lhs.elevation > rhs.elevation;
}
bool operator<( const nodium& lhs, const nodium& rhs )
{
  return lhs.elevation < rhs.elevation;
}

// This function is the main algorithm managing depression solving
int ModelRunner::solve_depression(int node)
{
  // Gathering the depression rerouter, I'll need it to check when I reach another pit vs true base level
  // xt::pytensor<int,1>& depressions = this->io_int_array["depression_to_reroute"];
  xt::pytensor<double,1>& surface_elevation = this->io_double_array["surface_elevation"];
  xt::pytensor<double,1>& lake_depth = this->io_double_array["lake_depth"];
  // I am making these aliases to avoid the cost of accessing the map element each nodes

  // Getting the total volume of water arriving in this depression: Q * dt
  double water_wolume = this->chonk_network[node].get_water_flux() * timestep;
  // My first node to be processed is the pit
  nodium working_node; working_node.node = node; working_node.elevation = surface_elevation[node] + lake_depth[node];

  // Initialising the priority queue I will be using to processed my nodes
  std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> > depressionfiller;
  // Adding my first node in it
  // depressionfiller.push(working_node);
  // NO i am not doing that cause I'll be processing the first working node twice if I do so
  
  // I'll be keeping track of how many and which nodes are in this depression
  int n_nodes_underwater = 0;
  std::vector<int> underwater_nodes;
  std::vector<bool> is_underwater(this->io_int["n_elements"], false);
  std::vector<bool> is_in_queue(this->io_int["n_elements"], false);

  // My current water level is the elevation of this node + eventually preexisting lake water from THIS timestep
  double current_water_level = surface_elevation[node] + lake_depth[node];
  // I will track my outlet status
  int potential_outlet = -9999;
  // Temp boolean I need to break the loop in some specific cases (e.g. an edge is reach and my water escapes)
  bool break_main_loop = false;
  // Now filling the depression, doing it while I have enough water to do so
  while(water_wolume>0)
  {
    // std::cout << "bulf: " << working_node.node << ":" << working_node.elevation << std::endl;
    // At the start of this loop, my working node is in the depression
    n_nodes_underwater++;
    underwater_nodes.push_back(working_node.node);
    is_underwater[working_node.node] = true;
    is_in_queue[working_node.node] = true;

    // HEre, checking if this node outlets the depression
    // I am getting all the receivers of the current working node, i.e. the downslope neighbors of my current one
    std::vector<int> recnodes = graph.get_MF_receivers_at_node(working_node.node);
    std::vector<int> extra_donodes;
    // iterating through each of them
    for(auto nenode:recnodes)
    {
      // Checking if valid (-1 = not a downslope neighbour)
      if(nenode<0)
        continue;

     // If I am at the first node, my receiver will be meself but I still want to keep on processing the pit so I am ignoring this check (node being the pit bottom)
      if(nenode != node)
      {
        // If my node is its own receiver, I am outletting outside and loosing my fluxes
        if(nenode == working_node.node && this->graph.is_depression(nenode) == false)
        {
          // std::cout << "OUTLETTING THE MODEL:" << nenode << std::endl; 
          // I am therefore at a baselevel and the fluxes can escape
          potential_outlet = -1;
          // i'll stop with this depression
          break_main_loop = true;
        }

        if(is_underwater[nenode] == false && lake_depth[nenode] <=0)
        {
          if(surface_elevation[nenode] + lake_depth[nenode] < surface_elevation[working_node.node] + lake_depth[working_node.node] )
          {
            // outlet! transmitting the water flux to this receiver
            potential_outlet = working_node.node;
            // std::cout << "OUTLETTING THE LAKE:" << potential_outlet << std::endl; 

            break_main_loop = true;
            // std::cout << "C" << std::endl;
            std::vector<int> recout = this->graph.get_MF_receivers_at_node(potential_outlet);
            // std::cout << "D" << std::endl;
            for(auto& rn:recout)
            {
              if(rn == -1)
                continue;
              if(is_underwater[rn])
                rn = -1;
            }
            // std::cout << "E" << std::endl;

            this->graph.update_receivers_at_node(potential_outlet,recout);
            // I am also reprocessing this chonk so I need to reset it to reinitialise its info
            // this->chonk_network[potential_outlet].reset();
            // std::cout << "F" << std::endl;
            // 
            this->chonk_network[potential_outlet].add_to_water_flux(water_wolume / this->timestep);
            // std::cout << "G" << std::endl;
          }
        }
        if(surface_elevation[nenode] + lake_depth[nenode] >= working_node.elevation)
        {
          extra_donodes.push_back(nenode);
        }
      }
    }
    // done with checking receiver

    // I am getting out of the loop if conditions are met
    if(break_main_loop)
      break;

    // otherwise I can just add all my neighbours into the queue
    // getting all upslope neighbors
    std::vector<int> donodes = graph.get_MF_donors_at_node(working_node.node);
    for(auto onode:extra_donodes)
      donodes.push_back(onode);
    // iterating thorough these
    for(auto nenode:donodes)
    {
      // checking if valid
      if(nenode<0)
        continue;
      if(is_in_queue[nenode])
        continue;
      // pushing this node in the queue: it will get sorted automatically in the queue
      // Note that I am adding the node with its elevation
      nodium newnode; newnode.node = nenode; newnode.elevation = surface_elevation[nenode] + lake_depth[nenode];
      depressionfiller.push(newnode);
      is_in_queue[nenode] = true;
    }

    // nodium next_node;

    // if(depressionfiller.empty() == false)
    // {
    //   // My potential next node is the top priority on the Queue
    //   next_node = depressionfiller.top();
    //   depressionfiller.pop(); // removing it from the queue
    //   water_wolume -= n_nodes_underwater * this->io_double["x_res"] * this->io_double["y_res"] * (next_node.elevation - current_water_level);
    //   current_water_level = next_node.elevation;
    // }
    // else
    // {
    //   current_water_level += water_wolume / (n_nodes_underwater * this->io_double["x_res"] * this->io_double["y_res"]);
    //   break;
    // }

    if(depressionfiller.empty())
      break;

    // My potential next node is the top priority on the Queue
    nodium next_node = depressionfiller.top();
    depressionfiller.pop(); // removing it from the queue

    // I am rising my whole lake level to the one of the neighbor, and removing the volume added to the total water
    water_wolume -= n_nodes_underwater * this->io_double["x_res"] * this->io_double["y_res"] * (next_node.elevation - current_water_level);
    current_water_level = next_node.elevation;

    // my next node to process is that one
    working_node = next_node;

    //
    if(water_wolume <0)
    {
      // I have too much water
      double excess_water = abs(water_wolume);
      current_water_level -= excess_water / (n_nodes_underwater * this->io_double["x_res"] * this->io_double["y_res"]);
      break;
    }
    // Moving to my next node if I still have some water
  
  } 

  // Right now let's take care of the sediment
  // I have a given volume of sediment
  double sediment_volume =  this->chonk_network[node].get_sediment_flux();
  double depression_volume = 0;
  // last thing about water: I need to change the lake depth for the nodes underwater, 
  // At the same time I will cancel any erosion/deposition that happened in these CHONKs
  for( auto this_node:underwater_nodes) 
  {
    // registering the lake deposition flux (converted into in rate of height change by unit of time) in case some lake deposition already happened in an upstream sub-depression
    double lake_deposition_flux_already_there = this->chonk_network[this_node].get_other_attribute("height_lake_sediments_tp1")/this->timestep;
    // Adding lake depth
    lake_depth[this_node] = current_water_level - surface_elevation[this_node];
    // Note I am acknowledging the sediment height already deposited
    depression_volume += (lake_depth[this_node] - lake_deposition_flux_already_there * this->timestep) * this->io_double["x_res"] * this->io_double["y_res"];
    // Registering current erosion
    double eflux = this->chonk_network[this_node].get_erosion_flux();
    // cancelling all erosion
    this->chonk_network[this_node].set_erosion_flux(0.);
    // removing the cancelled erosion due to covid-19 from the sediment flux (note the conversion from rate of elevation change for one unit of time to a volume)
    sediment_volume += -1*(this->io_double["x_res"] * this->io_double["y_res"] * eflux * this->timestep);
    sediment_volume += (this->chonk_network[this_node].get_deposition_flux() - lake_deposition_flux_already_there) * this->io_double["x_res"] * this->io_double["y_res"] * timestep ;
    this->chonk_network[this_node].set_sediment_flux(0.);
        // I am readding the cancelled deposition to the sediment flux (conversion to volume)
    // Setting my deposition flux to only my lake one
    this->chonk_network[this_node].set_deposition_flux(lake_deposition_flux_already_there);
    // std::cout << lake_deposition_flux_already_there << std::endl;
  } 

  
  // std::cout << sediment_volume << std::endl;
  // I can fill a certain proportion of the depression
  double proportion_filled = sediment_volume/water_wolume;

  // Cheking if I have an excess of sediment and if I can transmit them to a receiver
  double excess_sediments = 0;
  if(proportion_filled>1)
  {
    excess_sediments = proportion_filled - 1;
    // if my receiver is more than -1: it has a receiver which can receive the excess
    if(potential_outlet > -1)
    {
      proportion_filled = 1;
      this->chonk_network[potential_outlet].add_to_sediment_flux(excess_sediments * sediment_volume);
    }
    else if(potential_outlet == -1)
    {
      // in that case, the sediment can escape the model simply
      proportion_filled = 1;
    }
    // Last case, potential_outlet == -9999, it would mean I have more sediment volume than water but no outlet at all! 
    // It may happen in very small depression eventually. I chose to keep the mass balance and depositate a sediment "column", this can require work if I start to see unreasonable columns appearing.
  }

  // Applying the change
  for(auto this_node:underwater_nodes)
  {
    double this_depo = (lake_depth[this_node] - this->chonk_network[this_node].get_other_attribute("height_lake_sediments_tp1")) * proportion_filled;
    // std::cout << this_depo << "||" << proportion_filled << "||" << sediment_volume <<std::endl;
    this->chonk_network[this_node].set_other_attribute("height_lake_sediments_tp1", this_depo);
    this->chonk_network[this_node].add_deposition_flux(this_depo/this->timestep);
  }

  this->io_double_array["lake_depth"] = lake_depth;

  return potential_outlet;

}


void ModelRunner::process_inherited_water()
{
  // At the moment, I cannot really be fucked, let's just ttransfer it to the receiver
  for(size_t i = 0; i < this->io_int["n_elements"]; i++)
  {
    int this_node = int(i);
    double this_lake_depth = this->io_double_array["lake_depth"][this_node];
    if(this_lake_depth >0)
    {
      // int rec = this->io_int_array["post_rec"][this_node];
      // int rec = this_node;
      // this->chonk_network[rec].add_to_water_flux(this_lake_depth * this->io_double["x_res"] * this->io_double["y_res"]);
      this->io_double_array["lake_depth"][this_node] = 0;
    }
  }
}

//#################################################
//#################################################
//#################################################
//######### extracting attributes #################
//#################################################
//#################################################
//#################################################



xt::pytensor<double,1> ModelRunner::get_water_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_water_flux();
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_erosion_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_erosion_flux();
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_sediment_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_sediment_flux();
  }
  return output;

}


xt::pytensor<double,1> ModelRunner::get_other_attribute(std::string key)
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_other_attribute(key);
  }
  return output;
}





//#################################################
//#################################################
//#################################################
//######### DEBUGGING functions ###################
//#################################################
//#################################################
//#################################################

void ModelRunner::DEBUG_check_weird_val_stacks()
{
  int non_valid_mfstack = 0;
  int non_valid_mfrec = 0;
  int non_valid_don = 0;
  int non_valid_mflength = 0;
  for (size_t i=0; i<this->io_int["n_elements"]; i++)
  {
    if(this->graph.get_MF_stack_at_i(i)<0 || this->graph.get_MF_stack_at_i(i)>=io_int["n_elements"])
      non_valid_mfstack++;

    std::vector<int> rec = this->graph.get_MF_receivers_at_node(i);
    for (auto node:rec)
    {
      if(node<-2 || node>=io_int["n_elements"])
        non_valid_mfrec++;
    }
    
    std::vector<int> gurg = this->graph.get_MF_donors_at_node(i);
    for (auto node:gurg)
    {
      if(node<-2 || node>=io_int["n_elements"])
        non_valid_don++;
    }
    // std::vector<int> rec = this->graph.get_MF_donors_at_node(i);
    // for (auto node:rec)
    // {
    //   if(node<0 || (node>=io_int["n_elements"]))
    //     non_valid_don++;
    // }
  }
  std::cout << "FOUND N INVALID: " << non_valid_mfstack << " MSTACK || " << non_valid_mfrec << " MREC || " << non_valid_don << " MDON" << std::endl;
}






//#################################################################################
//#################################################################################
//#################################################################################
//#################################################################################
// OTher functions, with a clear one-off utility
//#################################################################################
//#################################################################################
//#################################################################################



xt::pytensor<double,1> pop_elevation_to_SS_SF_SPIL(xt::pytensor<int,1>& stack, xt::pytensor<int,1>& rec,xt::pytensor<double,1>& length , xt::pytensor<double,1>& erosion, 
  xt::pytensor<double,1>& K, double n, double m, double cellarea)
{
  // first I need to initialise a couple of variabes
  xt::pytensor<double,1> A = xt::zeros<double>({stack.size()});
  xt::pytensor<double,1> elevation = xt::zeros<double>({stack.size()});

  std::vector<int> ndonors(stack.size(),0);
  // Accunulating drainage area first
  for(int i =  stack.size(); i>=0; i--)
  {
    int this_node = stack[i];
    A[this_node] += cellarea;
    int this_rec = rec[this_node];
    if(this_rec == this_node)
      continue;
    A[this_rec] += A[this_node];
    ndonors[this_rec] += 1;
  }


  for(auto node:stack)
  {
    if(node == rec[node])
      continue;
    int this_rec = rec[node];
    elevation[node] = elevation[this_rec] + std::exp(1/n * (std::log(erosion[node]) - m *std::log(A[node]) - std::log(K[node]) ) + std::log(length[node]) );
  }

  return elevation;

}





#endif

