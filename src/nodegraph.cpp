#ifndef nodegraph_CPP
#define nodegraph_CPP

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
#include "xtensor/xsort.hpp"
#include "xtensor/xbuilder.hpp"
#include <iostream>
#include <numeric>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <queue>
#include "nodegraph.hpp"
#include <chrono>



bool SAFE_STACK_STOPPER = false;
bool EXTENSIVE_STACK_INFO = false;

// // nodiums are sorted by elevations for the depression filler
// bool operator>( const nodium& lhs, const nodium& rhs )
// {
//   return lhs.elevation > rhs.elevation;
// }
// bool operator<( const nodium& lhs, const nodium& rhs )
// {
//   return lhs.elevation < rhs.elevation;
// }

void set_DEBUG_switch_nodegraph(std::vector<std::string> params, std::vector<bool> values )
{
  for(size_t i=0; i<params.size();i++)
  {
    std::string param = params[i];
    bool value = values[i];
    std::cout << "Setting " << param << " to " << value << std::endl;
    if(param == "SAFE_STACK_STOPPER")
      SAFE_STACK_STOPPER = value;
    else if (param == "EXTENSIVE_STACK_INFO")
      EXTENSIVE_STACK_INFO = value;
  }
}



NodeGraphV2::NodeGraphV2(
  xt::pytensor<double,1>& elevation,
  xt::pytensor<bool,1>& active_nodes,
  double dx,
  double dy,
  int nrows,
  int ncols,
  bool lake_solver
  )
{
  // Saving general informations
  this->dx = dx;
  this->dy = dy;
  this->cellarea = dx * dy;
  this->n_element = nrows*ncols;
  this->un_element = nrows*ncols;
  this->nrows = nrows;
  this->ncols = ncols;
  this->lake_solver = lake_solver;

  this->flat_mask = xt::zeros<int>({this->un_element}) - 1;
  this->depression_tree = std::vector<Depression>();
  // this->depression_tree.reserve(100);
  this->top_depression = std::vector<int>(this->un_element,-1);

  // these vectors are additioned to the node indice to test the neighbors
  this->neightbourer.emplace_back(std::initializer_list<int>{-ncols - 1, - ncols, - ncols + 1, -1,1,ncols - 1, ncols, ncols + 1 }); // internal node 0
  this->neightbourer.emplace_back(std::initializer_list<int>{(nrows - 1) * ncols - 1, (nrows - 1) * ncols, (nrows - 1) * ncols + 1, -1,1,ncols - 1, ncols, ncols + 1 });// periodic_first_row 1
  this->neightbourer.emplace_back(std::initializer_list<int>{-ncols - 1, - ncols, - ncols + 1, -1,1,- (nrows - 1) * ncols - 1, - (nrows - 1) * ncols, - (nrows - 1) * ncols + 1 });// periodic_last_row 2
  this->neightbourer.emplace_back(std::initializer_list<int>{- 1, - ncols, - ncols + 1, (ncols - 1),1, 2 * ncols - 1, ncols, ncols + 1 }); // periodic_first_col 3
  this->neightbourer.emplace_back(std::initializer_list<int>{-ncols - 1, - ncols, - 2 * ncols + 1, -1,-ncols + 1, ncols - 1, ncols, 1 }); // periodic last_col 4
  this->neightbourer.emplace_back(std::initializer_list<int>{ -1,1,ncols - 1, ncols, ncols + 1 }); // normal_first_row 5
  this->neightbourer.emplace_back(std::initializer_list<int>{-ncols - 1, - ncols, - ncols + 1, -1,1}); // normal_last_row 6
  this->neightbourer.emplace_back(std::initializer_list<int>{ - ncols, - ncols + 1, 1,  ncols, ncols + 1 }); // normal_first_col 7
  this->neightbourer.emplace_back(std::initializer_list<int>{-ncols - 1, - ncols, -1, ncols - 1, ncols }); // normal_last_col 8
  this->neightbourer.emplace_back(std::initializer_list<int>{1, ncols, ncols + 1 }); // normal_top_left 9
  this->neightbourer.emplace_back(std::initializer_list<int>{ -1,ncols - 1, ncols}); // normal_top_right 10
  this->neightbourer.emplace_back(std::initializer_list<int>{ - ncols, - ncols + 1, 1}); // normal_bottom_left 11
  this->neightbourer.emplace_back(std::initializer_list<int>{-ncols - 1, - ncols, -1}); // normal_bottom_right 12


  this->is_border = std::vector<char>(this->un_element,'y');
  for(size_t i=0; i <this->un_element; i++)
  {
    if(active_nodes[i])
      is_border[i] = 'n';
  }

  double diag = std::sqrt(std::pow(dx,2) + std::pow(dy,2));
  this->lengthener = {diag,dy,diag,dx,dx,diag,dy,diag};

  // Initialising the vector of pit to reroute. The pits to reroute are all the local minima that are rerouted to another basin/edge
  // It does not include non active nodes (i.e. base level of the model/output of the model)
  this->pits_to_reroute = std::vector<bool>(un_element,false);

    graph.reserve(un_element);
  for(int i=0; i<n_element;i++)
    graph.emplace_back(Vertex());

  // computing receivers and donors information, multi and SS flow
  this->compute_receveivers_and_donors(active_nodes,elevation);
  // computing the original Single flow topological order
  this->Sstack = xt::zeros<int>({this->n_element});

  this->compute_stack();


  // Computing basin info from stack
  this->compute_basins(active_nodes);
  this->compute_pits(active_nodes);

  //# Iterating through the nodes on the stack to gather all the pits
  for(auto node:this->Sstack)
  {
    // If a pit is its single-flow receiver and an active node it is a pit to reroute
    if(this->graph[node].Sreceivers == node && active_nodes[node])
    {
      this->pits_to_reroute[node] = true;
    }
  }

  this->build_depression_tree(elevation, active_nodes);


  // Now I am labelling my basins
  //# Initialising the vector of basin labels
  std::vector<int> basin_labels(this->Sstack.size(),-1);
  //# Initialising the label to 0
  int label = 0;
  //# Iterating through all my single-flow stack from bottom to top (see Braun and Willett 2013 for meaning of stack)
  //# if a node is it's receiver then the label becomse that node. The resulting basin vector show which nodes are linked to which
  for(auto node:this->Sstack)
  {
    if(node == this->graph[node].Sreceivers)
    {
      label = node;
    }
    basin_labels[node] = label;
  }

  this->correct_flowrouting(active_nodes, elevation);

  //# I will need a vector gathering the nodes I'll need to check for potential cyclicity
  std::vector<int> node_to_check;
  //# their target basin
  std::vector<int> force_target_basin;
  //# and the origin of the pit
  std::vector<int> origin_pit;
  if(this->lake_solver == false)
  {  // Initialising the node graph, a vector of Vertexes with their edges

      //#Iterating through the nodes
      for(int i=0; i<n_element;i++)
      {
        // If this receiver is a pit to reroute, I need to add to the receiver list the receiver of the outlet
        if(pits_to_reroute[i] == true)
        {
          //# Node I wanna add
          int tgnode = this->graph[i].Sreceivers;
          if(pits_to_reroute[tgnode] == false)
            tgnode = this->graph[tgnode].Sreceivers;
          //# Will be a receiver
          this->graph[i].receivers.emplace_back(tgnode);
          //# I will have to check its own receivers for potential cyclicity
          node_to_check.emplace_back(tgnode);
          //# Which pit is it connected to
          origin_pit.emplace_back(i);
          //# Arbitrary length
          this->graph[i].length2rec.emplace_back(dx*10000.);
          //# The basin I DONT want to reach
          force_target_basin.emplace_back(i);
  
          // Keepign this check just in case, will remove later. Throw an error in case cyclicity is detected
          if(basin_labels[tgnode] == basin_labels[i])
          {
  
            throw std::runtime_error("Receiver in same basin! Node " + std::to_string(i) + " gives to " + std::to_string(this->graph[i].Sreceivers) + " gives to " + std::to_string(tgnode));
          }
        }
  
        // // I need these to initialise my Vertex
        // bool false1 = false;
        // bool false2 = false;
        // // Constructing the vertex in place in the vector. More efficient according to the internet
        // graph.emplace_back(Vertex(i,false1,false2,donors,receivers,length2rec));
      }
  
      // Now I am correcting my outlet nodes:
      // The idea is to force any of its node to be directed to the next basin and avoid any cyclicity. 
      //# iterating through them
      for(size_t i=0; i< node_to_check.size();i++)
      { 
        //# Gathering the node to check
        int this_node_to_check = node_to_check[i];
        //# The basin to avoid
        int this_target_basin = force_target_basin[i];
        // new list of receivers, I am only keeping the ones NOT draining to original basin
        std::vector<int> new_rec;
        std::vector<double> new_length;
        int idL=0;
        for(auto trec:graph[this_node_to_check].receivers)
        {
          if(basin_labels[trec] != this_target_basin  || active_nodes[trec] == false) //|| trec == this_node_to_check??
          {
            new_rec.emplace_back(trec);
            new_length.emplace_back(this->graph[this_node_to_check].length2rec[idL]);
          }
          idL++;
        }
        // Security check
        if(new_rec.size()==0 && active_nodes[this_node_to_check] == 1 )
          throw std::runtime_error("At node " + std::to_string(this_node_to_check) + " no receivers after corrections! it came from pit " + std::to_string(origin_pit[i]) +" and flat mask is " + std::to_string(this->flat_mask[this_node_to_check]));
        // Correcting the rec
        this->graph[this_node_to_check].receivers = new_rec;
        this->graph[this_node_to_check].length2rec = new_length;
      }
  }

  // I am now ready to create my topological order utilising a c++ port of the fortran algorithm from Jean Braun
  bool has_failed = false;
  Mstack = xt::adapt(multiple_stack_fastscape( n_element, graph, this->not_in_stack, has_failed));


  // DEBUG CHECKING
  // this->is_MF_outet_SF_outlet(); 
  if(this->lake_solver == false)
  {
    // I got my topological order, I can now restore the corrupted receiver I had
    for(size_t i=0; i<node_to_check.size(); i++)
    {
      // // New receiver array
      // std::vector<int> rec;
      // // Repicking the ones from fastscapelib-fortran
      // for(size_t j=0;j<8; j++)
      // {
      //   int trec = Mrec(node_to_check[i],j);
      //   if(trec < 0 || node_to_check[i] == trec)
      //   {continue;}
      //   rec.emplace_back(trec);
      // }
      std::vector<int> to_recompute = {node_to_check[i]};
      this->recompute_multi_receveivers_and_donors(active_nodes,elevation,to_recompute);

      // VERY IMPORTANT HERE!!!!!
      // In the particular case where my outlet is *also* a pit, I do not want to remove its receiver Y though???
      // if(pits_to_reroute[node_to_check[i]])
      // {
      //   this->graph[node_to_check[i]].receivers.emplace_back(this->graph[this->graph[node_to_check[i]].Sreceivers].Sreceivers);
      //   this->graph[node_to_check[i]].length2rec.emplace_back(dx * 10000.);
      // }

      // // Correcting the Vertex inplace
      // graph[node_to_check[i]].receivers = rec;
    }
  }

  // Last step: implementing the index in the stack
  this->index_in_Mstack = xt::zeros<int>({this->n_element});
  for(int i = 0; i< this->n_element ; i++)
    this->index_in_Mstack[this->Mstack[i]] = i;

  // std::cout << "PITS TO REROUTE::";
  // for(size_t i=0 ; i< this->un_element; i++)
  // {
  //   if(pits_to_reroute[i])
  //     std::cout << i << "||";
  // }
  // std::cout << std::endl;



  // #########################################
  // ################ DEBUG ##################
  // #########################################
  // uncomment to check if the node graph produced duplicated receivers
  for(int i = 0; i< this->n_element ; i++)
  {
    std::set<int> count_rec;

    for(auto nono:this->graph[i].receivers)
    {
      if (count_rec.find(nono) != count_rec.end())
        throw std::runtime_error("DuplicatedRecError:: node graph has duplicate in the MFD receiers");
      else
        count_rec.insert(nono);
      if(nono == i)
        throw std::runtime_error("DuplicatedRecError:: node graph giving to itself in the MFD receiers");

    }
  }

  
  //Done


  return;
}

std::vector<int> NodeGraphV2::get_Cordonnier_order()
{
  std::vector<int> output;
  for(int i = this->n_element - 1; i >= 0; i-- )
  {
    if(this->is_depression(i))
      output.emplace_back(i);
  }
  return output;
}

void NodeGraphV2::build_depression_tree(xt::pytensor<double,1>& elevation, xt::pytensor<bool,1>& active_nodes)
{
  // I am not running this code if there is no piut to reroute
  if(this->npits == 0)
    return;

  std::cout << "DEBUGDEP::building the Tree" << std::endl;

  // Initialising the potential volume vector
  this->potential_volume = std::vector<double>(this->n_element,0.);
  // this->depression_tree.reserve();

  // current depression ID
  int current_ID = -1;

  // initial build
  //# Iterating through all nodes
  for(int i = 0; i < this->n_element; i++)
  {
    // # is a pit?
    if(this->pits_to_reroute[i] == false)
      continue;
    
    //# Yes, then:
    //## Incrementing the ID
    current_ID++;
    //## Building the depression with no parent and level 0
    this->depression_tree.emplace_back(Depression(current_ID,-1,0, i));
    //## Pushing the initial node in it
    this->depression_tree[current_ID].nodes.push_back(i);

    //## Priority Flood - like algorithm to label the depression
    this->virtual_filling(elevation,active_nodes,current_ID,i);
  }
  std::cout << "DEBUGDEP::1" << std::endl;

  // Init. a switch to detect when the processing is done
  bool keep_processing = true;
  if(this->depression_tree.size() == 0)
    keep_processing = false;

  // Init a fake topography "filling" the lakes
  auto topography = xt::pytensor<double,1>(elevation);

  // Current depression level
  int level = 0;

  // Checking whether dep is done yet
  std::vector<char> dep_is_done(this->depression_tree.size(), 'n');

  while(keep_processing)
  {
    // Increasing depression level
    level++;
    std::cout << "DEBUGDEP::2::level" << level << std::endl;
    this->update_fake_topography(topography);

    // Check whether there is a need to reprocess depressions
    std::vector<int> next_deps = this->get_next_building_round(topography);



    // If there is no more deps to process, I stop here
    if(next_deps.size() == 0)
      break;

    // Updating the fake topogrpahy for the next round
    this->update_topdep();

    std::cout << "DEBUGDEP::2::GABUL" << level << std::endl;

    // Otherwise I start a new round
    for (auto depID : next_deps)
    {
      std::cout << "!->" << this->depression_tree[depID].hw_max << std::endl;
      // Depressions will merge now, so I need to make sure I am not double processing them
      if(dep_is_done[depID] == 'y')
      {
        std::cout << "done" << std::endl;
        continue;
      }
      dep_is_done[depID] = 'y';
      dep_is_done[this->top_depression[this->depression_tree[depID].connections.second]] = 'y';

      //## Incrementing the ID
      current_ID++;
      //## Building the depression with no parent and level 0
      this->depression_tree.emplace_back(Depression(current_ID,-1,level,this->depression_tree[depID].connections.first));
      dep_is_done.push_back('n');

      if(this->top_depression[this->depression_tree[depID].connections.first] != depID)
        std::cout << "ASJKDHLKSDFHL" << std::endl;

      // updating the parents of the child depressions
      this->depression_tree[depID].parent = current_ID;
      this->depression_tree[this->top_depression[this->depression_tree[depID].connections.second]].parent = current_ID;
      // And the children of the current one
      this->depression_tree[current_ID].children = {this->top_depression[this->depression_tree[depID].connections.first], this->top_depression[this->depression_tree[depID].connections.second]};
      this->depression_tree[current_ID].has_children = true;

      //## Pushing the initial node in it, here it is by convention the outlet of the first depresiion (it does not matter much)
      this->depression_tree[current_ID].nodes.push_back(this->depression_tree[depID].connections.first);

      //## Priority Flood - like algorithm to label the depression, but with the fake topo so that it is merging the two depressions
      this->virtual_filling(topography,active_nodes,current_ID,this->depression_tree[depID].connections.first);
    }
  }

  std::cout << "DEBUGDEP::3" << std::endl;

  // After the previous loop, I can create a correspondance tree between basins
  for (auto& dep:this->depression_tree)
  {
    if(dep.parent > -1)
    {
      //# This is straightforward for child depression as their connections are already part of other depressions by definitions
      dep.connections_bas =  std::pair<int,int>({this->top_depression[dep.connections.first], this->top_depression[dep.connections.second]}) ;
    }
    else
    {
      //# This operation is a tad more tedious for cases if the depression is an orphan, good new is that it should not matter to which depression it is linked as long as it is a DS one
      //# So we can simply follow the steepest descent route until reaching another depression or an outlet downstream.
      int node = dep.connections.second;
      while (this->top_depression[node] == -1 && active_nodes[node])
      {
        node = this->graph[node].Sreceivers;
      }
      dep.connections_bas =  std::pair<int,int>({this->top_depression[dep.connections.first], this->top_depression[node]}) ;
    }

  }
  // This is me done
  std::cout << "DEBUGDEP::Tree Builded" << std::endl;

}


std::vector<int> NodeGraphV2::get_next_building_round(xt::pytensor<double,1>& topography)
{
  std::vector<int> output;
  for (auto& dep:this->depression_tree)
  {

    // checking whether top-depression
    if(dep.parent == -1)
    {
      if(this->top_depression[dep.connections.second] >= 0 && topography[dep.connections.second] == topography[dep.connections.first])
      {
        output.push_back(dep.index);
      }

    }
  }
  return output;
}

void NodeGraphV2::update_fake_topography(xt::pytensor<double,1>& topography)
{
  for (auto& dep:this->depression_tree)
  {
    if(dep.parent == -1)
    {
      // std::cout << "2.1" << std::endl;
      auto vec = this->get_all_childrens(dep.index);
      // std::cout << "2.2" << std::endl;
      for (auto i : vec)
      {
        for (auto n: this->depression_tree[i].nodes)
        {
          topography[n] = dep.hw_max;
          // this->top_depression[n] = dep.index;
        }
      }
      // std::cout << "2.3" << std::endl;
    }
  }

}

void NodeGraphV2::update_topdep()
{
  for (auto& dep:this->depression_tree)
  {
    if(dep.parent == -1)
    {
      // std::cout << "2.1" << std::endl;
      auto vec = this->get_all_childrens(dep.index);
      // std::cout << "2.2" << std::endl;
      for (auto i : vec)
      {
        for (auto n: this->depression_tree[i].nodes)
        {
          // topography[n] = dep.hw_max;
          this->top_depression[n] = dep.index;
        }
      }
      // std::cout << "2.3" << std::endl;
    }
  }

}

// gather all children (direct and indirect) of a depression
std::vector<int> NodeGraphV2::get_all_childrens(int dep)
{
  std::vector<int> output;
  std::queue<int> next;
  next.push(this->depression_tree[dep].index);
  while(next.size() > 0)
  {
    // for (auto child:this->depression_tree[next.front()].children)
    // {
    if(this->depression_tree[next.front()].has_children)
    {
      next.push(this->depression_tree[next.front()].children.first);
      next.push(this->depression_tree[next.front()].children.second);
    }
    
    // }
    output.push_back(next.front());
    next.pop();
  }
  return output;
}

void NodeGraphV2::virtual_filling(xt::pytensor<double,1>& elevation, xt::pytensor<bool,1>& active_nodes, int depression_ID, int starting_node)
{
  // priotrity queue to virtually fill the lake. This container stores elevation nodes and sort them on the go by
  // elevation value. I am filling one node at a time and gathering all its neighbors in this queue
  std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> > depressionfiller;

  // Aliases and shortcups
  double cellarea = this->dx * this->dy;

  // Starting by adding the first node of the list
  depressionfiller.emplace(nodium(starting_node, elevation[starting_node]));

  // I also set the outlet to a values I can recognise as NO OUTLET
  int outlet = -9999;

  // is_in queue is an helper char array to check wether a node is already in the queue or not
  // 'y' -> yes, 'n' -> no. Not using boolean for memory otpimisation. See std::vector<bool> on google for why
  std::vector<char> is_in_queue(this->n_element,'n');
  // My first node, is in that queue
  is_in_queue[starting_node] = 'y';
  // Similar helper here but for which node is in this lake
  std::vector<char> is_in_lake(this->n_element,'n');

  // Current Water Elevation
  double current_hw = elevation[starting_node];
  this->depression_tree[depression_ID].hw_max = current_hw;

  // Starting the main loop
  while(outlet < 0)
  {
    // first get the next node in line. If first iteration, the entry node, else, the closest elevation
    nodium next_node = depressionfiller.top();
    // then pop the next node
    depressionfiller.pop();
    // go through neighbours manually and either feed the queue or detect an outlet
    std::vector<int> neightbors; std::vector<double> dummy ; this->get_D8_neighbors(next_node.node, active_nodes, neightbors, dummy);

    // For each of the neighbouring node: checking their status
    for(auto tnode:neightbors)
    {
      // if already in the queue: I pass
      if(is_in_queue[tnode] == 'y')
        continue;

      // If flat or higher AND in the same depression system: I keep
      if(
          elevation[tnode] >= elevation[next_node.node] && ( this->top_depression[tnode] == -1 || (
          ( 
            this->top_depression[tnode] == this->depression_tree[depression_ID].children.first || 
            this->top_depression[tnode] == this->depression_tree[depression_ID].children.second
          ) && this->depression_tree[depression_ID].has_children == true) 
          || this->depression_tree[depression_ID].has_children == false)
        )
      {
        depressionfiller.emplace(nodium(tnode, elevation[tnode]));
        is_in_queue[tnode] = 'y';
      }
      else
      {
        // if a single neighbour is lower (and not in the queue!) then the current node is the outlet
        outlet = next_node.node;
        // And registering the connections
        this->depression_tree[depression_ID].connections = {next_node.node, tnode};
        this->top_depression[outlet] = depression_ID; // Keeping track of the outlet aspart of the depression system, even if it has 0 volume to store
        this->depression_tree[depression_ID].nodes.push_back(next_node.node);
        std::cout<< "Dep is " << depression_ID << " and outlet is " << outlet << " which give in dep" << this->top_depression[tnode] << std::endl;
        break;
      }
    }

    // calculating the xy surface of the lake
    double area_component_of_volume = int(this->depression_tree[depression_ID].nodes.size()) * this->dx * this->dy;
    // Calculating the maximum volume to add until next node
    double dV = (next_node.elevation - current_hw) * area_component_of_volume;

    current_hw = next_node.elevation;

    this->depression_tree[depression_ID].hw_max = current_hw;

    this->depression_tree[depression_ID].volume += dV;
    this->potential_volume[this->depression_tree[depression_ID].nodes[this->depression_tree[depression_ID].nodes.size()-1]] = this->depression_tree[depression_ID].volume;

    if(outlet < 0)
    {
      this->top_depression[next_node.node] = depression_ID;
      this->depression_tree[depression_ID].nodes.push_back(next_node.node);
    }
  }
  std::cout << "OUT:" << outlet << std::endl;;

}


void NodeGraphV2::fix_cyclicity(
  std::vector<int>& node_to_check,
  xt::pytensor<int,1>& Sstack,
  xt::pytensor<int,1>& Srec,
  xt::pytensor<int,1>& Prec,
  xt::pytensor<int,2>& Mrec,
  int correction_level


  )
{
  std::vector<bool> is_not_in_stack(this->un_element,false);
  for(auto& node:this->not_in_stack)
    is_not_in_stack[node] = true;

  std::vector<int> nodes_to_double_check;


  // TRY 2:
  // Label the basins and identify the pits in the nodes to correct
  std::vector<int> baslab(this->un_element);
  int tlabel = Sstack[0];
  for(size_t i=0; i< this->un_element; i++)
  {
    if(Sstack[i] == Srec[Sstack[i]])
    {
      tlabel = Sstack[i];
      nodes_to_double_check.emplace_back(Sstack[i]);

    }
    baslab[Sstack[i]] = tlabel;
  }

  for(auto node:nodes_to_double_check)
  {
    int new_rec = node;
    // std::cout << "2::" << new_rec << std::endl;

    for(int i=1;i<=correction_level;i++)
    {
      new_rec = Prec[new_rec];
      if(i<correction_level)
      {
        std::vector<int> gurg;
        for(size_t j =0 ; j<8;j++)
        {
          if(Mrec(new_rec,j)>=0)
            gurg.emplace_back(Mrec(new_rec,j));
        }
        this->graph[new_rec].receivers = gurg;
      }
      else
      {
        int target_basin = baslab[target_basin];
        std::vector<int> gurg;
        for(auto rec:this->graph[new_rec].receivers)
        {
          if(baslab[rec] == target_basin)
            gurg.emplace_back(rec);
        }
        this->graph[new_rec].receivers = gurg;
        this->graph[node].receivers = {new_rec};
      }
    }

  }


  //identify the pits 

  std::cout << "I tried something, it should work." << std::endl;
}

// This function correct the multiple flow receivers and donrs which end up with some duplicates probably linked to my homemade corrections.
// it simply removes duplicate and inverse the corrected receivers, into the donors.
void NodeGraphV2::initial_correction_of_MF_receivers_and_donors(xt::pytensor<int,2>& tMF_rec, xt::pytensor<int,2>& tMF_don, xt::pytensor<double,1>& elevation)
{
  for (size_t i =0; i< this->un_element; i++)
  {
    std::set<int> setofstuff;
    for(size_t j=0; j<8;j++)
    {
      if(tMF_rec(i,j) < 0)
        continue;

      const bool is_in = setofstuff.find(tMF_rec(i,j)) != setofstuff.end();
      if(is_in)
        tMF_rec(i,j) = -1;
      else if ( elevation[i]<elevation[tMF_rec(i,j)])
        tMF_rec(i,j) = -1;
      else
        setofstuff.insert(tMF_rec(i,j));
    }
  }  

  xt::pytensor<int,1> ndon = xt::zeros<int>({this->un_element});
  for (size_t i =0; i< this->un_element; i++)
  {
    for(size_t j=0; j<8;j++)
    {
      int this_rec = tMF_rec(i,j);
      if(this_rec<0)
        continue;
      tMF_don(this_rec,ndon[this_rec]) = int(i);
      ndon[this_rec]++;
    }
  }

  // And labelling as no data the remaining donors
  for (size_t i =0; i< this->un_element; i++)
  {
    for(int j = ndon[i]; j<8; j++)
    {
      tMF_don(i,j) = -1;
    }
  }
}

//Homemade_calculation of receivers and donors
void NodeGraphV2::compute_receveivers_and_donors(xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation)
{
  std::vector<int> nodes_to_compute;
  nodes_to_compute.reserve(this->un_element);
  for(int i=0;i<this->un_element;i++)
    nodes_to_compute.emplace_back(i);
  this->compute_receveivers_and_donors(active_nodes,  elevation, nodes_to_compute);
}

void NodeGraphV2::compute_receveivers_and_donors(xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation, std::vector<int>& nodes_to_compute)
{
  std::vector<bool> processed(this->un_element, false);
  std::vector<bool> is_processed_for_flats(this->un_element, false);
  std::vector<int> node_to_check_after_flat;
  std::vector<int> flat_ID(this->un_element,-1);

  int flat_indenter = -1;
  for(auto& i:nodes_to_compute)
  {
    if(active_nodes[i] == false)
    {
      this->graph[i].Sreceivers = i;
      continue;
    }
    this->graph[i].receivers.clear();
    this->graph[i].length2rec.clear();
    this->graph[i].donors.clear();
    this->graph[i].length2don.clear();
    this->graph[i].donors.reserve(8);
    this->graph[i].Sdonors.reserve(8);
    this->graph[i].receivers.reserve(8);
    this->graph[i].length2rec.reserve(8);
    this->graph[i].length2don.reserve(8);
    // std::vector<int> receivers,donors;
    // std::vector<double> length2rec,length2don;

    int checker = this->get_checker(int(i), active_nodes[i]);


    double this_elev = elevation[i];
    double test_elev;
    double SS = std::numeric_limits<double>::min();
    int SSid = -1;
    int idL = -1;
    bool potential_flat = false;
    for(auto adder:this->neightbourer[checker])
    {
      int node = i+adder;
      
      // if(i == 100)

      idL++;
      test_elev = elevation[node];
      if(test_elev < this_elev)
      {
        this->graph[i].receivers.emplace_back(node);
        this->graph[i].length2rec.emplace_back(this->lengthener[idL]);
        double slope = (this_elev - test_elev)/ this->lengthener[idL];
        if(slope>SS)
        {
          SS = slope;
          SSid = node;
        }
      }
      else if(test_elev>this_elev)
      {
        this->graph[i].donors.emplace_back(node);
        this->graph[i].length2don.emplace_back(this->lengthener[idL]);
      }
      else if (test_elev == this_elev)
      {
        potential_flat = true;
      }
    }

    // this->graph[i].receivers = receivers;
    // this->graph[i].donors = donors;
    // this->graph[i].length2rec = length2rec;
    // this->graph[i].length2don = length2don;

    if(SSid>=0)
    {  

      this->graph[i].Sreceivers = SSid;
      this->graph[SSid].Sdonors.emplace_back(i);
    
    }
    else
    {
      // DEPRECATED
      // // In this cases I am first checking if I need to resolve flat surfaces
      // if(false)
      // // if(potential_flat && this->graph[i].receivers.size() == 0 && is_processed_for_flats[i] == false)
      // {
      //   // flat solver here
      //   flat_indenter++;

      //   std::queue<int> HighEdge,LowEdge;
      //   std::vector<char> is_high_edge,is_low_edge;
      //   std::map<int,int>  this_flat_surface_node_index;
      //   bool aretherelow = true;
      //   int bottomcounter = 0;

      //   std::vector<int> this_flat_surface_node = this->Barnes2014_identify_flat(int(i), elevation, active_nodes, checker, HighEdge, LowEdge, 
      //     is_high_edge, is_low_edge, this_flat_surface_node_index);
      //   std::vector<int> this_flat_mask(this_flat_surface_node.size(),0);
        
      //   // std::cout << "IDENTIFIED::" << this_flat_mask.size() << "::" << LowEdge.size() << "::" << HighEdge.size() << std::endl;

      //   int max_lab = -9999;
      //   double elev_checker = elevation[int(i)];
      //   for(auto node:this_flat_surface_node)
      //   {
      //     int index = this_flat_surface_node_index[node];
      //   }
        
        
      //   std::vector<char> is_lower_edge_copy = is_low_edge;

      //   this->Barnes2014_AwayFromHigh( this_flat_mask, this_flat_surface_node, this_flat_surface_node_index,
      //     checker, HighEdge,  elevation, elev_checker, is_high_edge,  max_lab);
      
      //   if(LowEdge.size()>0)
      //   {
      //     this->Barnes2014_TowardsLower( this_flat_mask, this_flat_surface_node, this_flat_surface_node_index,
      //         checker, LowEdge, elevation, elev_checker, is_low_edge,  max_lab);
      //   }
      //   else
      //   {
      //     aretherelow = false;
      //     for(auto& val:this_flat_mask)
      //       val = std::abs(val - max_lab);
      //   }


      //   for(size_t j = 0 ; j < this_flat_surface_node.size() ; j++)
      //   {
      //     int node = this_flat_surface_node[j];
      //     checker = this->get_checker(node, active_nodes[node]);

      //     bool SS_done = false;
      //     int idL = -1;
      //     double S_max = -1;
      //     double S_length = -1;
      //     int S_id = -1;
      //     this->flat_mask[node] = this_flat_mask[this_flat_surface_node_index[node]];
      //     // if(this_flat_mask[this_flat_surface_node_index[node]]<0)
      //     //   throw std::runtime_error("Argh, nef flat mask yo");

      //     if(active_nodes[node] == false)
      //       continue;

      //     if(is_lower_edge_copy[this_flat_surface_node_index[node]] == 't' && active_nodes[node])
      //       continue;

      //     is_processed_for_flats[node] = true;

      //     bool all_flat = true;


      //     for(auto adder:this->neightbourer[checker])
      //     {
      //       int next = node + adder;
      //       idL ++;
      //       if(elevation[next] != elevation[int(i)])
      //         continue;

      //       if(this_flat_mask[this_flat_surface_node_index[node]] > this_flat_mask[this_flat_surface_node_index[next]] )
      //       {
      //         all_flat = false;
      //         this->graph[node].receivers.emplace_back(next);
      //         this->graph[node].length2rec.emplace_back(this->lengthener[idL]);
      //         double this_slope = this_flat_mask[this_flat_surface_node_index[node]] - this_flat_mask[this_flat_surface_node_index[next]];
      //         this_slope = this_slope / this->lengthener[idL];

      //         if(this_slope > S_max)
      //         {
      //           S_max = this_slope;
      //           S_id = next;
      //           S_length = this->lengthener[idL];
      //         }
      //         // if(SS_done == false)
      //         // {
      //         //   SS_done = true;
      //         //   this->graph[node].Sreceivers = next;
      //         //   this->graph[node].length2Srec = this->lengthener[idL];
      //         //   this->graph[next].Sdonors.emplace_back(node);
      //         // }

      //       }
      //       else if(this_flat_mask[this_flat_surface_node_index[node]] < this_flat_mask[this_flat_surface_node_index[next]] )
      //       {
      //         // all_flat = false;
      //         this->graph[node].donors.emplace_back(next);
      //         this->graph[node].length2don.emplace_back(this->lengthener[idL]);
      //       }
      //     }

      //     if(S_id>=0)
      //     {
      //       if(processed[node] == true || (flat_ID[node] >= 0))
      //       {
      //         std::cout << "5.12:" << std::endl;
      //         throw std::runtime_error("flat_resolver::SS_ID assigned multiple times");
      //       }

      //       processed[node] = true;
      //       this->graph[node].Sreceivers = S_id;
      //       this->graph[node].length2Srec = S_length;
      //       this->graph[S_id].Sdonors.emplace_back(node);
      //     }

      //     if(all_flat)
      //     {
      //       node_to_check_after_flat.emplace_back(node);
      //     }

      //     flat_ID[node] = flat_indenter;

      //   }

      // }
      // else 
      if (this->graph[i].receivers.size() == 0)
      {
        this->graph[i].Sreceivers = i;
      }
    }
  }


  int n_sorted = 1;

  while(node_to_check_after_flat.size() > 0 && n_sorted >0)
  {
    std::vector<int> next_to_check;
    n_sorted = 0;
    for(auto i: node_to_check_after_flat)
    {
      bool is_sorted = false;
      // Attempted to sort stuff here
      if(this->graph[i].receivers.size() == 0)
      {
        int checker = this->get_checker(i, active_nodes[i]);
        int idL =-1;
        for(auto adder:this->neightbourer[checker])
        {
          int next = i + adder;
          idL ++;
          if(elevation[next]!= elevation[i])
            continue;

          if(this->graph[next].receivers.size()>0 && is_sorted == false && std::find(this->graph[next].receivers.begin(), this->graph[next].receivers.end(), i) == this->graph[next].receivers.end())
          {
            is_sorted = true;
            this->graph[i].Sreceivers = next;
            this->graph[i].length2Srec = this->lengthener[idL];
            this->graph[next].Sdonors.emplace_back(i);
            this->graph[next].donors.emplace_back(i);
            this->graph[i].length2don.emplace_back(this->lengthener[idL]);
            this->graph[i].receivers.emplace_back(next);
            this->graph[i].length2rec.emplace_back(this->lengthener[idL]);
          }
        }

      }

      if(is_sorted)
        n_sorted ++;
      else
        next_to_check.emplace_back(i);
      
      //   this->graph[i].Sreceivers = i;


    }
    node_to_check_after_flat = next_to_check;
  }

  for(auto no:node_to_check_after_flat)
  {
    this->graph[no].Sreceivers = no;
  }
  // for( aut)


}

// DEPRECATED
// std::vector<int> NodeGraphV2::Barnes2014_identify_flat(int starting_node, xt::pytensor<double,1>& elevation,xt::pytensor<bool,1>& active_nodes, int checker,  
//   std::queue<int>& HighEdge, std::queue<int>& LowEdge, std::vector<char>& is_high_edge, std::vector<char>& is_low_edge, std::map<int,int>&  this_flat_surface_node_index)
// {
//   std::vector<int> output;
//   std::set<int> is_queued;

//   std::queue<int> Quack;Quack.push(starting_node);

//   is_queued.insert(starting_node);
//   output.emplace_back(starting_node);
//   char lefalse = 'f', letrue = 't';

//   is_low_edge.emplace_back(lefalse);
//   is_high_edge.emplace_back(lefalse);
//   this_flat_surface_node_index[starting_node] = 0;

//   double checkelev = elevation[starting_node];
//   bool is_LE = false, is_HE = false;
//   int index = 1;

//   while(Quack.size() >0)
//   {
//     int next_node = Quack.front();
//     Quack.pop();

//     checker = this->get_checker(next_node, active_nodes[next_node]);

//     is_LE = false, is_HE = false;

//     if(active_nodes[next_node])
//     {      
//       for(auto adder:this->neightbourer[checker])
//       {
//         int node = next_node + adder;
//         if(node<0 || node >= this->n_element)
//         {
//           throw std::runtime_error("neightbourer problem flat");
//         }
        

//         if(elevation[node]>checkelev)
//           is_HE = true;
        
//         else if(elevation[node]<checkelev)
//           is_LE = true;

//         if(elevation[node] != checkelev)
//           continue;

//         if(is_queued.find(node) != is_queued.end())
//           continue;

//         Quack.push(node);
//         is_queued.insert(node);
//         output.emplace_back(node);
//         is_low_edge.emplace_back(lefalse);
//         is_high_edge.emplace_back(lefalse);
//         this_flat_surface_node_index[node] = index;
//         index++;

//       }
//     }
//     else
//       is_LE = true;

//     if(is_LE)
//     {
//       LowEdge.push(next_node);
//       is_low_edge[this_flat_surface_node_index[next_node]] = letrue;
//     }
//     else if(is_HE)
//     {
//       HighEdge.push(next_node);
//       is_high_edge[this_flat_surface_node_index[next_node]] = letrue;
//     }

//   }
//   is_low_edge.shrink_to_fit();
//   is_high_edge.shrink_to_fit();


//   return output;
    

// }
// DEPRECATED

// DEPRECATED
// void NodeGraphV2::Barnes2014_AwayFromHigh(std::vector<int>& flat_mask, std::vector<int>& this_flat_surface_node, std::map<int,int>& this_flat_surface_node_index,
//  int checker, std::queue<int>& HighEdge, xt::pytensor<double,1>& elevation, double elev_check, std::vector<char>& is_high_edge, int& max_lab)
// {

//   int score = 1;
//   int marker = -9999;
//   HighEdge.push(marker);
//   bool keep_going = true, last_one_was_marker = false;

//   while(keep_going)
//   {

//     int next_node = HighEdge.front(); HighEdge.pop();
//     checker = this->get_checker(next_node,active_nodes[next_node]);
//     if(next_node == marker)
//     {
//       if(last_one_was_marker)
//       {
//         keep_going = false;
//         break;
//       }

//       score ++;
//       last_one_was_marker = true;
//       HighEdge.push(marker);
//     }
//     else
//     {
//       last_one_was_marker = false;
//       int index = this_flat_surface_node_index[next_node];
//       flat_mask[index] = score;
//       max_lab = score;
//       for(auto adder:this->neightbourer[checker])
//       {
//         int node = next_node + adder;

//         if(elevation[node] != elev_check)
//           continue;

//         index = this_flat_surface_node_index[node];

//         if(is_high_edge[index] == 't')
//           continue;

//         HighEdge.push(node);
//         is_high_edge[index] = 't';
//       }
//     }
//   }


// }
// DEPRECATED

// DEPRECATED
// void NodeGraphV2::Barnes2014_TowardsLower(std::vector<int>& flat_mask, std::vector<int>& this_flat_surface_node, std::map<int,int>& this_flat_surface_node_index,
//  int checker, std::queue<int>& LowEdge, xt::pytensor<double,1>& elevation, double elev_check, std::vector<char>& is_low_edge, int max_lab)
// {
//   int score = 1;
//   int marker = -9999;
//   LowEdge.push(marker);
//   bool keep_going = true, last_one_was_marker = false;

//   for (auto& val:flat_mask)
//     val = -val;

//   while(keep_going)
//   {
//     int next_node = LowEdge.front(); LowEdge.pop();
//     checker = this->get_checker(next_node);
//     if(next_node == marker)
//     {
//       if(last_one_was_marker)
//       {
//         keep_going = false;
//         break;
//       }

//       score ++;
//       last_one_was_marker = true;
//       LowEdge.push(marker);
//     }
//     else
//     {
//       last_one_was_marker = false;
//       int index = this_flat_surface_node_index[next_node];
//       if(flat_mask[index]>0)
//         continue;
//       if(flat_mask[index] < 0)
//         flat_mask[index] = max_lab + flat_mask[index] + 2 * score;
//       else
//         flat_mask[index] = 2 * score;

//       for(auto& adder:this->neightbourer[checker])
//       {
//         int node = next_node + adder;

//         if(elevation[node] != elev_check)
//           continue;

//         index = this_flat_surface_node_index[node];

//         if(is_low_edge[index] == 't')
//           continue;
//         LowEdge.push(node);
//         is_low_edge[index] = 't';
//       }

//     }
//   }


// }
// DEPRECATED


void NodeGraphV2::recompute_multi_receveivers_and_donors(xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation, std::vector<int>& nodes_to_compute)
{

  for(auto& i:nodes_to_compute)
  {
    if(active_nodes[i] == false)
    {
      this->graph[i].Sreceivers = i;
      continue;
    }
    std::vector<int> receivers,donors;
    std::vector<double> length2rec,length2don;
    receivers.reserve(8);
    donors.reserve(8);
    length2rec.reserve(8);
    length2don.reserve(8);
    int checker;
    if(i<ncols)
      checker = 1;
    else if (i >= this->n_element - ncols)
      checker = 2;
    else if(i % ncols == 0 || i == 0)
      checker = 3;
    else if((i + 1) % (ncols) == 0 )
      checker = 4;
    else
      checker = 0;


    double& this_elev = elevation[i];
    double test_elev;
    int idL = -1;
    for(auto& adder:this->neightbourer[checker])
    {
      int node = i+adder;
      idL++;
      test_elev = elevation[node];
      if(test_elev<this_elev)
      {
        receivers.emplace_back(node);
        length2rec.emplace_back(this->lengthener[idL]);
      }
      else if(test_elev>this_elev)
      {
        donors.emplace_back(node);
        length2don.emplace_back(this->lengthener[idL]);
      }
    }
    this->graph[i].receivers = std::move(receivers);
    this->graph[i].donors = std::move(donors);
    this->graph[i].length2rec = std::move(length2rec);
    this->graph[i].length2don = std::move(length2don);

  }


}

int NodeGraphV2::get_checker(int i, bool is_active)
{
  // internal node 0
  // periodic_first_row 1
  // periodic_last_row 2
  // periodic_first_col 3
  // periodic last_col 4
  // normal_first_row 5
  // normal_last_row 6
  // normal_first_col 7
  // normal_last_col 8
  // normal_top_left 9
  // normal_top_right 10
  // normal_bottom_left 11
  // normal_bottom_right 12

  int checker;
  if(i < this->ncols)
  {
    if(is_active)
      checker = 1;
    else if(i == 0)
      checker = 9;
    else if( i == ncols-1)
      checker = 10;
    else
      checker = 5;
  }
  else if (i >= this->n_element - this->ncols)
  {
    if(is_active)
      checker = 2;
    else if(i == this->n_element - this->ncols)
      checker = 11;
    else if(i == this->n_element - 1)
      checker = 12;
    else
      checker = 6;
  }
  else if(i % this->ncols == 0 || i == 0)
  {
    if(is_active)
      checker = 3;
    else
      checker = 7;
  }
  else if((i + 1) % (this->ncols) == 0 )
  {
    if(is_active)
      checker = 4;
    else
      checker = 8;
  }
  else
    checker = 0;
  
  return checker;
}

// void NodeGraphV2::get_D8_neighbors(int i, xt::pytensor<bool,1>& active_nodes, std::vector<int>& neightbouring_nodes, std::vector<double>& length2neigh)
// {
//   // these vectors are additioned to the node indice to test the neighbors
//   neightbouring_nodes = std::vector<int>();
//   length2neigh = std::vector<double>();


//   // is my node active or not
//   // An active node at a boundary means periodic
//   bool isa = active_nodes[i];

//   // To find the neighbours, code utilises premade looper with the value to add to the indice to get the valid neighbours
//   // Neighbourer indices
//   // internal node 0
//   // periodic_first_row 1
//   // periodic_last_row 2
//   // periodic_first_col 3
//   // periodic last_col 4
//   // normal_first_row 5
//   // normal_last_row 6
//   // normal_first_col 7
//   // normal_last_col 8
//   // normal_top_left 9
//   // normal_top_right 10
//   // normal_bottom_left 11
//   // normal_bottom_right 12

//   // NOTE THAT ELEMENT ARE VECTORISED

//   int checker = this->get_checker(i,isa);

//   // // first row
//   // if(i<ncols)
//   // {
//   //   if(isa)
//   //     checker = 1;
//   //   else
//   //   {
//   //     if(i > 0 && i < ncols - 1 )
//   //       checker = 5;
//   //     else if (i == 0)
//   //       checker = 9;
//   //     else
//   //       checker = 10;
//   //   }
//   // }
//   // // last row
//   // else if (i >= this->n_element - ncols)
//   // {
//   //   if(isa)
//   //     checker = 2;
//   //   else if(i > this->n_element - ncols && i < this->n_element - 1)
//   //     checker = 6;
//   //   else if (i == this->n_element - ncols)
//   //     checker = 11;
//   //   else
//   //     checker = 12;
//   // }
//   // // first col corners excluded
//   // else if(i % ncols == 0)
//   // {
//   //   if(isa)
//   //     checker = 3;
//   //   else
//   //     checker = 7;
//   // }
//   // else if((i + 1) % (ncols) == 0 )
//   // {
//   //   if(isa)
//   //     checker = 4;
//   //   else
//   //     checker = 8;
//   // }
//   // else
//   // {
//   //   checker = 0;
//   // }

//   int idL = -1;
//   for(auto& adder:this->neightbourer[checker])
//   {
//     int node = i+adder;
//     idL++;
//     neightbouring_nodes.emplace_back(node);
//     length2neigh.emplace_back(this->lengthener[idL]);
//   }
// }

std::vector<int> NodeGraphV2::get_all_flat_from_node(int i, xt::pytensor<double,1>& topography,  xt::pytensor<bool,1>& active_nodes)
{
  std::vector<char> is_in_queue(this->un_element,'n');
  is_in_queue[i] = 'y';
  std::queue<int> baal;baal.emplace(i);
  std::vector<int> output;
  output.reserve(std::round(topography.size()/10));
  while(baal.empty() == false)
  {
    std::vector<int> neightbouring_nodes; std::vector<double> length2neigh;
    int next_node = baal.front();
    baal.pop();
    this->get_D8_neighbors(next_node,  active_nodes,  neightbouring_nodes,  length2neigh);
    bool is_there_lower = false;
    for(auto tnode:neightbouring_nodes)
    {
      if(is_in_queue[tnode] == 'y')
        continue;
      if(topography[i] == topography[tnode])
      {
        is_in_queue[tnode] = 'y';
        baal.emplace(tnode);
      }
      else if (topography[i] > topography[tnode])
        is_there_lower = true;
    }

    if(is_there_lower == false)
      output.emplace_back(next_node);

  }
  return output;

}

void NodeGraphV2::get_D8_neighbors(int i, xt::pytensor<bool,1>& active_nodes, std::vector<int>& neightbouring_nodes, std::vector<double>& length2neigh)
{
  // these vectors are additioned to the node indice to test the neighbors
  neightbouring_nodes = std::vector<int>();
  length2neigh = std::vector<double>();
  neightbouring_nodes.reserve(8);
  length2neigh.reserve(8);

  // if(active_nodes[i] == 0)
  //   return;

  int checker = this->get_checker(i,active_nodes[i]);
 
  int idL = -1;
  for(auto& adder:this->neightbourer[checker])
  {
    int node = i+adder;
    idL++;
    neightbouring_nodes.emplace_back(node);
    length2neigh.emplace_back(this->lengthener[idL]);
  }
}

void NodeGraphV2::get_D4_neighbors(int i, xt::pytensor<bool,1>& active_nodes, std::vector<int>& neightbouring_nodes, std::vector<double>& length2neigh)
{
  // these vectors are additioned to the node indice to test the neighbors
  neightbouring_nodes = std::vector<int>();
  length2neigh = std::vector<double>();
  neightbouring_nodes.reserve(4);
  length2neigh.reserve(4);

  if(active_nodes[i] == false)
    return;

  int checker;
  if(i<ncols)
    checker = 1;
  else if (i >= this->n_element - ncols)
    checker = 2;
  else if(i % ncols == 0 || i == 0)
    checker = 3;
  else if((i + 1) % (ncols) == 0 )
    checker = 4;
  else
    checker = 0;

  std::vector<size_t> looper_D4  = {1,3,4,6};
  for(auto it:looper_D4)
  {
    int adder = this->neightbourer[checker][it]; 
    int node = i+adder;
    neightbouring_nodes.emplace_back(node);
    length2neigh.emplace_back(this->lengthener[it]);
  }
}

void NodeGraphV2::update_receivers_at_node(int node, std::vector<int>& new_receivers)
{
  if(is_depression(node) == false)
    this->graph[node].receivers = new_receivers;
  else
    new_receivers = {};
}
void NodeGraphV2::update_donors_at_node(int node, std::vector<int>& new_donors)
{
  this->graph[node].donors = new_donors;
}


std::vector<int> multiple_stack_fastscape(int n_element, std::vector<Vertex>& graph, std::vector<int>& not_in_stack, bool& has_failed)
{

  std::vector<int>ndon(n_element,0);
  for(size_t i=0; i<n_element;i++)
  {
    for(auto trec:graph[i].receivers)
    {
      ndon[trec]  = ndon[trec] + 1;
    }
  }

  std::vector<int> vis(n_element,0), parse(n_element,-1), stack(n_element,-9999);
  
  int nparse = -1;
  int nstack = -1;


  // we go through the nodes
  for(size_t i=0; i<n_element;i++)
  {
    // when we find a "summit" (ie a node that has no donors)
    // we parse it (put it in a stack called parse)
    if (ndon[i] == 0)
    {
      nparse =  nparse+1;
      parse[nparse] = i;
    }
    // we go through the parsing stack
    while (nparse > -1)
    {
      int ijn = parse[nparse];
      nparse = nparse-1;
      nstack = nstack+1;

      stack[nstack] = ijn;
      for(int ijk=0; ijk < graph[ijn].receivers.size();ijk++)
      {
        int ijr = graph[ijn].receivers[ijk];
        vis[ijr] = vis[ijr]+1;
        // if the counter is equal to the number of donors for that node we add it to the parsing stack
        if (vis[ijr] == ndon[ijr])
        {
          nparse=nparse+1;
          parse[nparse]=ijr;
        }
        
      }
    } 
  }
  if(nstack < n_element - 1 )
  {
    has_failed = true;
    std::cout << "WARNING::STACK UNDERPOPULATED::" << nstack << "/" << stack.size() << std::endl;
    std::cout << "Investigating ..." << std::endl;
    std::vector<bool> is_in_stack(stack.size(),false);
    for(auto& node: stack)
    {
      if(node >=0)
        is_in_stack[node] = true;
      else
        node = 0;
    }
    std::cout << "identifying the ghost nodes ..." << std::endl;
    for(int i=0; i< int(is_in_stack.size()); i++)
    {
      if(is_in_stack[i] == false)
        not_in_stack.emplace_back(i);
    }
    std::cout << "Got them! you can access the ghost nodes with model.get_broken_nodes()" << std::endl;
    if(EXTENSIVE_STACK_INFO == false)
    {
      std::cout << "Important: If this happens at the start of the model with a random surface for few timesteps this is not critical." << std::endl;
      std::cout << "If it happens in the middle of a run with a mature mountain this is a problem." << std::endl;
      std::cout << "You can activate the SAFE_STACK_STOPPER switch to automatically stop if this happens," << std::endl;
      std::cout << "You can activate the EXTENSIVE_STACK_INFO switch to get extensive debugging report, be aware that it might involve (time-)expensive calculations" << std::endl;
    }
    else
    {
      // get index_in_stack
      std::vector<int> index_in_stack(stack.size());
      int incrid = 0;
      for(auto node:stack)
      {
        if(node >= 0 )
        index_in_stack[node] = incrid;
        incrid++;
      }

      // Find isolated nodes
      std::vector<int> isolated_nodes;
      std::vector<int> min_donor;
      std::vector<int> score;
      for(auto node:not_in_stack)
      {
        int min_ID = std::numeric_limits<int>::max();
        int min_score = std::numeric_limits<int>::max();
        bool found = false;
        for(auto tnode:graph[node].donors)
        {

          if(is_in_stack[tnode])
          {
            if(index_in_stack[tnode]<min_score)
            {
              min_score = index_in_stack[tnode];
              min_ID = tnode;
              found = true;
            }
          }
        }
        if(found)
        {
          isolated_nodes.emplace_back(node);
          min_donor.emplace_back(min_ID);
          score.emplace_back(min_score);
        }

      }

      std::cout << " isolated Nodes connected to donors in stack: ID|DONORS|ID_IN_STACK" << std::endl;
      for(size_t i=0;i<isolated_nodes.size();i++)
      {
        std::cout << isolated_nodes[i] << "|" << min_donor[i] << "|" << score[i] << std::endl;
      }

      std::cout << "Now checking for cyclicity ..." << std::endl;

      for(int i =0; i< int(stack.size()); i++)
      {
        std::vector<bool> inqueue(stack.size(),false);
        std::queue<int> Q;
        Q.push(i);
        inqueue[i] = true;
        while(Q.empty() == false)
        {
          int nnode = Q.front();
          Q.pop();
          for(auto node:graph[nnode].receivers)
          {
            if(node == i)
              std::cout << "CYCLICITY DECTEDTED:" << i << ":->...->" << nnode << ":-->" << i << ", is node in stack? " << std::to_string(is_in_stack[i]) << std::endl;

            if(inqueue[node])
              continue;
            inqueue[node] = true;
            Q.push(node);
          }
        }
      }


    }




    if(SAFE_STACK_STOPPER)
      throw std::runtime_error("stack underpopulated, SAFE_STACK_STOPPER triggered (CHONK.set_DEBUG_switch_nodegraph(['SAFE_STACK_STOPPER'],false) to deactivate).");
  }

  return stack;
}


// Quack
//   __
// <(o )___
//  ( ._> /
//   `---'  
// Quack
//    quack quack  __
//             ___( o)>
//             \ <_. )
//    ~~~~~~~~  `---' 





// the following functions are a portage of xarray-topo to get the single stack and the depression solver working
void NodeGraphV2::compute_stack()
{    
  int istack = 0;

  for (int inode = 0; inode< this->n_element;inode++)
  {
    if (this->graph[inode].Sreceivers == inode)
    {
      this->Sstack[istack] = inode;
      istack ++;
      istack = this->_add2stack(inode, istack);
    }
  }
  if(istack > this->n_element)
  {
    std::cout << "Sstack inconsistent? " << istack << std::endl;
    throw std::runtime_error("Sstack inconsistent");
  }
}

int NodeGraphV2::_add2stack(int& inode, int& istack)
{
  for(int k = 0; k < int(this->graph[inode].Sdonors.size());k++)
  {
    int idonor = this->graph[inode].Sdonors[k];
    Sstack[istack] = idonor;
    istack ++;
    istack = this->_add2stack(idonor, istack);
  }

  return istack;
}


void NodeGraphV2::compute_basins(xt::pytensor<bool,1>& active_nodes)
{
  int ibasin = -1, istack,irec;
  SBasinID = xt::zeros<int>({this->un_element});
  for(int inode=0; inode<this->n_element;inode++)
  {  
    // std::cout << inode << "||";

    istack = this->Sstack[inode];
    irec = this->graph[istack].Sreceivers;

    if (irec == istack)
    {
      ibasin ++;
      SBasinOutlets.emplace_back(istack);
    }
    SBasinID[istack] = ibasin;
  }

  this->nbasins = ibasin + 1;
}



void NodeGraphV2::compute_pits(xt::pytensor<bool,1>& active_nodes)
{
  int ipit = 0;

  for (int ibasin = 0; ibasin<nbasins; ibasin++)
  {
    int inode = this->SBasinOutlets[ibasin];

    if (active_nodes[inode])
    {
      pits.emplace_back(inode);
      ipit += 1;
    }
  }
  this->npits = ipit;

}

void NodeGraphV2::collapse_depression_tree(xt::pytensor<int,2>& conn_basins, xt::pytensor<int,2>& conn_nodes, 
                                           xt::pytensor<double,1>& conn_weights, xt::pytensor<double,1>& elevation, int& basin0)
{
  // initialising all the connections
  std::vector<std::pair<int,int> > preoutput;
  std::vector<std::pair<int,int> > preoutput_nodes;
  // preoutput.reserve(20); // arbitrary memory reserve

  for (auto& dep: this->depression_tree)
  {
    preoutput.emplace_back( dep.connections_bas );
    if(dep.connections_bas.second == -1 && basin0 == -1)
      basin0 = dep.index;
    int this_dep = dep.index;
    int recdep = dep.connections_bas.second;
    int this_node_rec = dep.connections.second;
    while(this->depression_tree[this_dep].has_children)
    {
      this_dep = this->depression_tree[this_dep].children.first;
      preoutput.emplace_back( std::make_pair(this_dep, recdep) );
      preoutput_nodes.emplace_back(std::make_pair(this->depression_tree[this_dep].connections.first, this_node_rec) );
    }
  }

  int i=0;
  for (int i = 0; i< preoutput.size(); i++)
  {
    conn_basins(i,0) = preoutput[i].first;
    conn_basins(i,1) = preoutput[i].second;
    conn_nodes(i,0) = preoutput_nodes[i].first;
    conn_nodes(i,1) = preoutput_nodes[i].second;
    conn_weights[i] = std::max(elevation[conn_nodes(i,0)],elevation[conn_nodes(i,1)]);
    i++;
  }

}

void NodeGraphV2::correct_flowrouting(xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation)
{
    // """Ensure that no flow is captured in sinks.

    // If needed, update `receivers`, `dist2receivers`, `ndonors`,
    // `donors` and `stack`.

    // """

    if(this->depression_tree.size() == 0)
      return;

    // I first need to collapse the depression tree to calculate the planar connections between basins
    xt::pytensor<int,2> conn_basins,conn_nodes;
    xt::pytensor<double,1> conn_weights; //conn_weights = np.empty(nconn_max, dtype=np.float64)
    int basin0;
    this->collapse_depression_tree(conn_basins, conn_nodes, conn_weights, elevation,basin0); // (Although the tree collapses, the carbon balance os good because it has just grown)

    int nconn = int(conn_weights.size());

    this->nbasins = int(this->depression_tree.size());

    // We do not need this anymore
    // this->_connect_basins(conn_basins, conn_nodes, conn_weights, active_nodes, elevation, nconn, basin0);

    // g = 6? I am not sure what is happening here
    int g = 6;

    mstree = _compute_mst_kruskal(conn_basins, conn_weights);
    this->_orient_basin_tree(conn_basins, conn_nodes, basin0, mstree);
    this->_update_pits_receivers(conn_basins , conn_nodes, mstree, elevation);    
}

// """Connect adjacent basins together through their lowest pass.

// Creates an (undirected) graph of basins and their connections.

// The resulting graph is defined by:

// - `conn_basins` (nconn, 2): ids of adjacent basins forming the edges
// - `conn_nodes` (nconn, 2): ids of grid nodes forming the lowest passes
//   between two adjacent basins.
// - `conn_weights` (nconn) weights assigned to the edges. It is equal to the
//   elevations of the passes, i.e., the highest elevation found for each
//   node couples defining the passes.

// The function returns:

// - `nconn` : number of edges.
// - `basin0` : id of one open basin (i.e., where `outlets[id]` is not a
//   pit node) given as reference.

// The algorithm parses each grid node of the flow-ordered stack and checks if
// the node and (each of) its neighbors together form the lowest pass between
// two different basins.

// Node neighbor lookup doesn't include diagonals to ensure that the
// resulting graph of connected basins is always planar.

// Connections between open basins are handled differently:

// Instead of finding connections between adjacent basins, virtual
// connections are added between one given basin and all other
// basins.  This may save a lot of uneccessary computation, while it
// ensures a connected graph (i.e., every node has at least an edge),
// as required for applying minimum spanning tree algorithms implemented in
// this package.

// """
void NodeGraphV2::_connect_basins(xt::pytensor<int,2>& conn_basins, xt::pytensor<int,2>& conn_nodes, xt::pytensor<double,1>& conn_weights,          
                   xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation, int& nconn, int& basin0)
{
  int iconn = 0;

  basin0 = -1; //intp?
  int ibasin = 0;

  xt::pytensor<int,1> conn_pos = xt::zeros<int>({this->nbasins}); //np.full(nbasins, -1, dtype=np.intp)
  for(auto& v:conn_pos)
    v = -1;


  xt::pytensor<int,1> conn_pos_used = xt::empty<int>({this->nbasins});  // conn_pos_used = np.empty(nbasins, dtype=np.intp)

  int conn_pos_used_size = 0;

  bool iactive = false;

  for(auto& istack : this->Sstack)
  {
    int irec = this->graph[istack].Sreceivers;

    // # new basin
    if(irec == istack)
    {
      ibasin = this->SBasinID[istack];
      iactive = active_nodes[istack];

      // for iused in conn_pos_used[:conn_pos_used_size]
      for(int iused = 0; iused < conn_pos_used_size;iused++)
      {
        conn_pos[conn_pos_used[iused]] = -1;
      }

      conn_pos_used_size = 0;

      if (iactive == false)
      {
        if(basin0 == -1)
            basin0 = ibasin;
        else
        {
          // conn_basins[iconn] = (basin0, ibasin);
          conn_basins(iconn,0) = basin0;
          conn_basins(iconn,1) = ibasin;

          // conn_nodes[iconn] = (-1, -1)
          conn_nodes(iconn,0) = -1;
          conn_nodes(iconn,1) = -1;

          conn_weights[iconn] = - std::numeric_limits<double>::max();
          iconn ++;
        }
      }
    }

    if(iactive)
    {
      std::vector<int> D4n;
      std::vector<double> D4l;
      this->get_D8_neighbors(istack,active_nodes, D4n, D4l);
      for(auto ineighbor:D4n)
      {

        int ineighbor_basin = SBasinID[ineighbor];
        int ineighbor_outlet = SBasinOutlets[ineighbor_basin];

        // # skip same basin or already connected adjacent basin
        // # don't skip adjacent basin if it's an open basin
        if(ibasin >= ineighbor_basin && active_nodes[ineighbor_outlet])
          continue;

        double weight = std::max(elevation[istack], elevation[ineighbor]);
        int conn_idx = conn_pos[ineighbor_basin];

        // # add new connection
        if(conn_idx == -1)
        {
          // conn_basins[iconn] = (ibasin, ineighbor_basin);
          conn_basins(iconn,0) = ibasin;
          conn_basins(iconn,1) = ineighbor_basin;
          // conn_nodes[iconn] = (istack, ineighbor);
          conn_nodes(iconn,0) = istack;
          conn_nodes(iconn,1) = ineighbor;

          conn_weights[iconn] = weight;

          conn_pos[ineighbor_basin] = iconn;
          iconn ++;

          conn_pos_used[conn_pos_used_size] = ineighbor_basin;
          conn_pos_used_size ++;
        }

        // # update existing connection
        else if (weight < conn_weights[conn_idx])
        {
          // conn_nodes[conn_idx] = (istack, ineighbor);
          conn_nodes(conn_idx,0) = istack;
          conn_nodes(conn_idx,1) = ineighbor;
          conn_weights[conn_idx] = weight;
        }
      }
    }
  }
  nconn = iconn;

  return;
}


// """Compute the minimum spanning tree of the (undirected) basin graph.

// The method used here is Kruskal's algorithm. Applied to a fully
// connected graph, the complexity of the algorithm is O(m log m)
// where `m` is the number of edges.

// """
xt::xtensor<int,1> NodeGraphV2::_compute_mst_kruskal(xt::pytensor<int,2>& conn_basins, xt::pytensor<double,1>& conn_weights)
{
  xt::xtensor<int,1> mstree = xt::empty<int>({int(this->depression_tree.size()) - 1});
  int mstree_size = 0;

  // # sort edges
  auto sort_id = xt::argsort(conn_weights);


  UnionFind uf(int(this->depression_tree.size()));

  for (auto eid : sort_id)
  {
    // if(eid > 1000000 || eid <0)
      // std::cout << "FLURB::" << eid << std::endl; 
    int b0 = conn_basins(eid, 0);
    int b1 = conn_basins(eid, 1);

    if (uf.Find(b0) != uf.Find(b1))
    {
      mstree[mstree_size] = eid;
      // mstree_translated.emplace_back(std::initializer_list<int>{this->depression_tree[b0].pit, this->depression_tree[b1].pit});
      mstree_size ++;
      uf.Union(b0, b1);
    }
  }
  return mstree;
}

// """Orient the graph (tree) of basins so that the edges are directed in
// the inverse of the flow direction.

// If needed, swap values given for each edges (row) in `conn_basins`
// and `conn_nodes`.

// """
void NodeGraphV2::_orient_basin_tree(xt::pytensor<int,2>& conn_basins, xt::pytensor<int,2>& conn_nodes, int& basin0, xt::xtensor<int,1>& tree)
{
  // # nodes connections
  xt::xtensor<int,1> nodes_connects_size = xt::zeros<int>({this->nbasins});
  xt::xtensor<int,1> nodes_connects_ptr = xt::empty<int>({this->nbasins});

  // # parse the edges to compute the number of edges per node
  for (auto i : tree)
  {
    nodes_connects_size[conn_basins(i, 0)]++;
    nodes_connects_size[conn_basins(i, 1)]++;
  }

  // # compute the id of first edge in adjacency table
  nodes_connects_ptr[0] = 0;
  // for i in range(1, nbasins):
  for (int i = 1; i < nbasins; i++)
  {
    nodes_connects_ptr[i] = (nodes_connects_ptr[i - 1] + nodes_connects_size[i - 1]);
    nodes_connects_size[i - 1] = 0;
  }

  // # create the adjacency table
  int nodes_adjacency_size = nodes_connects_ptr[nbasins - 1] + nodes_connects_size[nbasins - 1];
  nodes_connects_size[this->nbasins -1] = 0;
  xt::xtensor<int,1> nodes_adjacency = xt::zeros<int>({nodes_adjacency_size});

  // # parse the edges to update the adjacency
  for (auto i : tree)
  {

    int n1 = conn_basins(i, 0);
    int n2 = conn_basins(i, 1);
    nodes_adjacency[nodes_connects_ptr[n1] + nodes_connects_size[n1]] = i;
    nodes_adjacency[nodes_connects_ptr[n2] + nodes_connects_size[n2]] = i;
    nodes_connects_size[n1] ++;
    nodes_connects_size[n2] ++;
  }


  // # depth-first parse of the tree, starting from basin0
  // # stack of node, parent
  xt::xtensor<int,2> stack = xt::empty<int>({this->nbasins, 2});
  int stack_size = 1;
  stack(0,0) = basin0;// (basin0, basin0)
  stack(0,1) = basin0;

  
  // CHEKCED UNTIL HERE
  int n_turn=0;
  while (stack_size > 0)
  {
    n_turn++;
    // # get parsed node
    stack_size = stack_size - 1;
    int node = stack(stack_size, 0);
    int parent = stack(stack_size, 1);

    // # for each edge of the graph
    // for i in range(nodes_connects_ptr[node], nodes_connects_ptr[node] + nodes_connects_size[node])
    for( int i = nodes_connects_ptr[node]; i < (nodes_connects_ptr[node] + nodes_connects_size[node]); i++) 
    {
      int edge_id = nodes_adjacency[i];

      // # the edge comming from the parent node has already been updated.
      // # in this case, the edge is (parent, node)
      if (conn_basins(edge_id, 0) == parent && node != parent)
      {
          // std::cout << SBasinOutlets[conn_basins(edge_id, 0)] << "||" << SBasinOutlets[conn_basins(edge_id, 1)] << std::endl;
          continue;
      }
      // std::cout << "GUR::" << SBasinOutlets[conn_basins(edge_id, 0)] << "||" << SBasinOutlets[conn_basins(edge_id, 1)] << std::endl;

      // # we want the edge to be (node, next)
      // # we check if the first node of the edge is not "node"
      if(node != conn_basins(edge_id, 0))
      {
        // # swap n1 and n2
        int cb1 = conn_basins(edge_id, 1);
        int cb0 = conn_basins(edge_id, 0);
        conn_basins(edge_id, 0) = cb1;
        conn_basins(edge_id, 1) = cb0;

        cb1 = conn_nodes(edge_id, 1);
        cb0 = conn_nodes(edge_id, 0);
        // # swap p1 and p2
        conn_nodes(edge_id, 0) = cb1;
        conn_nodes(edge_id, 1) = cb0;
      }
      // # add the opposite node to the stack
      stack(stack_size,0) = conn_basins(edge_id, 1);
      stack(stack_size,1) = node;
      stack_size ++;
    }
  }

}


// """Update receivers of pit nodes (and possibly lowest pass nodes)
// based on basin connectivity.

// Distances to receivers are also updated. An infinite distance is
// arbitrarily assigned to pit nodes.

// A minimum spanning tree of the basin graph is used here. Edges of
// the graph are also assumed to be oriented in the inverse of flow direction.

// """
void NodeGraphV2::_update_pits_receivers(xt::pytensor<int,2>& conn_basins,xt::pytensor<int,2>& conn_nodes, xt::xtensor<int,1>& mstree, xt::pytensor<double,1>& elevation)
{
  // for i in mstree:
  for(auto i : mstree)
  {

    int node_to = conn_nodes(i, 0);
    int node_from = conn_nodes(i, 1);

    // # skip open basins
    if (node_from == -1)
    {
        continue;
    }
      // std::cout << SBasinOutlets[conn_basins(i,0)] << "-" << SBasinOutlets[conn_basins(i,1)] << "--" << << "||";



    int outlet_from = this->SBasinOutlets[conn_basins(i, 1)];


    // std::cout << outlet_from << "->" << node_to << " || ";

    // if (elevation[node_from] < elevation[node_to])
    // {
     this->graph[outlet_from].Sreceivers = node_to;
     this->graph[outlet_from].length2Srec = this->dx*1000; // just to have a length but it should not actually be used
    // }
    // else
    // {
    //   this->graph[outlet_from].Sreceivers = node_to;
    //   // this->graph[node_from].Sreceivers = node_to;
    //   this->graph[outlet_from].length2Srec = this->dx*1000; // just to have a length but it should not actually be used
    //   // this->graph[node_from].length2Srec = this->dx; // TODO correct here the correct distance. Should not be used anyway. 
    // }
  }
}




//                         . - ~ ~ ~ - .
//       ..     _      .-~               ~-.
//      //|     \ `..~                      `.
//     || |      }  }              /       \  \
// (\   \\ \~^..'                 |         }  \
//  \`.-~  o      /       }       |        /    \
//  (__          |       /        |       /      `.
//   `- - ~ ~ -._|      /_ - ~ ~ ^|      /- _      `.
//               |     /          |     /     ~-.     ~- _
//               |_____|          |_____|         ~ - . _ _~_-_





std::vector<double> NodeGraphV2::get_distance_to_receivers_custom(int node, std::vector<int> list_of_receivers)
{
  std::vector<double> tdist2rec = std::vector<double>(list_of_receivers.size(), -1);

  for(size_t i=0; i<list_of_receivers.size(); i ++)
  {
    int tnode = list_of_receivers[i];
    for(size_t j=0; j < this->graph[node].receivers.size(); j++)
    {
      if(tnode == this->graph[node].receivers[j])
      {
        tdist2rec[i] = this->graph[node].length2rec[j];
        break;
      }
    } 
  }

  return tdist2rec;
}


// if(this_bool_map["isolate_pixels_draining_to_fixed_channel"])
//     {
//       // first read the
//       // Get the latitude and longitude
//       cout << "I am reading points from the file: "+ this_string_map["fixed_channel_csv_name"] << endl;
//       LSDSpatialCSVReader source_points_data( RI, (DATA_DIR+this_string_map["fixed_channel_csv_name"]) );
//       vector<float> X_coords; 
//       vector<float> Y_coords; 
//       vector<int> nodes_from_channel;
//       if(this_bool_map["use_xy_for_node_index"])
//       {
//         cout << "I am going to get node indices from X-Y coordinates." << endl;
        
//         source_points_data.get_nodeindices_from_x_and_y_coords(FlowInfo, X_coords, Y_coords, nodes_from_channel);        
//       } else{
//         cout << "I am going to get node indices from lat-long coordinates." << endl;
//         vector<int> nodes_from_channel = source_points_data.get_nodeindices_from_lat_long(FlowInfo); 
//       }
      
//       //vector<int> new_nodes_from_channel = source_points_data.get_nodeindex_vector();
//       //cout << "Old nodes: " <<  new_nodes_from_channel.size() << " and new: " << nodes_from_channel.size() << endl;
//       //for (int i = 0; i< int(nodes_from_channel.size()); i++)
//       //{
//       //  if (nodes_from_channel[i] < 0)
//       //  {
//       //    cout << "Invalid node index: " << nodes_from_channel[i] << endl;
//       //  }
//       //}
//       // Now run the flowinfo routine
//       LSDRaster NodesRemovedRaster = FlowInfo.find_nodes_not_influenced_by_edge_draining_to_nodelist(nodes_from_channel,filled_topography);
//       string remove_raster_name = OUT_DIR+OUT_ID+"_IsolateFixedChannel";
//       NodesRemovedRaster.write_raster(remove_raster_name,raster_ext);
//     }



// //  //###############################################  
// //  // 
// //  //           ,-.
// //  //       ,--' ~.).
// //  //     ,'         `.
// //  //    ; (((__   __)))
// //  //    ;  ( (#) ( (#)
// //  //    |   \_/___\_/|
// //  //   ,"  ,-'    `__".
// //  //  (   ( ._   ____`.)--._        _
// //  //   `._ `-.`-' \(`-'  _  `-. _,-' `-/`.
// //  //    ,')   `.`._))  ,' `.   `.  ,','  ;
// //  //  .'  .     `--'  /     ).   `.      ;
// //  // ;     `-        /     '  )         ;
// //  // \                       ')       ,'
// //  //  \                     ,'       ;
// //  //   \               `~~~'       ,'
// //  //    `.                      _,'
// //  //      `.                ,--'
// //  //        `-._________,--'


#endif