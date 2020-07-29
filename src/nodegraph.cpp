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
  xt::pytensor<int,1>& Sstack,
  xt::pytensor<int,1>& Srec, 
  xt::pytensor<int,1>& Prec, 
  xt::pytensor<double,1>& SLength, 
  xt::pytensor<int,2>& Mrec,
  xt::pytensor<double,2>& Mlength, 
  xt::pytensor<double,1>& elevation,
  xt::pytensor<bool,1>& active_nodes,
  double dx,
  double dy,
  int nrows,
  int ncols
  )
{
  // Saving general informations
  this->dx = dx;
  this->dy = dy;
  this->cellarea = dx * dy;
  this-> n_element = Sstack.size();
  this-> un_element = Sstack.size();
  this->nrows = nrows;
  this->ncols = ncols;

  // these vectors are additioned to the node indice to test the neighbors
  this->neightbourer.push_back({-ncols - 1, - ncols, - ncols + 1, -1,1,ncols - 1, ncols, ncols + 1 }); // internal node 0
  this->neightbourer.push_back({(nrows - 1) * ncols - 1, (nrows - 1) * ncols, (nrows - 1) * ncols + 1, -1,1,ncols - 1, ncols, ncols + 1 });// periodic_first_row 1
  this->neightbourer.push_back({-ncols - 1, - ncols, - ncols + 1, -1,1,- (nrows - 1) * ncols - 1, - (nrows - 1) * ncols, - (nrows - 1) * ncols + 1 });// periodic_last_row 2
  this->neightbourer.push_back({- 1, - ncols, - ncols + 1, (ncols - 1),1, 2 * ncols - 1, ncols, ncols + 1 });// periodic_first_col 3
  this->neightbourer.push_back({-ncols - 1, - ncols, - 2 * ncols + 1, -1,-(ncols -1), ncols - 1, ncols,  1 }); //periodic last_col 4

  double diag = std::sqrt(std::pow(dx,2) + std::pow(dy,2));
  this->lengthener = {diag,dy,diag,dx,dx,diag,dy,diag};

  // Initialising the vector of pit to reroute. The pits to reroute are all the local minima that are rerouted to another basin/edge
  // It does not include non active nodes (i.e. base level of the model/output of the model)
  pits_to_reroute = std::vector<bool>(un_element,false);

  //# Iterating through the nodes on the stack to gather all the pits
  for(auto node:Sstack)
  {
    // If a pit is its single-flow receiver and an active node it is a pit to reroute
    if(Srec[node] == node && active_nodes[node])
    {
      pits_to_reroute[node] = true;
    }
  }

  // Now I am labelling my basins
  //# Initialising the vector of basin labels
  std::vector<int> basin_labels(Sstack.size(),-1);
  //# Initialising the label to 0
  int label = 0;
  //# Iterating through all my single-flow stack from bottom to top (see Braun and Willett 2013 for meaning of stack)
  //# if a node is it's receiver then the label becomse that node. The resulting basin vector show which nodes are linked to which
  for(auto node:Sstack)
  {
    if(node == Srec[node])
    {
      label = node;
    }
    basin_labels[node] = label;
  }

  // // Initialising a temporary donor matrix to match the fastscape one
  // xt::pytensor<int,2> Mdon({int(un_element), 8});
  // for(size_t i=0;i<un_element;i++)
  // for(size_t j=0; j<8;j++)
  //   Mdon(i,j) = -1;

  // // Correcting receivers from fastscapelib (removing duplicates and creating the donors array)
  // this->initial_correction_of_MF_receivers_and_donors(Mrec, Mdon, elevation);

  graph.reserve(un_element);
  for(int i=0; i<n_element;i++)
    graph.emplace_back(Vertex(i));

  this->compute_receveivers_and_donors(active_nodes,elevation);

  // Initialising the node graph, a vector of Vertexes with their edges
  //# I will need a vector gathering the nodes I'll need to check for potential cyclicity
  std::vector<int> node_to_check;
  //# their target basin
  std::vector<int> force_target_basin;
  //# and the origin of the pit
  std::vector<int> origin_pit;
  //#Iterating through the nodes
  for(int i=0; i<n_element;i++)
  {
    // // gathering local info for this vertex
    // std::vector<int>donors,receivers;
    // std::vector<double> length2rec;
    // for(size_t j=0; j<8;j++)
    // {
    //   // Getting donors
    //   int nD = Mdon(i,j);
    //   // -1 -> not a donor (fortran does not have easily usable sparse matrix)
    //   if(nD>=0 && nD != i)
    //     donors.push_back(nD);
    //   // Same with receivers
    //   int nR = Mrec(i,j);
    //   if(nR>=0)
    //   {
    //     receivers.push_back(nR);
    //     length2rec.push_back(Mlength(i,j));
    //   }
    // }
    // Done with classic receivers
    // If this receiver is a pit to reroute, I need to add to the receiver list the receiver of the outlet
    if(pits_to_reroute[i] == true)
    {
      //# Node I wanna add
      int tgnode = Prec[Prec[i]];
      //# Will be a receiver
      this->graph[i].receivers.push_back(tgnode);
      //# I will have to check its own receivers for potential cyclicity
      node_to_check.push_back(tgnode);
      //# Which pit is it connected to
      origin_pit.push_back(i);
      //# Arbitrary length
      this->graph[i].length2rec.push_back(1);
      //# The basin I DONT want to reach
      force_target_basin.push_back(i);

      // Keepign this check just in case, will remove later. Throw an error in case cyclicity is detected
      if(basin_labels[tgnode] == basin_labels[i])
      {
        throw std::runtime_error("Receiver in same basin!");
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
      if(basin_labels[trec] != this_target_basin || trec == this_node_to_check || active_nodes[trec] == false)
      {
        new_rec.push_back(trec);
        new_length.push_back(this->graph[this_node_to_check].length2rec[idL]);
      }
      idL++;
    }
    // Security check
    if(new_rec.size()==0 && active_nodes[this_node_to_check] == 1 )
      throw std::runtime_error("At node " + std::to_string(this_node_to_check) + " no receivers after corrections! it came from pit " + std::to_string(origin_pit[i]));
    // Correcting the rec
    this->graph[this_node_to_check].receivers = new_rec;
    this->graph[this_node_to_check].length2rec = new_length;
  }

  // I am now ready to create my topological order utilising a c++ port of the fortran algorithm from Jean Braun
  bool has_failed = false;
  Mstack = xt::adapt(multiple_stack_fastscape( n_element, graph, this->not_in_stack, has_failed));
  // int correction_level = 2;
  // while(has_failed)
  // {
  //   std::cout << "Fast stack calculation has failed, correcting it, this slows down the model, if significantly remind me to sort this problem" << std::endl;
  //   this->fix_cyclicity(node_to_check, Sstack, Srec,Prec,Mrec, correction_level);
  //   has_failed = false;
  //   std::cout << this->not_in_stack.size() << " nodes not in stack before" << std::endl;
  //   this->not_in_stack ={};
  //   Mstack = xt::adapt(multiple_stack_fastscape( n_element, graph, this->not_in_stack, has_failed));
  //   std::cout << this->not_in_stack.size() << " nodes not in stack after" << std::endl;
  //   correction_level ++;
  //   // if(has_failed)
  //   //   throw std::runtime_error("Stack failed second iteration, critical error");
  // }

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
    //   rec.push_back(trec);
    // }
    std::vector<int> to_recompute = {node_to_check[i]};
    this->compute_receveivers_and_donors(active_nodes,elevation,to_recompute);

    // VERY IMPORTANT HERE!!!!!
    // In the particular case where my outlet is *also* a pit, I do not want to remove its receiver
    if(pits_to_reroute[node_to_check[i]])
    {
      this->graph[node_to_check[i]].receivers.push_back(Prec[Prec[node_to_check[i]]]);
      this->graph[node_to_check[i]].length2rec.push_back(1);
    }

    // // Correcting the Vertex inplace
    // graph[node_to_check[i]].receivers = rec;
  }
  
  //Done


  return;
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

   // TRY 1: failed
  // for(auto& node:node_to_check)
  // {
  //   if(is_not_in_stack[node])
  //     nodes_to_double_check.push_back(node);

  // }
  // std::cout << "Potential problem on nodes::";
  // for (auto node: nodes_to_double_check)
  //   std::cout << node << "||";
  // std::cout << std::endl;
  // std::cout << " I am on it ..." << std::endl;

  // std::vector<int> index_in_oStack(this->un_element);
  // for(size_t i =0; i< Sstack.size();i++)
  //   index_in_oStack[Sstack[i]] = int(i);

  // for(auto& node: nodes_to_double_check)
  // {
  //   std::vector<int> old_rec = this->graph[node].receivers;
  //   int min_rec = std::numeric_limits<int>::min();
  //   int tid_stack = std::numeric_limits<int>::min();
  //   for(auto rec:old_rec)
  //   {
  //     if(index_in_oStack[rec]>tid_stack)
  //     {
  //       tid_stack = index_in_oStack[rec];
  //       min_rec = rec;
  //     }
  //   }
  //   if(old_rec.size()>0)
  //     this->graph[node].receivers = {min_rec};
  // }

  // TRY 2:
  // Label the basins and identify the pits in the nodes to correct
  std::cout << "1" << std::endl;
  std::vector<int> baslab(this->un_element);
  int tlabel = Sstack[0];
  for(size_t i=0; i< this->un_element; i++)
  {
    if(Sstack[i] == Srec[Sstack[i]])
    {
      tlabel = Sstack[i];
      nodes_to_double_check.push_back(Sstack[i]);

    }
    baslab[Sstack[i]] = tlabel;
  }
  std::cout << "2" << std::endl;

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
            gurg.push_back(Mrec(new_rec,j));
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
            gurg.push_back(rec);
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

  for(auto& i:nodes_to_compute)
  {
    if(active_nodes[i] == false)
      continue;
    std::vector<int> receivers,donors;
    std::vector<double> length2rec,length2don;
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

    // if(checker>0)
    //   std::cout << "CHECKER IS " << checker << " ||";

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
        receivers.push_back(node);
        length2rec.push_back(this->lengthener[idL]);
      }
      else if(test_elev>this_elev)
      {
        donors.push_back(node);
        length2don.push_back(this->lengthener[idL]);
      }
    }
    this->graph[i].receivers = receivers;
    this->graph[i].donors = donors;
    this->graph[i].length2rec = length2rec;
    this->graph[i].length2don = length2don;

    if(i==6299)
    {
      std::cout << "CHECKER WAS :: " << checker << std::endl;
      std::cout << "receivers::";
      for(auto r:receivers)
        std::cout << r << "||";
      std::cout<<std::endl << "donors::";
      for(auto r:donors)
        std::cout << r << "||";
      std::cout<<std::endl;
    }
  }


}

void NodeGraphV2::get_D8_neighbors(int i, xt::pytensor<bool,1>& active_nodes, std::vector<int>& neightbouring_nodes, std::vector<double>& length2neigh)
{
  // these vectors are additioned to the node indice to test the neighbors
  neightbouring_nodes = std::vector<int>();
  length2neigh = std::vector<double>();

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

  int idL = -1;
  for(auto& adder:this->neightbourer[checker])
  {
    int node = i+adder;
    idL++;
    neightbouring_nodes.push_back(node);
    length2neigh.push_back(this->lengthener[idL]);
  }
}

void NodeGraphV2::get_D4_neighbors(int i, xt::pytensor<bool,1>& active_nodes, std::vector<int>& neightbouring_nodes, std::vector<double>& length2neigh)
{
  // these vectors are additioned to the node indice to test the neighbors
  neightbouring_nodes = std::vector<int>();
  length2neigh = std::vector<double>();

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
    neightbouring_nodes.push_back(node);
    length2neigh.push_back(this->lengthener[it]);
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
        not_in_stack.push_back(i);
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
          isolated_nodes.push_back(node);
          min_donor.push_back(min_ID);
          score.push_back(min_score);
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
  std::cout << "BUTE::" << SAFE_STACK_STOPPER << std::endl;

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
      Sstack[istack] = inode;
      istack ++;
      istack = this->_add2stack(inode, istack);
    }
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
  int ipit = 0;
  SBasinID = xt::zeros<int>({this->un_element});

  for(int inode=0; inode<this->n_element;inode++)
  {  
    istack = this->Sstack[inode];
    irec = this->graph[istack].Sreceivers;

    if (irec == istack)
    {
      ibasin ++;
      SBasinOutlets.push_back(istack);
      if(active_nodes[istack])
      {
        pits.push_back(istack);
        ipit++;
      }
    }
    SBasinID[istack] = ibasin;
  }

  this->nbasins = ibasin + 1;
  this->npits = ipit;
}

void NodeGraphV2::correct_flowrouting(xt::pytensor<bool,1>& active_nodes, xt::pytensor<double,1>& elevation)
{
    // """Ensure that no flow is captured in sinks.

    // If needed, update `receivers`, `dist2receivers`, `ndonors`,
    // `donors` and `stack`.

    // """
    int& nnodes = this->n_element;

    // # theory of planar graph -> max nb. of connections known
    int nconn_max = this->nbasins * 3;
    xt::pytensor<int,2> conn_basins = xt::zeros<int>({nconn_max,2}); // np.empty((nconn_max, 2), dtype=np.intp)
    for(size_t i=0;i<nconn_max;i++)
    {
      conn_basins(i,1) = -2;
      conn_basins(i,0) = -2;
    }
    xt::pytensor<int,2> conn_nodes = xt::zeros<int>({nconn_max,2}); //    conn_nodes = np.empty((nconn_max, 2), dtype=np.intp)
    for(size_t i=0;i<nconn_max;i++)
    {
      conn_nodes(i,1) = -2;
      conn_nodes(i,0) = -2;
    }
    xt::pytensor<double,1> conn_weights = xt::zeros<double>({nconn_max}); //conn_weights = np.empty(nconn_max, dtype=np.float64)
    for(auto& v:conn_weights)
      v = -2;

    int nconn, basin0;
    this->_connect_basins(conn_basins, conn_nodes, conn_weights, active_nodes, elevation, nconn, basin0);

    // if method == 'mst_linear':
    //     mstree = _compute_mst_linear(conn_basins, conn_weights, nbasins)
    // elif method == 'mst_kruskal':
    //     mstree = _compute_mst_kruskal(conn_basins, conn_weights, nbasins)
    // else:
    //     raise ValueError("invalid flow correction method %r" % method)
    xt::xtensor<int,1> mstree = _compute_mst_kruskal(conn_basins, conn_weights);

    this->_orient_basin_tree(conn_basins,conn_nodes,basin0, mstree);

    // _update_pits_receivers(receivers, dist2receivers, outlets,
    //                        conn_basins, conn_nodes,
    //                        mstree, elevation, nx, dx, dy)
    // compute_donors(ndonors, donors, receivers, nnodes)
    // compute_stack(stack, ndonors, donors, receivers, nnodes)
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

  xt::pytensor<int,1> conn_pos = xt::zeros<int>({size_t(this->nbasins)}); //np.full(nbasins, -1, dtype=np.intp)
  for(auto& v:conn_pos)
    v = -1;


  xt::pytensor<int,1> conn_pos_used = xt::zeros<int>({size_t(this->nbasins)});  // conn_pos_used = np.empty(nbasins, dtype=np.intp)

  int conn_pos_used_size = 0;

  bool iactive = false;

  // # king (D4) neighbor lookup
  std::vector<int> dr = {0, -1, 1, 0};
  std::vector<int> dc = {-1, 0, 0, 1};

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

          conn_weights[iconn] = std::numeric_limits<double>::min();
          iconn ++;
        }
      }
    }

    if(iactive)
    {
      std::vector<int> D4n;
      std::vector<double> D4l;
      this->get_D4_neighbors(istack,active_nodes, D4n, D4l);
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
          conn_nodes(conn_idx,0) = ineighbor;
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
  xt::xtensor<int,1> mstree = xt::empty<int>({nbasins - 1});
  int mstree_size = 0;

  // # sort edges
  auto sort_id = xt::argsort(conn_weights);

  UnionFind uf(nbasins);

  for (auto& eid : sort_id)
  {
    // ignoring not basin
    if(conn_weights[eid] == -2)
      continue;

    int b0 = conn_basins(eid, 0);
    int b1 = conn_basins(eid, 1);

    if (uf.Find(b0) != uf.Find(b1))
    {
      mstree[mstree_size] = eid;
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
  for (auto& i : tree)
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
  int nodes_adjacency_size = nodes_connects_ptr[-1] + nodes_connects_size[-1];
  nodes_connects_size[-1] = 0;
  xt::xtensor<int,1> nodes_adjacency = xt::zeros<int>({nodes_adjacency_size});

  // # parse the edges to update the adjacency
  for (auto& i : tree)
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
  xt::xtensor<int,2> stack = xt::empty<int>({nbasins, 2});
  int stack_size = 1;
  stack(0,0) = basin0;// (basin0, basin0)
  stack(0,1) = basin0;

  while (stack_size > 0)
  {
    // # get parsed node
    stack_size -= 1;
    int node = stack(stack_size, 0);
    int parent = stack(stack_size, 1);

    // # for each edge of the graph
    // for i in range(nodes_connects_ptr[node], nodes_connects_ptr[node] + nodes_connects_size[node])
    for( int i = nodes_connects_ptr[node]; i< nodes_connects_ptr[node] + nodes_connects_size[node]; i++) 
    {
      int edge_id = nodes_adjacency[i];

      // # the edge comming from the parent node has already been updated.
      // # in this case, the edge is (parent, node)
      if (conn_basins(edge_id, 0) == parent && node != parent || conn_basins(edge_id, 0) == -2)
          continue;

      // # we want the edge to be (node, next)
      // # we check if the first node of the edge is not "node"
      if(node != conn_basins(edge_id, 0))
      {
        // # swap n1 and n2
        conn_basins(edge_id, 0), conn_basins(edge_id, 1) = (conn_basins(edge_id, 1), conn_basins(edge_id, 0));
        // # swap p1 and p2
        conn_nodes(edge_id, 0), conn_nodes(edge_id, 1) = (conn_nodes(edge_id, 1), conn_nodes(edge_id, 0));
      }
      // # add the opposite node to the stack
      stack[stack_size] = (conn_basins(edge_id, 1), node);
      stack_size ++;
    }
  }
}



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



// DEPRECATED LAND, keeping for legacy

// Set of functions managing graph traversal and topological sorts, lightweight and customisable


// // Depth First Function utilised for customised topological sorting from a custom set of points.
// bool dfs(Vertex& vertex, // The investigated vertex
//  std::vector<Vertex>& stack, // The stack of vertexes (current topological sort)
//  std::vector<int>& next_vertexes, // next vertexes to be investigated during the topological sort
//  int& index_of_reading, // ignore this
//  int& index_of_pushing, // index incrementing the next vertexes in the topological sort ("local modified" topological sort)
//  std::vector<bool>& is_in_queue, // Check if a node has already been added in the next vertexes
//  std::vector<Vertex>& graph, // the original, unsorted graph containing all nodes
//  std::string& direction // Direction being "donors" or "receivers" to deterine in which way the DAG is considered
//  ) 
// {
//     // std::cout << "6.0" << std::endl;
//     vertex.visiting = true;
//     // std::cout << "6.0.1" << std::endl;
//     const std::vector<int>& childNode = (direction == "donors")? vertex.donors : vertex.receivers;

//     for (auto chid : childNode) 
//     {
//       // std::cout << "6.1" << std::endl;
//       if(chid<0)
//           continue;
//       // std::cout << "6.2" << std::endl;
//       Vertex& childNode = graph[chid];

//       // std::cout << "6.3::" << childNode.val << std::endl;
//       if (childNode.visited == false) 
//       {
//         // std::cout << "6.4" << std::endl;
//         if(is_in_queue[childNode.val] == false)
//         {
//           next_vertexes[index_of_pushing] = childNode.val;
//           is_in_queue[childNode.val] = true;
//           index_of_pushing++;
//         }
//         // std::cout << "6.5" << std::endl;
//         // check for back-edge, i.e., cycle
//         if (childNode.visiting) 
//         {
//           return false;
//         }
//         // std::cout << "6.6::" << vertex.val << std::endl;

//         bool childResult = dfs(childNode, stack, next_vertexes, index_of_reading, index_of_pushing, is_in_queue, graph, direction);

//         if (childResult == false) 
//         {
//           return false;
//         }
//         // std::cout << "6.7" << std::endl;
            
//       }
//     }
    
//     // std::cout << "6.8" << std::endl;
    
//     // now that you have completely visited all the
//     // donors of the vertex, push the vertex in the stack

//     stack.emplace_back(vertex);

//     // std::cout << "6.9" << std::endl;


//     // this vertex is processed (all its descendents, i.e,
//     // the nodes that are dependent on
//     // this vertex along with this vertex itself have been visited 
//     // and processed). So mark this vertex as Visited.
//     vertex.visited = true;
//     // std::cout << "6.10" << std::endl;

    
//     // mark vertex as visiting as false since we 
//     // have completed visiting all the subtrees of 
//     // the vertex, including itself. So the vertex is no more
//     // in visiting state because it is done visited.
//     // Another reason is (the critical one) now that all the 
//     // descendents of the vertex are visited, any future 
//     // inbound edge to the vertex won't be a back edge anymore. 
//     vertex.visiting = false;  
//     // std::cout << "6.11" << std::endl;


//     return true;
// }

// // Second version of the depth first algorithm where the original set of nodes is fixed, eg full topological sorting
// bool dfs(Vertex& vertex,
//  std::vector<Vertex>& stack,
//  std::vector<Vertex>& graph,
//  std::string& direction
//  ) 
// {
//     // std::cout << "6.0" << std::endl;
//     vertex.visiting = true;
//     // std::cout << "6.0.1" << std::endl;
//     const std::vector<int>& childNode = (direction == "donors")? vertex.donors : vertex.receivers;

//     for (auto chid : childNode) 
//     {
//       // std::cout << "6.1" << std::endl;
//       if(chid<0)
//           continue;
//       // std::cout << "6.2" << std::endl;
//       Vertex& childNode = graph[chid];

//       // std::cout << "6.3::" << childNode.val << std::endl;
//       if (childNode.visited == false) 
//       {
//         if (childNode.visiting) 
//         {
//           return false;
//         }

//         bool childResult = dfs(childNode, stack, graph, direction);

//         if (childResult == false) 
//         {
//           return false;
//         }
//         // std::cout << "6.7" << std::endl;
            
//       }
//     }
    
//     // std::cout << "6.8" << std::endl;
    
//     // now that you have completely visited all the
//     // donors of the vertex, push the vertex in the stack

//     stack.emplace_back(vertex);

//     // std::cout << "6.9" << std::endl;


//     // this vertex is processed (all its descendents, i.e,
//     // the nodes that are dependent on
//     // this vertex along with this vertex itself have been visited 
//     // and processed). So mark this vertex as Visited.
//     vertex.visited = true;
//     // std::cout << "6.10" << std::endl;

    
//     // mark vertex as visiting as false since we 
//     // have completed visiting all the subtrees of 
//     // the vertex, including itself. So the vertex is no more
//     // in visiting state because it is done visited.
//     // Another reason is (the critical one) now that all the 
//     // descendents of the vertex are visited, any future 
//     // inbound edge to the vertex won't be a back edge anymore. 
//     vertex.visiting = false;  
//     // std::cout << "6.11" << std::endl;


//     return true;
// }


//DEPRECATED
// // returns null if no topological sort is possible
// std::vector<int> topological_sort_by_dfs(std::vector<Vertex>& graph, int starting_node, std::string& direction) 
// {
//     std::vector<Vertex> stack;
//     std::vector<int> next_vertexes(graph.size(), -1);
//     std::vector<bool> is_in_queue(graph.size(), false);
//     // std::cout << "1" << std::endl;
//     // next_vertexes.reserve(graph.size());
//     next_vertexes[0] = starting_node;
//     is_in_queue[starting_node] = true;
//     int index_of_reading = 0;
//     int index_of_pushing = 1;
//     // std::cout << "2" << std::endl;

//     while(true)
//     {
//         // std::cout << "3" << std::endl;

//         int next_ID = next_vertexes[index_of_reading];
//         index_of_reading++;
//         // if(next_ID==1714)
//         //   std::cout << "GURG2.0!!!!!" << std::endl;
//         // std::cout << "4" << std::endl;
//         if(next_ID == -1)
//             break;
//         // std::cout << "5" << std::endl;
//         Vertex vertex = graph[next_ID];
//         // std::cout << "6" << std::endl;
//         if (vertex.visited ==  false) 
//         {
//             bool dfs_result = dfs(vertex, stack, next_vertexes, index_of_reading, index_of_pushing, is_in_queue, graph,direction);
//             // std::cout << "6bis" << std::endl;

//             // if cycle found then there is no topological sort possible
//             if (dfs_result == false) 
//             {
//               // std::cout << "gabul" << std::endl;
//                 return {-9999};
//             }
//         }
//         // std::cout << "7" << std::endl;
//     }

//     stack.shrink_to_fit();
//     std::vector<int> result;result.reserve(stack.size());

//     for (auto vertex : stack) {
//         result.emplace_back(vertex.val);
//     }
//     result.shrink_to_fit();
//     return result;
// }

// // returns null if no topological sort is possible
// std::vector<int> topological_sort_by_dfs(std::vector<Vertex>& graph, std::string& direction) 
// {
//     std::vector<Vertex> stack;
//     stack.reserve(graph.size());


//     for(int next_ID = 0; next_ID< graph.size(); next_ID++)
//     {
//         Vertex& vertex = graph[next_ID];
//         if (vertex.visited ==  false) 
//         {
//             bool dfs_result = dfs(vertex, stack, graph,direction);
//             // std::cout << "6bis" << std::endl;

//             // if cycle found then there is no topological sort possible
//             if (dfs_result == false) 
//             {
//               // std::cout << "gabul" << std::endl;
//                 return {-9999};
//             }
//         }
//         // std::cout << "7" << std::endl;
//     }

//     stack.shrink_to_fit();
//     std::vector<int> result;result.reserve(stack.size());

//     for (auto vertex : stack) 
//     {
//         result.emplace_back(vertex.val);
//     }
//     result.shrink_to_fit();
//     return result;
// }

#endif