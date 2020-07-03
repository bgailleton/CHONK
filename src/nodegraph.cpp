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


// Set of functions managing graph traversal and topological sorts, lightweight and customisable


// Depth First Function utilised for customised topological sorting from a custom set of points.
bool dfs(Vertex& vertex, // The investigated vertex
 std::vector<Vertex>& stack, // The stack of vertexes (current topological sort)
 std::vector<int>& next_vertexes, // next vertexes to be investigated during the topological sort
 int& index_of_reading, // ignore this
 int& index_of_pushing, // index incrementing the next vertexes in the topological sort ("local modified" topological sort)
 std::vector<bool>& is_in_queue, // Check if a node has already been added in the next vertexes
 std::vector<Vertex>& graph, // the original, unsorted graph containing all nodes
 std::string& direction // Direction being "donors" or "receivers" to deterine in which way the DAG is considered
 ) 
{
    // std::cout << "6.0" << std::endl;
    vertex.visiting = true;
    // std::cout << "6.0.1" << std::endl;
    const std::vector<int>& childNode = (direction == "donors")? vertex.donors : vertex.receivers;

    for (auto chid : childNode) 
    {
      // std::cout << "6.1" << std::endl;
      if(chid<0)
          continue;
      // std::cout << "6.2" << std::endl;
      Vertex& childNode = graph[chid];

      // std::cout << "6.3::" << childNode.val << std::endl;
      if (childNode.visited == false) 
      {
        // std::cout << "6.4" << std::endl;
        if(is_in_queue[childNode.val] == false)
        {
          next_vertexes[index_of_pushing] = childNode.val;
          is_in_queue[childNode.val] = true;
          index_of_pushing++;
        }
        // std::cout << "6.5" << std::endl;
        // check for back-edge, i.e., cycle
        if (childNode.visiting) 
        {
          return false;
        }
        // std::cout << "6.6::" << vertex.val << std::endl;

        bool childResult = dfs(childNode, stack, next_vertexes, index_of_reading, index_of_pushing, is_in_queue, graph, direction);

        if (childResult == false) 
        {
          return false;
        }
        // std::cout << "6.7" << std::endl;
            
      }
    }
    
    // std::cout << "6.8" << std::endl;
    
    // now that you have completely visited all the
    // donors of the vertex, push the vertex in the stack

    stack.emplace_back(vertex);

    // std::cout << "6.9" << std::endl;


    // this vertex is processed (all its descendents, i.e,
    // the nodes that are dependent on
    // this vertex along with this vertex itself have been visited 
    // and processed). So mark this vertex as Visited.
    vertex.visited = true;
    // std::cout << "6.10" << std::endl;

    
    // mark vertex as visiting as false since we 
    // have completed visiting all the subtrees of 
    // the vertex, including itself. So the vertex is no more
    // in visiting state because it is done visited.
    // Another reason is (the critical one) now that all the 
    // descendents of the vertex are visited, any future 
    // inbound edge to the vertex won't be a back edge anymore. 
    vertex.visiting = false;  
    // std::cout << "6.11" << std::endl;


    return true;
}

// SEcond version of the depth first algorithm where the original set of nodes is fixed, eg full topological sorting
bool dfs(Vertex& vertex,
 std::vector<Vertex>& stack,
 std::vector<Vertex>& graph,
 std::string& direction
 ) 
{
    // std::cout << "6.0" << std::endl;
    vertex.visiting = true;
    // std::cout << "6.0.1" << std::endl;
    const std::vector<int>& childNode = (direction == "donors")? vertex.donors : vertex.receivers;

    for (auto chid : childNode) 
    {
      // std::cout << "6.1" << std::endl;
      if(chid<0)
          continue;
      // std::cout << "6.2" << std::endl;
      Vertex& childNode = graph[chid];

      // std::cout << "6.3::" << childNode.val << std::endl;
      if (childNode.visited == false) 
      {
        if (childNode.visiting) 
        {
          return false;
        }

        bool childResult = dfs(childNode, stack, graph, direction);

        if (childResult == false) 
        {
          return false;
        }
        // std::cout << "6.7" << std::endl;
            
      }
    }
    
    // std::cout << "6.8" << std::endl;
    
    // now that you have completely visited all the
    // donors of the vertex, push the vertex in the stack

    stack.emplace_back(vertex);

    // std::cout << "6.9" << std::endl;


    // this vertex is processed (all its descendents, i.e,
    // the nodes that are dependent on
    // this vertex along with this vertex itself have been visited 
    // and processed). So mark this vertex as Visited.
    vertex.visited = true;
    // std::cout << "6.10" << std::endl;

    
    // mark vertex as visiting as false since we 
    // have completed visiting all the subtrees of 
    // the vertex, including itself. So the vertex is no more
    // in visiting state because it is done visited.
    // Another reason is (the critical one) now that all the 
    // descendents of the vertex are visited, any future 
    // inbound edge to the vertex won't be a back edge anymore. 
    vertex.visiting = false;  
    // std::cout << "6.11" << std::endl;


    return true;
}

// returns null if no topological sort is possible
std::vector<int> topological_sort_by_dfs(std::vector<Vertex>& graph, int starting_node, std::string& direction) 
{
    std::vector<Vertex> stack;
    std::vector<int> next_vertexes(graph.size(), -1);
    std::vector<bool> is_in_queue(graph.size(), false);
    // std::cout << "1" << std::endl;
    // next_vertexes.reserve(graph.size());
    next_vertexes[0] = starting_node;
    is_in_queue[starting_node] = true;
    int index_of_reading = 0;
    int index_of_pushing = 1;
    // std::cout << "2" << std::endl;

    while(true)
    {
        // std::cout << "3" << std::endl;

        int next_ID = next_vertexes[index_of_reading];
        index_of_reading++;
        // if(next_ID==1714)
        //   std::cout << "GURG2.0!!!!!" << std::endl;
        // std::cout << "4" << std::endl;
        if(next_ID == -1)
            break;
        // std::cout << "5" << std::endl;
        Vertex vertex = graph[next_ID];
        // std::cout << "6" << std::endl;
        if (vertex.visited ==  false) 
        {
            bool dfs_result = dfs(vertex, stack, next_vertexes, index_of_reading, index_of_pushing, is_in_queue, graph,direction);
            // std::cout << "6bis" << std::endl;

            // if cycle found then there is no topological sort possible
            if (dfs_result == false) 
            {
              // std::cout << "gabul" << std::endl;
                return {-9999};
            }
        }
        // std::cout << "7" << std::endl;
    }

    stack.shrink_to_fit();
    std::vector<int> result;result.reserve(stack.size());

    for (auto vertex : stack) {
        result.emplace_back(vertex.val);
    }
    result.shrink_to_fit();
    return result;
}

// returns null if no topological sort is possible
std::vector<int> topological_sort_by_dfs(std::vector<Vertex>& graph, std::string& direction) 
{
    std::vector<Vertex> stack;
    stack.reserve(graph.size());


    for(int next_ID = 0; next_ID< graph.size(); next_ID++)
    {
        Vertex& vertex = graph[next_ID];
        if (vertex.visited ==  false) 
        {
            bool dfs_result = dfs(vertex, stack, graph,direction);
            // std::cout << "6bis" << std::endl;

            // if cycle found then there is no topological sort possible
            if (dfs_result == false) 
            {
              // std::cout << "gabul" << std::endl;
                return {-9999};
            }
        }
        // std::cout << "7" << std::endl;
    }

    stack.shrink_to_fit();
    std::vector<int> result;result.reserve(stack.size());

    for (auto vertex : stack) 
    {
        result.emplace_back(vertex.val);
    }
    result.shrink_to_fit();
    return result;
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

  // Initialising a temporary donor matrix to match the fastscape one
  xt::pytensor<int,2> Mdon({int(un_element), 8});
  for(size_t i=0;i<un_element;i++)
  for(size_t j=0; j<8;j++)
    Mdon(i,j) = -1;

  // Correcting receivers from fastscapelib (removing duplicates and creating the donors array)
  this->initial_correction_of_MF_receivers_and_donors(Mrec, Mdon, elevation);

  // Initialising the node graph, a vector of Vertexes with their edges
  graph.reserve(un_element);
  //# I will need a vector gathering the nodes I'll need to check for potential cyclicity
  std::vector<int> node_to_check;
  //# their target basin
  std::vector<int> force_target_basin;
  //# and the origin of the pit
  std::vector<int> origin_pit;
  //#Iterating through the nodes
  for(int i=0; i<n_element;i++)
  {
    // gathering local info for this vertex
    std::vector<int>donors,receivers;
    std::vector<double> length2rec;
    for(size_t j=0; j<8;j++)
    {
      // Getting donors
      int nD = Mdon(i,j);
      // -1 -> not a donor (fortran does not have easily usable sparse matrix)
      if(nD>=0 && nD != i)
        donors.push_back(nD);
      // Same with receivers
      int nR = Mrec(i,j);
      if(nR>=0)
      {
        receivers.push_back(nR);
        length2rec.push_back(Mlength(i,j));
      }
    }
    // Done with classic receivers
    // If this receiver is a pit to reroute, I need to add to the receiver list the receiver of the outlet
    if(pits_to_reroute[i] == true)
    {
      //# Node I wanna add
      int tgnode = Prec[Prec[i]];
      //# Will be a receiver
      receivers.push_back(tgnode);
      //# I will have to check its own receivers for potential cyclicity
      node_to_check.push_back(tgnode);
      //# Which pit is it connected to
      origin_pit.push_back(i);
      //# Arbitrary length
      length2rec.push_back(1);
      //# The basin I DONT want to reach
      force_target_basin.push_back(i);

      // Keepign this check just in case, will remove later. Throw an error in case cyclicity is detected
      if(basin_labels[tgnode] == basin_labels[i])
      {
        throw std::runtime_error("Receiver in same basin!");
      }
    }

    // I need these to initialise my Vertex
    bool false1 = false;
    bool false2 = false;
    // Constructing the vertex in place in the vector. More efficient according to the internet
    graph.emplace_back(Vertex(i,false1,false2,donors,receivers,length2rec));
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
    for(auto trec:graph[this_node_to_check].receivers)
    {
      if(basin_labels[trec] != this_target_basin || trec == this_node_to_check || active_nodes[trec] == false)
      {
        new_rec.push_back(trec);
      }

    }
    // Security check
    if(new_rec.size()==0 && active_nodes[this_node_to_check] == 1 )
      throw std::runtime_error("At node " + std::to_string(this_node_to_check) + " no receivers after corrections! it came from pit " + std::to_string(origin_pit[i]));
    // Correcting the rec
    graph[this_node_to_check].receivers = new_rec;
  }

  // I am now ready to create my topological order utilising a c++ port of the fortran algorithm from Jean Braun
  Mstack = xt::adapt(multiple_stack_fastscape( n_element, graph));

  // I got my topological order, I can now restore the corrupted receiver I had
  for(size_t i=0; i<node_to_check.size(); i++)
  {
    // New receiver array
    std::vector<int> rec;
    // Repicking the ones from fastscapelib-fortran
    for(size_t j=0;j<8; j++)
    {
      int trec = Mrec(node_to_check[i],j);
      if(trec < 0 || node_to_check[i] == trec)
      {continue;}
      rec.push_back(trec);
    }
    // VERY IMPORTANT HERE!!!!!
    // In the particular case where my outlet is *also* a pit, I do not want to remove its receiver
    if(pits_to_reroute[node_to_check[i]])
      rec.push_back(Prec[Prec[node_to_check[i]]]);
    // Correcting the Vertex inplace
    graph[node_to_check[i]].receivers = rec;
  }
  
  //Done


  return;
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


std::vector<int> multiple_stack_fastscape(int n_element, std::vector<Vertex>& graph)
{

  std::vector<int>ndon(n_element,0);
  for(size_t i=0; i<n_element;i++)
  {
    for(auto trec:graph[i].receivers)
    {
      ndon[trec]  = ndon[trec] + 1;
    }
  }

  std::vector<int> vis(n_element,0), parse(n_element,-1), stack(n_element,0);
  
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
  // if(nstack < n_element - 1 )
  // {
  //   std::cout << "WARNING::STACK UNDERPOPULATED::" << nstack << std::endl;;
  //   throw std::runtime_error("stack underpopulated somehow:");
  // }

  return stack;



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
}

// void find_multiple_receivers_and_donors(xt::pytensor<double>& elevation)
// {

  



//   do j=bounds_j1,bounds_j2
//     do i=bounds_i1,bounds_i2
//       ij = (j-1)*nx + i
//       slopemax = 0.
//       do jj=-1,1
//         jjj= j + jj
//         if (jjj.lt.1.and.bounds_ycyclic) jjj=jjj+ny
//         jjj=max(jjj,1)
//         if (jjj.gt.ny.and.bounds_ycyclic) jjj=jjj-ny
//         jjj=min(jjj,ny)
//         do ii=-1,1
//           iii = i + ii
//           if (iii.lt.1.and.bounds_xcyclic) iii=iii+nx
//           iii=max(iii,1)
//           if (iii.gt.nx.and.bounds_xcyclic) iii=iii-nx
//           iii=min(iii,nx)
//           ijk = (jjj-1)*nx + iii
//           if (h0(ij).gt.h0(ijk)) then
//             nrec(ij)=nrec(ij)+1
//             rec(nrec(ij),ij) = ijk
//             lrec(nrec(ij),ij) = sqrt((ii*dx)**2 + (jj*dy)**2)
//             wrec(nrec(ij),ij) = (h0(ij) - h0(ijk))/lrec(nrec(ij),ij)
//           endif
//         enddo
//       enddo
//     enddo
//   enddo
// }








































// // This empty constructor is just there to have a default one.
// void NodeGraph::create()
// {
//   std::string yo = "I am an empty constructor yo!";

// }


// // This constructor is now deprecated, too messy ...
// void NodeGraph::create(xt::pytensor<int,1>& pre_stack,xt::pytensor<int,1>& pre_rec, xt::pytensor<int,1>& post_rec, xt::pytensor<int,1>& post_stack,
//   xt::pytensor<int,1>& tMF_stack, xt::pytensor<int,2>& tMF_rec,xt::pytensor<int,2>& tMF_don, xt::pytensor<double,1>& elevation, xt::pytensor<double,2>& tMF_length,
//   float XMIN, float XMAX, float YMIN, float YMAX, float XRES, float YRES, int NROWS, int NCOLS, float NODATAVALUE)
// {
//   auto start = std::chrono::steady_clock::now();
//   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//   // Step N:  
//   // I am first correcting the donors using the receivers: the receivers seems alright but somehow my donors are buggy
//   // This is simply done by inverting the receiver to the donors
//   // Also extra step I need to remove the duplicate receivers
//   this->initial_correction_of_MF_receivers_and_donors( post_stack, tMF_rec, tMF_don, elevation);
//   auto end = std::chrono::steady_clock::now();
//   std::chrono::duration<double>  dat_time = end - start;
//   std::cout<< "TIMER_NODEGRAPH_C1::" << dat_time.count() << std::endl;
//   start = std::chrono::steady_clock::now();


//   // Inithalising general attributes
//   this->NROWS = NROWS;
//   this->NCOLS = NCOLS;
//   this->XMIN = XMIN;
//   this->XMAX = XMAX;
//   this->YMIN = YMIN;
//   this->YMAX = YMAX;
//   this->XRES = XRES;
//   this->YRES = YRES;
//   this->NODATAVALUE = NODATAVALUE;
//   this->MF_stack = tMF_stack;
//   this->MF_receivers = tMF_rec;
//   this->MF_lengths = tMF_length;
//   this->MF_donors = tMF_don;

//   int nint_element = int(this->MF_stack.size());
//   size_t nuint_element = this->MF_stack.size();

//   // Now I need to post-process the MF stack

//   // std::vector<std::vector<int> > basin_multi_label(nuint_element);
//   std::vector<int> all_base_levels_nocorr; //, index_in_first_MFstack(nint_element);
//   // std::vector<double> elev_BL;
//   std::map<int,int> BL_to_index;

//   pit_to_reroute  = std::vector<bool>(nuint_element,false);

//   std::vector<double> separated_trees_elev_BL;
//   std::vector<std::vector<int> > separated_trees;
//   std::map<int,int> BL_to_sep_tree;
//   int index_st = 0;
//   separated_trees.push_back({});

//   std::vector<bool> labelz(pre_stack.size(), false);
//   int preincr = -1;
//   for (auto sta:pre_stack)
//   {
//     if(sta == pre_rec[sta])
//       preincr++;
//     labelz[sta] = preincr;
//   }

//   // xt::pytensor<int,1> basin_labels = xt::zeros<int>({nuint_element});

//   // // Using Cordonnier's algorithm: 
//   // fastscapelib::BasinGraph<int,int,double> Bgraph;
//   // //# Computing basins
//   // Bgraph.compute_basins(basin_labels, pre_stack, pre_rec); 
//   // //# update receivers
//   // Bgraph.update_receivers(pre_rec,  dist2receivers,
//   //                         const Basins_XT& basins,
//   //                         const Stack_XT& stack, const Active_XT& active_nodes,
//   //                         const Elevation_XT& elevation, Elevation_T dx, Elevation_T dy);



//   // std::cout <<"URG" << std::endl;
//   // Step I:
//   // # Order my basin by order of preocessing from top to bottom in the corrected stack
//   int incr2 = 0;
//   for(int i = nint_element - 1; i>=0; i--)
//   {
//     int node = post_stack[i];
//     if(pre_rec[node] == node)
//     {
//       all_base_levels_nocorr.push_back(node);
//       incr2++;    
//       if(post_rec[node] != node)
//       {
//         pit_to_reroute[node] = true;
//       }
//     }
//     // basin_multi_label.emplace_back(std::vector<int>());
//   }

//   end = std::chrono::steady_clock::now();
//   dat_time = end - start;
//   std::cout<< "TIMER_NODEGRAPH_C2::" << dat_time.count() << std::endl;
//   start = std::chrono::steady_clock::now();


//   std::vector<Vertex> graph;graph.reserve(nuint_element);
//   for(size_t i = 0; i< nuint_element; i++)
//   {
//     Vertex this_vertex;
//     this_vertex.val = int(i);
//     this_vertex.visiting = false;
//     this_vertex.visited = false;
//     auto dons = this->get_MF_donors_at_node(int(i));
//     std::vector<int> donors;
//     for(auto& datd : dons)
//     {

//       if(datd != int(i) && datd>=0)
//       {
//         if(elevation[datd] >= elevation[int(i)])
//         {
//           donors.push_back(datd);
//         }
//       }
//       // if(elevation[datd]<elevation[int(i)])
//       // {
//       //   std::cout << "!!!!!!" << std::endl;
//       //   exit(EXIT_FAILURE);
//       // }
//     } 
//     this_vertex.donors = donors;

//     // if(this_vertex.val == )

//     graph.emplace_back(this_vertex);
//   }

//   end = std::chrono::steady_clock::now();
//   dat_time = end - start;
//   std::cout<< "TIMER_NODEGRAPH_C3::" << dat_time.count() << std::endl;
//   start = std::chrono::steady_clock::now();

//   std::vector<int> pixel_score(nuint_element,0);
//   std::vector<bool> is_in_stack(nuint_element,false);
//   std::vector<std::vector<int> > vecofvec;
//   int incr = this->MF_stack.size() - 1;


//   for(int no = all_base_levels_nocorr.size()-1 ; no >= 0; no--)
//   // for(int no = 0 ; no < all_base_levels_nocorr.size(); no++)
//   {
//     int node = all_base_levels_nocorr[no];
//     // std::cout << "start the toposort for node "<< node << std::endl;
//     std::string direction = "donors";
//     std::vector<int> tempstack = topological_sort_by_dfs(graph, node, direction);
//     // std::cout << "done tid  dah toposort for node "<< node << std::endl;

//     for(int i =  tempstack.size() - 1; i>=0;i--)
//     // for(int i =  0; i< tempstack.size();i++)
//     {
//       int this_node = tempstack[i];
//       if(is_in_stack[this_node])
//         continue;
//       this->MF_stack[incr] = this_node;
//       incr--;
//       is_in_stack[this_node] = true; 


//     }
//     // vecofvec.push_back(tempstack);
//   }


//   // // int incr = 0;
//   // std::vector<int> incremental_pixel_score(nuint_element,0);
//   // for (int i = vecofvec.size()-1; i >= 0;i--)
//   // // for (int i=0; i<vecofvec.size();i++)
//   // {
//   //   std::vector<int>& this_sub_stack = vecofvec[i];
//   //   for(int j =  this_sub_stack.size() - 1; j>=0;j--)
//   //   // for(int j = 0;j< this_sub_stack.size(); j++)
//   //   {
//   //     int this_node = this_sub_stack[j];
//   //     incremental_pixel_score[this_node] = incremental_pixel_score[this_node] + 1;
//   //     if(incremental_pixel_score[this_node] == pixel_score[this_node])
//   //     {
//   //       this->MF_stack[incr] = this_node;
//   //       // incr++;
//   //       incr--;
//   //     }
//   //   }
//   // }
  
//   //right now I should have what it takes

//   // std::cout << incr + 1 << "/" << this->MF_stack.size() << std::endl;

//   end = std::chrono::steady_clock::now();
//   dat_time = end - start;
//   std::cout<< "TIMER_NODEGRAPH_C4::" << dat_time.count() << std::endl;
//   start = std::chrono::steady_clock::now();



//   // all_base_levels_nocorr now has all the baselevel nodes ordered by graph solving
//   // std::cout <<"URG2" << std::endl;


//   // Older tests, once more

//   // //Step II: label the basins from these nodes on the uncorrected multiple stack
//   // this->label_basins_MF(basin_multi_label, all_base_levels_nocorr, post_rec);
//   // basin_multi_label.shrink_to_fit();

//   // end = std::chrono::steady_clock::now();
//   // dat_time = end - start;
//   // std::cout<< "TIMER_NODEGRAPH_C2::" << dat_time.count() << std::endl;
//   // start = std::chrono::steady_clock::now();

//   // // Step III: breaking my multi-basins nodes to alised nodes to break any cyclicity
//   // std::vector<int> VertexDon,  VertexRec; std::vector<double> VertexLength;
//   // std::vector<bool> has_aliases;
//   // std::unordered_map<int,std::vector<int> > node2aliases;
//   // std::vector<int> aliases2nodes; 
//   // std::unordered_map<int,int> aliases2ID; 
//   // std::vector<std::vector<int> > aliases_rec; 
//   // std::vector<std::vector<int> > aliases_length;
//   // std::vector<int> aliases_basin_recs;
//   // this->generate_vector_of_adjacency_unique_basin(basin_multi_label, VertexDon, VertexRec, VertexLength, has_aliases, node2aliases, aliases2nodes, aliases2ID, aliases_rec, aliases_length,  aliases_basin_recs);
//   // end = std::chrono::steady_clock::now();
//   // dat_time = end - start;
//   // std::cout<< "TIMER_NODEGRAPH_C3::" << dat_time.count() << std::endl;
//   // start = std::chrono::steady_clock::now();

//   // std::cout << "flubhere" << std::endl;
//   // // Step IV: reroute my pits from top to bottom
//   // // this->link_pit_vertex_to_receivers_or_their_aliases(post_rec, all_base_levels_nocorr, basin_multi_label, has_aliases, VertexDon, VertexRec,  
//   // //   VertexLength, aliases2ID ,aliases_basin_recs, node2aliases, aliases2nodes);
//   // VertexDon.shrink_to_fit();
//   // VertexRec.shrink_to_fit();
//   // VertexLength.shrink_to_fit();

//   // end = std::chrono::steady_clock::now();
//   // dat_time = end - start;
//   // std::cout<< "TIMER_NODEGRAPH_C4::" << dat_time.count() << std::endl;
//   // start = std::chrono::steady_clock::now();
//   // //step V: constructing the graph
//   // // DirectedGraph g;

//   // for(size_t i=0; i < VertexDon.size(); i++)
//   // {
//   //   if(VertexDon[i] == VertexRec[i])
//       // std::cout << "BITE" << std::endl; // I am pissed off
//   //   // boost::add_edge (VertexDon[i], VertexRec[i], g);
//   // }

//   // std::vector<std::vector<int> > temptrec(VertexDon.size());
//   // for(size_t i =0; i< VertexRec.size(); i++)
//   // {
//   //   // if(VertexDon[i] == VertexRec[i])
//   //   //   std::cout << "WARNING:;:NODE IS ITS RECEIVER" << std::endl;
//   //   int donnode = VertexDon[i];
//   //   int recnode = VertexRec[i];
//   //   temptrec[donnode].push_back(recnode);

//   //   if(donnode>=nint_element)
//   //   {
//   //     int ID = aliases2ID[donnode];
//   //     donnode = aliases2nodes[ID];
//   //   }
//   //   if(recnode>=nint_element)
//   //   {
//   //     int ID = aliases2ID[recnode];
//   //     recnode = aliases2nodes[ID];
//   //   }

//   // }

//   // debug_graph_rec = temptrec;
//   // DEBUG_node_to_aliases = node2aliases;

//   // end = std::chrono::steady_clock::now();
//   // dat_time = end - start;
//   // std::cout<< "TIMER_NODEGRAPH_C5::" << dat_time.count() << std::endl;
//   // start = std::chrono::steady_clock::now();


//   // std::cout << "THIS WILL TAKE A LOT OF TIME, CHECKING CYCLICITY SYSTEMATICALLY" << std::endl;
//   // for (auto node : VertexDon)
//   // {
//   //   // std::cout << "processing::" << node << std::endl;
//   //   std::queue<int> related_nodes;
//   //   related_nodes.push(node);
//   //   std::vector<int> is_in_queue(nuint_element,false);
//   //   is_in_queue[node] == true;
//   //   while(related_nodes.size() > 0)
//   //   {
//   //     // std::cout << "size = " << related_nodes.size() << std::endl;
//   //     int this_node = related_nodes.front();
     
//   //     related_nodes.pop();
//   //     std::vector<int> recs = temptrec[this_node];
//   //     // std::vector<int> recs = this->get_MF_receivers_at_node(this_node);

//   //     for(auto rec:recs)
//   //     {
//   //       if(rec == node && this_node != node)
//   //       {
//   //         std::cout << "CYCLE DETECTED O: " << node << " D: " << this_node << " R: " << rec  << std::endl << "BASINS::";
//   //         for(auto bas:basin_multi_label[node])
//   //           std::cout << bas << "||";
//   //         std::cout << std::endl;
//   //       }

//   //       if(rec < 0 || rec == this_node || is_in_queue[rec] == true)
//   //         continue;

//   //       related_nodes.push(rec);
//   //       is_in_queue[rec] = true;

//   //       // if(rec == node)
//   //       //   std::cout << "CYCLE DETECTED" << std::endl;
//   //     }
//   //   }
//   // }
  

//   // // boost::add_edge (0, 1, 1, g);
//   // // boost::add_edge (0, 2, 1, g);
//   // // boost::add_edge (0, 3, 1, g);
//   // // boost::add_edge (2, 3, 1, g);
//   // // boost::add_edge (2, 1, 1, g);
//   // // boost::add_edge (1, 3, 1, g);
//   // // boost::add_edge (3, 4, 1, g);
//   // // std::cout << "VertexDon::" << VertexDon.size() << std::endl;

//   // // step VI: topological sort to get teh stack
//   // std::vector<Vertex> new_MF_stack;
//   // // boost::topological_sort(g, std::back_inserter(new_MF_stack));
//   // // std::cout << new_MF_stack.size() << "<<<<<<<" << std::endl;

//   // // for(int i = int(new_MF_stack.size())-1; i>=0; i--)
//   // // {
//   // //   std::cout << new_MF_stack[i] << std::endl;
//   // // }
//   // end = std::chrono::steady_clock::now();
//   // dat_time = end - start;
//   // std::cout<< "TIMER_NODEGRAPH_C6::" << dat_time.count() << std::endl;
//   // start = std::chrono::steady_clock::now();
  
//   // // return;

//   // // step VII: reimplementing the final stack
//   // std::vector<bool> is_in_stack(nuint_element,false);
//   // int incr = 0;
//   // for(int i = new_MF_stack.size()-1; i>=0; i--)
//   // {
//   //   int j = i - new_MF_stack.size()-1;
//   //   // std::cout << i << std::endl;
//   //   int next_node = new_MF_stack[i];

//   //   if(is_link_node[next_node])
//   //     continue;
//   //   // std::cout << next_node << "||" ;
    

//   //   if(next_node>=nint_element)
//   //   {
//   //     // then is an alias and need to recover normal node index
//   //     int ID = aliases2ID[next_node];
//   //     next_node = aliases2nodes[ID];

//   //   }
//   //   // std::cout <<next_node << std::endl;
//   //   // if is already processed, then I do not add
//   //   if(is_in_stack[next_node])
//   //     continue;

//   //   // Else add and consider processed
//   //   this->MF_stack[incr] = next_node;
//   //   incr++;
//   //   is_in_stack[next_node] = true;

//   // }

//   // end = std::chrono::steady_clock::now();
//   // dat_time = end - start;
//   // std::cout<< "TIMER_NODEGRAPH_C7::" << dat_time.count() << std::endl;
//   // start = std::chrono::steady_clock::now();


//   // basin_multi_label.shrink_to_fit();
//   // debug_baslab = basin_multi_label;
//   // debug_baslab.shrink_to_fit();

//   return;



//   return;



// }

// void NodeGraph::recursive_progapagate_label(int node, int label, std::vector<bool>& is_processed, std::vector<std::vector<int> >& labelz)
// {
//   is_processed[node] = true;
//   labelz[node].push_back(label);
//   std::vector<int> donodes = this->get_MF_donors_at_node(node);
//   for(auto nnode:donodes)
//   {
//     if(nnode<0 || is_processed[nnode])
//       continue;
//     this->recursive_progapagate_label(nnode,label, is_processed, labelz);
//   }
// }

// void NodeGraph::update_receivers_at_node(int node, std::vector<int>& new_receivers)
// {
//   for(size_t i = 0; i<8; i++)
//   {
//     if(i>=new_receivers.size())
//       this->MF_receivers(node,i) = -1;
//     else
//       this->MF_receivers(node,i) = new_receivers[i];
//   }
  
// }










// xt::pytensor<int,1> NodeGraph::get_all_nodes_in_depression()
// {
// //      _                               _           _ 
// //     | |                             | |         | |
// //   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
// //  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
// // | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
// //  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
// //            | |                                     
// //            |_|                                    

//   xt::pytensor<int,1> output = xt::zeros<int>({size_t(NROWS*NCOLS)});

//   for(auto vec:pits_pixels)
//   {
//     for(auto node:vec)
//       output[node]=1;
//   }
//   return output;
// }



// void NodeGraph::calculate_inherited_water_from_previous_lakes(xt::pytensor<double,1>& previous_lake_depth, xt::pytensor<int,1>& post_rec)
// {
//      // _                               _           _ 
// //     | |                             | |         | |
// //   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
// //  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
// // | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
// //  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
// //            | |                                     
// //            |_|                                    

//   // Iterating through the inverse stack
//   std::vector<bool> is_processed(post_rec.size(),false);
//   for(size_t i = 0; i<post_rec.size();i++)
//   {
//     // getting current node ID
//     int this_node = this->MF_stack[i];
//     double this_lake_depth = previous_lake_depth[this_node];

//     // Checking if there is any excess of water 
//     if(this_lake_depth == 0)
//     {
//       is_processed[this_node] = true;
//       continue;
//     }

//     if(is_processed[this_node])
//       continue;
    
//     is_processed[this_node] = true;
//     // alright, I have unprocessed excess of water there
//     double excess_volume = this_lake_depth * XRES * YRES;

//     // does it fall into an existing pit
//     int this_pit_ID = this->pits_ID[this_node];
//     int outlet_node = -9999; // triggering segfault if this happens not to be modified. #ExtremeDebugging
//     if(this_pit_ID>=0)
//     {
//       // Yes, I am just adding the excess water to the pit outlet
//       this->pits_inherited_water_volume[this_pit_ID] += excess_volume;
//       continue;
//     }
//     else
//     {
//       outlet_node = post_rec[this_node];
//     }
    
//     if(this->has_excess_water_from_lake[outlet_node])
//       excess_volume += node_to_excess_of_water[outlet_node];

//     this->node_to_excess_of_water[outlet_node] = excess_volume;
//     this->has_excess_water_from_lake[outlet_node] = true;

//   }

// }





// // This function preprocesses the stack after coorection by Cordonnier et al., 2019. It keeps the order but "unroute" the pits so that the depression solving is not affected
// std::vector<xt::pytensor<int,1> > preprocess_stack(xt::pytensor<int,1>& pre_stack, xt::pytensor<int,1>& pre_rec, xt::pytensor<int,1>& post_stack, xt::pytensor<int,1>& post_rec)
// { 
// // This function was my test #2: I was preprocessng the single flow stack before calculating the multiple stack in the hope to get it right.
// // It crashes the fortran code because it enforced some circularity in the MF stack, which would be quite convoluted to get rid of compared to the new version of the code 
// //      _                               _           _ 
// //     | |                             | |         | |
// //   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
// //  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
// // | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
// //  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
// //            | |                                     
// //            |_|                                    
//   // I'll need basin labels
//   xt::pytensor<int,1> baslab = xt::zeros<int>({pre_stack.size()});
//   // I am keeping tracks of the base levels
//   std::vector<int> base_levels;base_levels.reserve(pre_stack.size());
//   std::vector<int> base_levels_index;base_levels_index.reserve(pre_stack.size());
//   // I will return which pits have been rerouted and needs to be processes
//   xt::pytensor<int,1> rerouted_pits = xt::zeros<bool>({pre_stack.size()});

//   // First I am labelling the basins
//   int this_label = 0;
//   for(int i = pre_stack.size() -1 ; i>=0; i--)
//   {
//     int this_node = pre_stack[i];
//     int this_receiver = pre_rec[this_node];
//     baslab[this_node] = this_label;
//     // if base-level in the pre_stack (pre correction) then different basin
//     if(this_node == this_receiver)
//     {
//       if(this_node == 2075)
//         std::cout << "flub: "<< std::endl;
//       base_levels.emplace_back(this_node);
//       base_levels_index.emplace_back(i);
//       this_label++;
//     }
//   }
//   // cleaning space
//   base_levels.shrink_to_fit();
//   base_levels_index.shrink_to_fit();

//   // Now I am back calculating the corrections on the stack while keeping the right order in the basin calculations
//   for(size_t i=0; i<base_levels.size(); i++)
//   {   
//     int node = base_levels[i];
//     int index = base_levels_index[i];
//     if(node == 2075)
//         // std::cout << "bulf: " << std::endl;
//     if(pre_rec[node] == post_rec[node])
//     {
//       // If the receiver has never been corrected -> this is a true base level and fluxes can escape
//       rerouted_pits[node] = -1;
//       continue;
//     }
//     if(node == 2075)
//         // std::cout << "bulf2 " << std::endl;

//     // For each correction I gather the node to correct and their receivers
//     std::vector<int> nodes_to_correct;
//     std::vector<int> rec_to_correct;

//     // Dealing with current node first
//     int this_node = node;
//     int this_rec = post_rec[node];

//     // While I am (i) still in the same basin (ii) the node has been corrected and (iii) I am not at a base-level, I keep on gathering nodes
//     // Note that if I reach a "true" base level it means my depression is outleting outside of the model
//     while(this_rec != pre_rec[this_node] && this_node != this_rec)
//     {
//       if(node == 2075)
//         // std::cout << "bulf3 " << std::endl;
//       // Above conditions are met: this node will be to be corrected
//       nodes_to_correct.push_back(this_node);
//       rec_to_correct.push_back(this_rec);
//       // preparing the next one
//       this_node = this_rec;
//       this_rec = post_rec[this_node];
//       if(baslab[this_rec] != baslab[this_node])
//       {
//         nodes_to_correct.push_back(this_node);
//         rec_to_correct.push_back(this_rec);
//         break;
//       }
//     }

//     // I have supposidely gathered everything. Just checking I have at least one node in the thingy
//     if(nodes_to_correct.size()>0)
//     {
//       // The pit bottom is rerouted to itself -> CORRECTION: to ensure the mstack to keep the basin order, I think i need to make the receiver
//       int new_receveir_for_base_level = post_rec[nodes_to_correct[nodes_to_correct.size()-1]];
//       while(pre_rec[post_rec[new_receveir_for_base_level]] != pre_rec[new_receveir_for_base_level])
//         new_receveir_for_base_level = post_rec[new_receveir_for_base_level];

//       post_rec[nodes_to_correct[0]] = new_receveir_for_base_level;


//       // For each node, I correct their receiver to the presious one to reroute them to the centre of the pit
//       for(size_t i=1; i< nodes_to_correct.size(); i++)
//       {
//         post_rec[nodes_to_correct[i]] = pre_rec[nodes_to_correct[i]];
//       }

//       // I need to correct the stack now
//       for(int j=0; j<nodes_to_correct.size();j++)
//       {
//         post_stack[index + j] = nodes_to_correct[nodes_to_correct.size() - 1 - j];
//       }

//       // I mark this node has to be processed
//       rerouted_pits[nodes_to_correct[0]] = 1;
//     }
//     // Done with this base level
//   }

//   // For some reason, this function did not modify my arrays in place, idk why... So I return the results instead which could be otpimised in the future to avoid copy
//   std::vector<xt::pytensor<int,1> > output;output.reserve(3);
//   output.emplace_back(post_rec);
//   output.emplace_back(post_stack);
//   output.emplace_back(rerouted_pits);

//   // returning output
//   return output;
// }



// // This function correct the multiple flow receivers and donrs which end up with some duplicates probably linked to my homemade corrections.
// // it simply removes duplicate and inverse the corrected receivers, into the donors.
// void NodeGraph::initial_correction_of_MF_receivers_and_donors(xt::pytensor<int,1>& post_stack, xt::pytensor<int,2>& tMF_rec, xt::pytensor<int,2>& tMF_don, xt::pytensor<double,1>& elevation)
// {
//   for (size_t i =0; i< post_stack.size(); i++)
//   {
//     std::set<int> setofstuff;
//     for(size_t j=0; j<8;j++)
//     {
//       if(tMF_rec(i,j) < 0)
//         continue;

//       const bool is_in = setofstuff.find(tMF_rec(i,j)) != setofstuff.end();
//       if(is_in)
//         tMF_rec(i,j) = -1;
//       else if ( elevation[i]<elevation[tMF_rec(i,j)])
//         tMF_rec(i,j) = -1;
//       else
//         setofstuff.insert(tMF_rec(i,j));
//     }
//   }  

//   xt::pytensor<int,1> ndon = xt::zeros<int>({post_stack.size()});
//   for (size_t i =0; i< post_stack.size(); i++)
//   {
//     for(size_t j=0; j<8;j++)
//     {
//       int this_rec = tMF_rec(i,j);
//       if(this_rec<0)
//         continue;
//       tMF_don(this_rec,ndon[this_rec]) = int(i);
//       ndon[this_rec]++;
//     }
//   }

//   // And labelling as no data the remaining donors
//   for (size_t i =0; i< post_stack.size(); i++)
//   {
//     for(int j = ndon[i]; j<8; j++)
//     {
//       tMF_don(i,j) = -1;
//     }
//   }
// }


// // this functions labels the basins with multiple labels
// // THIS FUNCTION IS REALLY SLOW, NEED WORK
// void NodeGraph::label_basins_MF(std::vector<std::vector<int> >& MF_labels, std::vector<int>& all_base_levels, xt::pytensor<int,1>& post_rec)
// {
//     //Step II: label the basins from these nodes on the uncorrected multiple stack
//   for(auto node : all_base_levels)
//   {
//     std::vector<bool> is_noted(MF_labels.size(),false);
//     // this->recursive_progapagate_label(node, node,is_noted,basin_multi_label);
//     // Old try, less optimised
//     std::queue<int> nodes_to_label;
//     nodes_to_label.push(node);
//     // std::vector<bool> is_noted(nuint_element,false);
//     is_noted[node] = true;

//     while(nodes_to_label.empty() == false)
//     {
//       int this_node = nodes_to_label.front();
//       // std::cout << this_node << std::endl;
//       MF_labels[this_node].push_back(node);
//       nodes_to_label.pop();
//       std::vector<int> neighbors = this->get_MF_donors_at_node(this_node);
//       for(auto nenode:neighbors)
//       {
//         if(nenode<0 || nenode == this_node || is_noted[nenode])
//         {
//           continue;
//         }
//         nodes_to_label.push(nenode);
//         is_noted[nenode] = true;

//       }
//     }
//   }

//   std::vector<int> is_basin_processed(post_rec.size(),false);


//   // Getting the link between basins
//   is_link_node = std::vector<bool>(post_rec.size(), false);
//   for(auto node:all_base_levels)
//   {
//     if(is_basin_processed[node])
//       continue;

//     int this_node = node;
//     while(true)
//     {
//       // is_basin_processed[this_node] = true;
//       if(MF_labels[this_node].size()>1 && std::find(MF_labels[this_node].begin(),MF_labels[this_node].end(),node) != MF_labels[this_node].end() )
//       {
//         is_link_node[this_node] = true;
//       }
//       if(this_node == post_rec[this_node])
//         break;
//       this_node = post_rec[this_node];
//       if(std::find(all_base_levels.begin(),all_base_levels.end(),this_node) != all_base_levels.end())
//         break;
//     }
//   }

// }

// void NodeGraph::generate_vector_of_adjacency_unique_basin(std::vector<std::vector<int> >& MF_labels, std::vector<int>& VertexDon, std::vector<int>& VertexRec, std::vector<double>& VertexLength, 
//   std::vector<bool>& has_aliases, std::unordered_map<int,std::vector<int> >& node2aliases, std::vector<int>& aliases2nodes, std::unordered_map<int,int>& aliases2ID, 
//   std::vector<std::vector<int> >& aliases_rec, std::vector<std::vector<int> >& aliases_length, std::vector<int>& aliases_basin_recs)
// {










//   //TODO AFTER LUNCH::ADD aliases to MFLABELS VECTOR TO BEING ABLE TO CHECK IN THE LINKING!!!!!!










//   // Initialising the vectors to a maximum size
//   VertexDon = std::vector<int>();VertexDon.reserve(this->MF_stack.size() * 8);
//   VertexRec = std::vector<int>();VertexRec.reserve(this->MF_stack.size() * 8);
//   VertexLength = std::vector<double>();VertexLength.reserve(this->MF_stack.size() * 8);

//   // none of my nodes have aliases so far
//   has_aliases = std::vector<bool>(this->MF_stack.size(),false);
//   std::vector<bool> is_processed(this->MF_stack.size(),false);

//   int alincementor = -1;

//   int total_size = int(this->MF_stack.size());
//   // Iterating through all the nodes and their donors/receivers
//   for(size_t i=0; i < this->MF_stack.size(); i++)
//   {
//     int node = int(i);
//     if(is_processed[node])
//       continue;
//     // checking if the node belongs to multiple basins
//     if(is_link_node[node] == false)
//     {
//       // Simple scenario:
//       // Adding the receivers in the vector ready for directed graph
//       std::vector<int> recnodes = this->get_MF_receivers_at_node(node);
//       for(size_t j=0;j<recnodes.size();j++)
//       {
//         int this_node = recnodes[j];
//         if(this_node<0)
//           continue;
//         if(this_node == node)
//         {
//           is_processed[node] = true;
//           break;
//         }
//         VertexDon.emplace_back(node);
//         VertexRec.emplace_back(this_node);
//         VertexLength.emplace_back(this->MF_lengths(node,j));
//       }
//       is_processed[node] = true;
//     }
//     else
//     {
//       // Preparing the aliases, preprocessing is not processing
//       // gathering the different nodes
//       has_aliases[node] = true;
//       // Getting the different basins crossed by this t
//       auto target_basins = MF_labels[node];
//       // Gathering each aliases preparations: I am first just assigning which part of the node goes where
//       std::vector<int> this_node2aliases;
//       for(auto basin:target_basins)
//       {
//         // std::cout << basin << "<<";
//         alincementor++;
//         int this_alias = alincementor + total_size;
//         aliases2ID[this_alias] = alincementor;
//         aliases2nodes.push_back(node);
//         this_node2aliases.push_back(this_alias);
//         aliases_basin_recs.push_back(basin);
//         MF_labels.push_back({basin});
//       }
//       // std::cout << std::endl;
//       node2aliases[node] = this_node2aliases;

//       // // this last step consists in not lossing this node in the graph: I am linking it arbitrarily to its first alias, otherwise the node becomes independent and appear weirdly-early in the stack
//       VertexDon.emplace_back(node);
//       VertexRec.emplace_back(this_node2aliases[0]);
//       VertexLength.emplace_back(1);

//       auto donnodes = this->get_MF_donors_at_node(node);
//       for(auto don:donnodes)
//       {
//         if(don<0 || don == node)
//           continue;
//         else
//         {
//           VertexDon.emplace_back(don);
//           VertexRec.emplace_back(node);
//           VertexLength.emplace_back(1);
//           for (auto tal:this_node2aliases)
//           {
//             VertexDon.emplace_back(don);
//             VertexRec.emplace_back(tal);
//             VertexLength.emplace_back(1);

//           }
//         }

//       }

//       // std::cout << node << std::endl;

//     }
//   }
//   // std::cout << "GURG::" << MF_labels[10343][0] << std::endl;


//   // return;
  

//   // Second iteration, that time only on node bearing aliases, which should be a small part of all nodes
//   for(size_t i=0; i < this->MF_stack.size(); i++)
//   {
//     int node = int(i);
//     if(is_processed[node])
//       continue;
//     // If not processed yet, it is aliased
//     auto target_basins = MF_labels[node];
//     std::vector<int> recnodes = this->get_MF_receivers_at_node(node);
//     std::vector<double> these_length = this->get_MF_lengths_at_node(node);

//     for(size_t j=0; j<node2aliases[node].size(); j++)
//     {
//       int this_node = node2aliases[node][j];
//       int ID = aliases2ID[this_node];
//       int this_basin = aliases_basin_recs[ID];
//       // if(this_node == 250010)
//       //   std::cout << "NODE 250010::belongsto::" << this_basin << "::";
//       for(size_t k=0; k< 8; k++)
//       {
//         int recnode = recnodes[k];
//         if(recnode<0)
//           continue;
//         // if(this_node == 250010)
//         //   std::cout << "has_neighbors::";
//         if(has_aliases[recnode] ==  false && std::find(MF_labels[recnode].begin(),MF_labels[recnode].end(),this_basin) != MF_labels[recnode].end() ) // this_basin == MF_labels[recnode][0]) //  it should only have one basin label if it has no alias
//         {
//           // if(this_node == 250010)
//           //   std::cout << "notalias::" << recnode;
//           VertexDon.emplace_back(this_node);
//           VertexRec.emplace_back(recnode);
//           VertexLength.emplace_back(these_length[k]);

//         }
//         else
//         {
//           // if(this_node == 250010)
//           //   std::cout << "neighalias::" << recnode;
//           for(size_t l =0; l< node2aliases[recnode].size(); l++)
//           {
//             int tested_alias = node2aliases[recnode][l];
//             int this_ID = aliases2ID[tested_alias];
//             if(this_basin != aliases_basin_recs[this_ID])
//               continue;
//             VertexDon.emplace_back(this_node);
//             VertexRec.emplace_back(tested_alias);
//             VertexLength.emplace_back(these_length[k]);

//           }
//         }
//       }
//       // std::cout << std::endl;
//     }
//   }




// }


// void NodeGraph::link_pit_vertex_to_receivers_or_their_aliases(xt::pytensor<int,1>& post_rec,std::vector<int>& all_base_levels_ordered, std::vector<std::vector<int> >& basin_multi_label,
// std::vector<bool>& has_aliases, std::vector<int>& VertexDon, std::vector<int>& VertexRec, std::vector<double>& VertexLength, std::unordered_map<int,int>& aliases2ID , std::vector<int>&aliases_basin_recs, 
// std::unordered_map<int,std::vector<int> > node2aliases, std::vector<int>& aliases2nodes)//, xt::pytensor<double,1>& elevation)
// {

//   // TODO HERE: ADD A HIERARCHY OF BASINS  RATHER THAN ONLY CHECKING IF THE BASIN HAS BEEN PROCESSED!
//   // CURRENT PROBLEM: EACH BASIN IS COUNTED AS PROCESSED ONCE BUT THEY CAN RECEIVE MULTIPLE BASINS SO I JUST NEED TO CHECK IF THEY ARE AFTER IN THE TOP TO DOWN TREE

//   // keeping a map of already processed basin, I do not want to stop my flow jsut because it recrosses an already processed basin
//   std::map<int,int> basin_order;
//   std::map<int,int> basin_family;
//   // std::map<int,bool> is_mother_basin;

//   // Filling the map for each basin to preprocessed
//   int incrementor_basin_order =0;
//   int incrementor_basin_family =0;
//   for(auto bas:all_base_levels_ordered)
//   {
//     basin_order[bas] = incrementor_basin_order;
//     incrementor_basin_order++;
//     int next_node = bas;
//     while(true)
//     {
//       next_node = post_rec[next_node];

//       if(next_node == post_rec[next_node])
//       {
//         basin_family[bas] = next_node;
//         break;
//       }
//     }

//   }

//   // for(size_t i=0; i< all_base_levels_ordered.size(); i++)
//   // {
//   //   int this_basin = all_base_levels_ordered[i];
//   //   basin_family[this_basin] = incrementor_basin_family;
//   //   int target;
//   //   if(i < all_base_levels_ordered.size()-1)
//   //     target = all_base_levels_ordered[i+1];
//   //   else
//   //     continue;

//   //   int next_node = this_basin;
//   //   while(true)
//   //   {
//   //     next_node = post_rec[next_node];
//   //     if(next_node == target)
//   //       break;
//   //     if(next_node == post_rec[next_node])
//   //     {
//   //       incrementor_basin_family++;
//   //       break;
//   //     }
//   //   }
//   // }

//   // Now iterating through all base levels
//   for(auto node:all_base_levels_ordered)
//   {
//     int next_node = node;
//     // preparing the rereouting
//     //# switch to stop the reordering when an outlet is found
//     bool keep_on_searching = true;
//     bool is_BL = false;
//     int out_basin = -9999;
//     while(keep_on_searching)
//     {
//       // My next node is the corrected receiver
//       next_node = post_rec[next_node];
//       // If it has not been rerouted by Cordonnier et al 2017, ignore that node
//       if(next_node == post_rec[next_node])
//       {
//         keep_on_searching = false;
//         is_BL = true;
//         break;
//       }
//       // Else I have to check the receivers of the node
//       std::vector<int> these_basins = basin_multi_label[next_node];
//       for( auto gurg:these_basins)
//       {
//         if(gurg != node && basin_order[gurg] > basin_order[node] && basin_family[gurg] == basin_family[node])
//         {
//           // if there is any receiving basin different than the current one that has not been processed already, bingo I can stop the loop
//           keep_on_searching = false;
//           out_basin = gurg;
//           break;
//         }
//       }    

//     }

//     // if I am a base level, I ignore the next steps (no rerouting)
//     if(is_BL)
//     {
//       continue;
//     }

//     // if my first receiver has no aliases (i.e. no cyclicity coming back to the previous basin)
//     // I can then directly push everything into the 
//     if(has_aliases[next_node] == false)
//     {
//       VertexDon.emplace_back(node);
//       VertexRec.emplace_back(next_node);
//       VertexLength.emplace_back(XRES*10); // TODO REPLACE WITH EUCLIDIAN DISTANCE
//     }
//     else
//     {
//       // In this case the node has multiple aliases
//       // Finding the alisases first
//       int save_node = next_node;
//       std::vector<int> aliases = node2aliases[next_node];
//       for(auto al : aliases)
//       {
//         // find the internal ID in the alias vectors
//         int ID = aliases2ID[al];
//         // Getting the unique basin linked to the alias
//         int basID = aliases_basin_recs[ID];
//         // if the alias comes back to the same basin: nope but adding a link to the original node in order to keep it in the general graph and avoid isolation
//         if(basID == node)
//         {
//           VertexDon.emplace_back(save_node);
//           VertexRec.emplace_back(al);
//           VertexLength.emplace_back(1); 
//           continue;
//         }

//         // if the alias goes in an already processed basin: nope
//         if(basin_order[basID] <= basin_order[node] || basID != out_basin || basin_family[basID] != basin_family[node])
//         {
//           continue;
//         }

//         // if(node == 6221)
//         //   std::cout << "FALUF" << std::endl;

//         // else, I am keeping it and...
//         next_node = al;
//         // ... breaking the cycle
//         break;
//       }


//       // I should have founbd my outlet and adding it to the vertexes.
//       // std::cout << node << std::endl;
//       VertexDon.emplace_back(node);
//       VertexRec.emplace_back(next_node);
//       VertexLength.emplace_back(XRES*10); // TODO REPLACE WITH EUCLIDIAN DISTANCE
//     }
//     // basin_order[node] = true;/

//   }


// }






// //##############################

// // // saving a nearly working version here to work on a new one:
// // // This empty constructor is just there to have a default one.
// // void NodeGraph::create(xt::pytensor<int,1>& pre_stack,xt::pytensor<int,1>& pre_rec, xt::pytensor<int,1>& post_rec, xt::pytensor<int,1>& post_stack,
// //   xt::pytensor<int,1>& tMF_stack, xt::pytensor<int,2>& tMF_rec,xt::pytensor<int,2>& tMF_don, xt::pytensor<double,1>& elevation, xt::pytensor<double,2>& tMF_length,
// //   float XMIN, float XMAX, float YMIN, float YMAX, float XRES, float YRES, int NROWS, int NCOLS, float NODATAVALUE)
// // {

// //   // I am first correcting the donors using the receivers: the receivers seems alright but somehow my donors are buggy
// //   // This is simply done by inverting the receiver to the donors
// //   // Also extra step I need to remove the duplicate receivers
// //   for (size_t i =0; i< post_stack.size(); i++)
// //   {
// //     std::set<int> setofstuff;
// //     for(size_t j=0; j<8;j++)
// //     {
// //       const bool is_in = setofstuff.find(tMF_rec(i,j)) != setofstuff.end();
// //       if(is_in)
// //         tMF_rec(i,j) = -1;
// //       else
// //         setofstuff.insert(tMF_rec(i,j));
// //     }
// //   }  

// //   xt::pytensor<int,1> ndon = xt::zeros<int>({post_stack.size()});
// //   for (size_t i =0; i< post_stack.size(); i++)
// //   {
// //     for(size_t j=0; j<8;j++)
// //     {
// //       int this_rec = tMF_rec(i,j);
// //       if(this_rec<0)
// //         continue;
// //       tMF_don(this_rec,ndon[this_rec]) = int(i);
// //       ndon[this_rec]++;
// //     }
// //   }

// //   // And labelling as no data the remaining donors
// //   for (size_t i =0; i< post_stack.size(); i++)
// //   {
// //     for(int j = ndon[i]; j<8; j++)
// //     {
// //       tMF_don(i,j) = -1;
// //     }
// //   }
  
// //   // Inithalising general attributes
// //   this->NROWS = NROWS;
// //   this->NCOLS = NCOLS;
// //   this->XMIN = XMIN;
// //   this->XMAX = XMAX;
// //   this->YMIN = YMIN;
// //   this->YMAX = YMAX;
// //   this->XRES = XRES;
// //   this->YRES = YRES;
// //   this->NODATAVALUE = NODATAVALUE;
// //   this->MF_stack = tMF_stack;
// //   this->MF_receivers = tMF_rec;
// //   this->MF_lengths = tMF_length;
// //   this->MF_donors = tMF_don;

// //   int nint_element = int(this->MF_stack.size());
// //   size_t nuint_element = this->MF_stack.size();

// //   // Now I need to post-process the MF stack

// //   std::vector<std::vector<int> > basin_multi_label(nuint_element);
// //   std::vector<int> all_base_levels_nocorr, index_in_first_MFstack(nint_element);
// //   std::map<int,int> BL_to_index;

// //   pit_to_reroute  = std::vector<bool>(nuint_element,false);

// //   std::vector<std::vector<int> > separated_trees;
// //   std::map<int,int> BL_to_sep_tree;
// //   int index_st = 0;
// //   separated_trees.push_back({});


// //   // std::cout <<"URG" << std::endl;
// //   // Step I:
// //   // # Order my basin by order of preocessing from top to bottom in the corrected stack
// //   int incr2 = 0;
// //   for(int i = nint_element - 1; i>=0; i--)
// //   {
// //     int node = post_stack[i];
// //     if(pre_rec[node] == node)
// //     {
// //       all_base_levels_nocorr.push_back(node);
// //       separated_trees[index_st].push_back(node);
// //       BL_to_sep_tree[node] = index_st;
// //       BL_to_index[node] = incr2;
// //       incr2++;
    
// //       if(post_rec[node] != node)
// //       {
// //         pit_to_reroute[node] = true;
// //       }
// //       else
// //       {
// //         separated_trees.push_back({});
// //         index_st++;
// //       }
// //     }
// //     basin_multi_label.emplace_back(std::vector<int>());
// //   }
// //   // all_base_levels_nocorr now has all the baselevel nodes ordered by graph solving
// //   // std::cout <<"URG2" << std::endl;


// //   //Step II: label the basins from these nodes on the uncorrected multiple stack
// //   for(auto node : all_base_levels_nocorr)
// //   {
// //     std::vector<bool> is_noted(nuint_element,false);
// //     // this->recursive_progapagate_label(node, node,is_noted,basin_multi_label);
// //     // Old try, less optimised
// //     std::queue<int> nodes_to_label;
// //     nodes_to_label.push(node);
// //     // std::vector<bool> is_noted(nuint_element,false);
// //     is_noted[node] = true;

// //     while(nodes_to_label.empty() == false)
// //     {
// //       int this_node = nodes_to_label.front();
// //       // std::cout << this_node << std::endl;
// //       basin_multi_label[this_node].push_back(node);
// //       nodes_to_label.pop();
// //       std::vector<int> neighbors = this->get_MF_donors_at_node(this_node);
// //       for(auto nenode:neighbors)
// //       {
// //         if(nenode<0 || nenode == this_node || is_noted[nenode])
// //         {
// //           continue;
// //         }
// //         nodes_to_label.push(nenode);
// //         is_noted[nenode] = true;

// //       }
// //     }
// //   }
// //   // std::cout <<"URG3" << std::endl;

// //   for(int i=0; i< nint_element; i++)
// //   {
// //     index_in_first_MFstack[this->MF_stack[i]] = i;
// //   }

// //   xt::pytensor<int,1> new_MF_stack = xt::zeros<int>({nuint_element});

// //   std::vector<bool> is_processed(nuint_element,false);
// //   int incr = 0;
// //   int next_node = 0;

// //   // version 2:: Does not work so far...
// //   // while (incr<nint_element)
// //   // {
// //   //   int current_basin_tree_id = BL_to_sep_tree[basin_multi_label[this->MF_stack[next_node]][0]];
// //   //   std::cout << current_basin_tree_id << std::endl;
// //   //   std::vector<int>& current_BT = separated_trees[current_basin_tree_id];

// //   //   int last_index_start;
// //   //   for(auto BL:current_BT )
// //   //   {
      
// //   //     int index_start = index_in_first_MFstack[BL];
// //   //     if(is_processed[ this->MF_stack[index_start] ])
// //   //       std:: cout << "FATAERROR::2" << std::endl;
// //   //     int index_stop = index_in_first_MFstack[BL];
// //   //     for(int j = index_start; j>=0;j--)
// //   //     {
// //   //       int this_node = this->MF_stack[j];
// //   //       if(is_processed[this_node])
// //   //         break;
// //   //       index_stop = j;

// //   //     }

// //   //     for(int i = index_stop; i<=index_start; i++)
// //   //     {
// //   //       int this_node = this->MF_stack[i];
// //   //       new_MF_stack[incr] = this_node;
// //   //       incr++;
// //   //       is_processed[this_node] = true;
// //   //       if(incr>=nint_element)
// //   //         std::cout << "FATALERROR" << std::endl;
// //   //     }
// //   //     last_index_start = index_start;
// //   //   }


// //   //   if(incr<nint_element)
// //   //     next_node = this->MF_stack[last_index_start+1];
// //   // }


// //   // version 1, more secure but slower
// //   for(auto BL: all_base_levels_nocorr)
// //   {
// //     // std::cout << "BL:" << std::endl;
// //     int n_element_this_vector = 0;
// //     int index = index_in_first_MFstack[BL];
// //     // std::cout << BL << "||" << index << std::endl;
// //     std::vector<int> temp;temp.reserve(nuint_element);
// //     for(int j = index; j>=0;j--)
// //     {
// //       int this_node = this->MF_stack[j];

// //       if(is_processed[this_node])
// //         continue;

// //       std::vector<int>& baslabs = basin_multi_label[this_node];

// //       if(std::find(baslabs.begin(), baslabs.end(), BL) != baslabs.end())
// //       {
// //         is_processed[this_node] = true;
// //         temp.emplace_back(this_node);
// //         n_element_this_vector++;
// //       }
// //     }
// //     // temp.shrink_to_fit();
// //     // std::cout << temp.size() << std::endl;


// //     for(int i = int(n_element_this_vector-1); i>=0;i--)
// //     // for(size_t i=0; i<temp.size(); i++)
// //     {
// //       int this_node = temp[i];
// //       // std::cout << this_node << std::endl;
// //       new_MF_stack[incr] = this_node;
// //       incr++;
// //     }
// //   }

// //   // std::cout <<"URG4" << std::endl;

// //   this->MF_stack =  new_MF_stack;
// //   // std::cout <<"URG5" << std::endl;

// //   return;


  
// // The following part of this function is
// //      _                               _           _ 
// //     | |                             | |         | |
// //   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
// //  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
// // | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
// //  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
// //            | |                                     
// //            |_|                                      
// // I keep it just in case




//   //  // I need the pre-accumulation vector to calculate the nodes draining to a certain pit
//   // xt::pytensor<int,1> pre_contributing_pixels = xt::zeros<int>({pre_stack.size()});
//   // // Initialising my basin array to 0
//   // basin_label = xt::zeros<int>({pre_stack.size()});
//   // pit_to_reroute = std::vector<bool>(pre_stack.size(), false);

//   // // labeling my pits
//   // int label = 0;
//   // for(int i=int(pre_stack.size()-1); i>=0; i--)
//   // {
//   //   // Current node and its receiver
//   //   int this_node = pre_stack[i];
//   //   int this_rec = pre_rec[this_node];
//   //   // Labelling the basin
//   //   basin_label[this_node] = label;
//   //   // If I reach a base level, my node is its own receiver by convention (Braun et Willett, 2013)
//   //   if(this_rec != this_node)
//   //   {
//   //     // Adding the accumulation
//   //     pre_contributing_pixels[this_rec] += pre_contributing_pixels[this_node]+1 ;
//   //   }
//   //   else
//   //   {
//   //     // Incrementing the label
//   //     label++;
//   //     if(pre_rec[this_node] != post_rec[ this_node])
//   //     {
//   //       // labelling this node to be rerouted. I am differentiating pits from model outlet that way
//   //       pit_to_reroute[this_node] = true;
//   //     }
//   //   }
//   // }


//   // // Option1:
//   // // Now ordering my pits by how last they appear in the reverse stack
//   // // The idea is to process the original pits by order of appearance in the reverse stack
//   // std::vector<int> order_basin(label+1,-1);
//   // int this_order = -1;
//   // for(int i= 0; i< int(pre_stack.size()); i++)
//   // {
//   //   int this_node = post_stack[i];
//   //   int this_label = basin_label[this_node];
//   //   if(order_basin[this_label] == -1)
//   //   {
//   //     this_order++;
//   //     order_basin[this_label] = -1*this_order;
//   //   }
//   // }
//   // for(int i= 0; i< label+1; i++)
//   // {    
//   //   order_basin[i] += this_order;
//   // }

//   // // now reordering the multiple-flow stack by order of base-level processing
//   // // I am assigning a "basin order" to each node by the minimum basin order of all his receiver
//   // // then storing the node in temporary local stacks linked to each base levels, but still in the order of 
//   // std::vector<std::vector<int> > new_MF_stack(this_order+1);
//   // for(size_t i=0; i< new_MF_stack.size(); i++)
//   //   new_MF_stack[i] = {};
//   // // Iterating through all nodes in the MF stack and applying the change
//   // for(auto node: this->MF_stack)
//   // {
//   //   std::vector<int> rec = this->get_MF_receivers_at_node(node);
//   //   // std::vector<int> rec = this->get_MF_donors_at_node(node);
//   //   // Initialising the minimum order to the maximum +1
//   //   int min_order = this_order + 1;
//   //   for(auto recnode:rec)
//   //   {
//   //     // if receiver is valid
//   //     if(recnode>=0)
//   //     {
//   //       // if this order is lower than the current one, I save it
//   //       if(order_basin[basin_label[recnode]]<min_order)
//   //         min_order = order_basin[basin_label[recnode]];
//   //     }
//   //   }
//   //   // This happens whan I am a base-level, hence my basin order is the one of my own node
//   //   if(min_order == this_order + 1)
//   //     min_order = order_basin[basin_label[node]];
//   //   // Assigning that node to the temporary stack
//   //   new_MF_stack[min_order].push_back(node);
//   // }

//   // // now recreating the MFstack with the correct order
//   // int incr = 0;
//   // for(int i =0; i < this_order; i++)
//   // {
//   //   std::vector<int>& this_sub_stack = new_MF_stack[i];
//   //   for (auto node : this_sub_stack)
//   //   {
//   //     this->MF_stack[incr] = node;
//   //     incr++;
//   //   }
//   // }


//   // Option2: Well I need to find one


  
// // The following part of this function was a first test preprocessing depressions before solving them
// // It failed to capture the essences of subdepressions correctly so I changed the method to something more universal
// // I am keeping it  in case I come back to it later
// //      _                               _           _ 
// //     | |                             | |         | |
// //   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
// //  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
// // | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
// //  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
// //            | |                                     
// //            |_|                                      


// //   // labelisation for pit ID, starting at -1 as I increment prior to pushing back
// //   int this_pit_ID = -1;
// //   // this vector will contain -1 if the node is not in a pit and pitID if it is
// //   pits_ID = std::vector<int>(pre_stack.size(),-1);

 

// //   // First I need the accumulation vector of the prestack,
// //   // I also labelise the basins as it will be needed for the regroupping
// //   // To do so I iterate through the stack backward
// //   // int label = 0;
// //   for(int i=int(pre_stack.size()-1); i>=0; i--)
// //   {
// //     // Current node and its receiver
// //     int this_node = pre_stack[i];
// //     int this_rec = pre_rec[this_node];
// //     // Labelling the basin
// //     basin_label[this_node] = label;
// //     // If I reach a base level, my node is its own receiver by convention (Braun et Willett, 2013)
// //     if(this_rec != this_node)
// //     {
// //       // Adding the accumulation
// //       pre_contributing_pixels[this_rec] += pre_contributing_pixels[this_node]+1 ;
// //     }
// //     else
// //     {
// //       // Incrementing the label
// //       label++;
// //     }
// //   }
// //   // Done with the pre labelling

// //   // Now I need some sort of hiererchy in my basins, with the corrected path.
// //   std::vector<int> score_basin(label + 1, -1);

// //   int score_incrementor = 0;
// //   for(auto node: this->MF_stack)
// //   {
// //     int this_label = basin_label[node];
// //     if(node == pre_rec[node])
// //     {
// //       score_basin[this_label] = score_incrementor;
// //       score_incrementor++;
// //     }

// //   }



// //   // First step is to register the pits before correction by Cordonnier et al., 2019
// //   // I am therefore detecting where there are internal base levels 
// //   for(size_t i=0; i< pre_stack.size(); i++)
// //   {
// //     // Getting current node and its receiver pre/post correction
// //     int this_node = pre_stack[i];
// //     int this_receiver = pre_rec[this_node];
// //     int tpost_rec = post_rec[this_node];

// //     // Checking if it is a pit, which is equivalent to checking if:
// //     // my node is draining to itself pre-correction but not post-correction
// //     // If still draining to itself post-correction, it is a model base level, i.e. an outlet 
// //     if(this_node == this_receiver && this_receiver != tpost_rec)
// //     {
// //       // Right, I am at the bottom of a depression, a pit
// //       // Incrementing the pit ID
// //       this_pit_ID++;

// //       // Registering it
// //       this->pits_ID[this_node] = this_pit_ID;
// //       // I know that I will save the deposition/erosion flux at that node ID to back correct it if necessary
// //       this->register_deposition_flux[this_node] = 0;
// //       this->register_erosion_flux[this_node] = 0;

// //       // The bottom of the pit is this node POTENTIAL OPTIMISATION HERE: PREDEFINE A NUMBER OF NODES IN THE PIT
// //       pits_bottom.push_back(this_node);
// //       // initialising the number of pixels to 1
// //       pits_npix.push_back(1);

// //       // I need to find its outlet here, so I will follow the receiving correction until I fall in a node in a different basin
// //       int this_pit_outlet = post_rec[this_node]; // starting at this node 
// //       int this_basin_label = basin_label[this_node]; // saving the label
// //       int label_receiver = basin_label[post_rec[this_pit_outlet]]; // and the receiving label
// //       int elevation_node = this_pit_outlet; // the elevation of the outlet will be the highest of the two nodes at the outlet
// //       // Iterating until I either find a new basin or an model base-level (depression splilling water out of the model)
// //       while(this_basin_label == label_receiver && this_pit_outlet != post_rec[this_pit_outlet] && score_basin[this_basin_label] >= score_basin[label_receiver])
// //       {
// //         elevation_node = this_pit_outlet;
// //         this_pit_outlet = post_rec[this_pit_outlet];
// //         this_basin_label = basin_label[this_pit_outlet];
// //         label_receiver = basin_label[post_rec[this_pit_outlet]];
// //       }

// //       // Checking which of the past two nodes have the same elevation
// //       if(elevation[elevation_node] < elevation[this_pit_outlet])
// //         elevation_node = this_pit_outlet;

// //       // saving the outlet
// //       this->pits_outlet.push_back(this_pit_outlet);

// //       // Quack
// //       //   __
// //       // <(o )___
// //       //  ( ._> /
// //       //   `---'  
// //       // Quack

// //       // Saving the elevation iof this local outlet
// //       double outlet_elevation = elevation[elevation_node];

// //       // initialising the volume of the pit to zero
// //       this->pits_volume.push_back(0);
// //       //Initialising the list ofpixels for each pits
// //       this->pits_pixels.push_back({});

// //       // Getting the basin label right
// //       this->pits_baslab.push_back(this->basin_label[this_node]);

// //       // Getting all the node draining into that pit and detecting which one are below the elevation
// //       for(size_t j = i; j <= i+pre_contributing_pixels[this_node]; j++)
// //       {
// //         // Waht is this node
// //         int tested_node = pre_stack[j];
// //         // If within the pit, I register it and add to the volume
// //         if(elevation[tested_node]<outlet_elevation)
// //         {
// //           pits_ID[tested_node] = this_pit_ID;
// //           register_deposition_flux[tested_node] = 0;
// //           register_erosion_flux[tested_node] = 0;
// //           pits_volume[this_pit_ID] += XRES*YRES*(outlet_elevation-elevation[tested_node]);
// //           pits_npix[this_pit_ID] += 1;
// //           pits_pixels[this_pit_ID].push_back(tested_node);
// //         }
// //       //Done with labelling that pit
// //       }
// //       // Available volume for sediment so far is the pit volume
// //       this->pits_available_volume_for_sediments.push_back(pits_volume[this_pit_ID]);

// //     // done with checking that node
// //     }
// //   // Done with labelling all pits
// //   }


// // //    quack quack  __
// // //             ___( o)>
// // //             \ <_. )
// // //    ~~~~~~~~  `---'  

// //   // Saving my preacc, I might need it for later (TO DELETE IF NOT)
// //   this->preacc = pre_contributing_pixels;

// //   // Now I need to deal with depression hierarchy
// //   // Something I may have forgot to consider in the v0.01 of this code: some depression are intricated
// //   // A simple solution is to go through the multiple stack and decide which depression is actually a subset of the new one
// //   // then when I'll be filling my depressions, I'll jsut have to keep in mind already existing lake depths

// //   // First initialising the vector to the right size
// //   this->sub_depressions = std::vector<std::vector<int> >(pits_npix.size());
// //   for (size_t i=0; i<pits_npix.size(); i++)
// //     this->sub_depressions[i] = {};

// //   // keeping track of which pit I processed
// //   std::vector<bool> is_pit_processed(pits_npix.size(),false);
// //   for(auto& node:this->MF_stack)
// //   {
// //     int tID = this->pits_ID[node];
// //     // Ignoring non pits
// //     if(tID <0)
// //       continue;
// //     // Ignoring already processed pits
// //     if(is_pit_processed[tID] == true)
// //       continue;

// //     // unprocessed pit
// //     // Its ID:
// //     int this_outlet = this->pits_outlet[tID];
// //     // Its outlet pit ID
// //     int oID = this->pits_ID[this_outlet];

// //     // My pit is processed
// //     is_pit_processed[tID] = true;

// //     // If my outlet is not a pit, the current pit is not a subset of another large pit
// //     if (oID < 0)
// //       continue;

// //     // else it is a subset of its donor pit
// //     this->sub_depressions[oID].push_back(tID);
// //   }

// //   // Done with the sub-depression routine

// //   // Initialising the argument for calculating inherited lake waters
// //   this->has_excess_water_from_lake = std::vector<bool>(pre_stack.size(),false);
// //   this->pits_inherited_water_volume = std::vector<double>(this_pit_ID+1, 0);

// // }




































// // // OLDER TESTS!!!


// // // This empty constructor is just there to have a default one.
// // void cppintail::create()
// // {
// // 	std::string yo = "I am an empty constructor yo!";

// // }

// // // Basic constructor
// // void cppintail::create(float tXMIN, float tXMAX, float tYMIN, float tYMAX, float tXRES, float tYRES, int tNROWS, int tNCOLS, float tNODATAVALUE)
// // {
// // 	// I think all of these are pretty explicit
// // 	XMIN = tXMIN;
// // 	XMAX = tXMAX;
// // 	YMIN = tYMIN;
// // 	YMAX = tYMAX;
// // 	XRES = tXRES;
// // 	YRES = tYRES;
// // 	NROWS = tNROWS;
// // 	NCOLS = tNCOLS;
// // 	NODATAVALUE = tNODATAVALUE;

  

// // }

// // //-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
// // // This function compute the flow direction
// // // -> DEM: numpy array
// // //-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
// // void cppintail::compute_neighbors(xt::pytensor<float,2>& DEM)
// // {
// //   // Initialising flowdir
// //   FLOWDIR = xt::zeros<int>({NROWS,NCOLS});

// //   // First processing the centre of the DEM
// //   // Avoiding the edgae to avoid having to test systematically if I am close to the edge
// //   for(size_t i = 1; i < NROWS - 1; i++)
// //   for(size_t j = 1; j < NCOLS - 1; j++)
// //   {
// //     if(DEM(i,j) == NODATAVALUE)
// //       continue;

// //     float this_elevation = DEM(i,j);
// //     if(DEM(i-1, j-1) < this_elevation)
// //       FLOWDIR(i,j) += 1;
// //     if(DEM(i-1, j) < this_elevation)
// //       FLOWDIR(i,j) += 10;
// //     if(DEM(i-1, j+1) < this_elevation)
// //       FLOWDIR(i,j) += 100;
// //     if(DEM(i, j+1) < this_elevation)
// //       FLOWDIR(i,j) += 1000;
// //     if(DEM(i+1, j+1) < this_elevation)
// //       FLOWDIR(i,j) += 10000;
// //     if(DEM(i+1, j) < this_elevation)
// //       FLOWDIR(i,j) += 100000;
// //     if(DEM(i+1, j-1) < this_elevation)
// //       FLOWDIR(i,j) += 1000000;
// //     if(DEM(i, j-1) < this_elevation)
// //       FLOWDIR(i,j) += 10000000;
// //   }
// //   std::cout << "DEBUG::done with core flowdir" << std::endl;

// //   // WNow I am looping through the edges, I can add many tests as there are much less nodes
// //   // 🦆
// //   for(size_t i = 0; i < NROWS ; i++)
// //   {
// //     // FIRST COLUMN
// //     size_t j = 0;
// //     float this_elevation = DEM(i,j);

// //     if(DEM(i,j) == NODATAVALUE)
// //       continue;

// //     if(i > 0)
// //     {
// //       if(DEM(i-1, j) < this_elevation)
// //         FLOWDIR(i,j) += 10;
// //       if(DEM(i-1, j+1) < this_elevation)
// //         FLOWDIR(i,j) += 100;
// //     }

// //     if(i < NROWS - 1)
// //     {
// //       if(DEM(i+1, j+1) < this_elevation)
// //         FLOWDIR(i,j) += 10000;
// //       if(DEM(i+1, j) < this_elevation)
// //         FLOWDIR(i,j) += 100000;
// //     }

// //     if(DEM(i, j+1) < this_elevation)
// //       FLOWDIR(i,j) += 1000;    


// //     // LAST column
// //     j = NCOLS - 1;
// //     this_elevation = DEM(i,j);

// //     if(DEM(i,j) == NODATAVALUE)
// //       continue;

// //     if(i > 0)
// //     {
// //       if(DEM(i-1, j-1) < this_elevation)
// //         FLOWDIR(i,j) += 1;
// //       if(DEM(i-1, j) < this_elevation)
// //         FLOWDIR(i,j) += 10;
// //     }
// //     if(i < NROWS - 1)
// //     {
// //       if(DEM(i+1, j) < this_elevation)
// //         FLOWDIR(i,j) += 100000;
// //       if(DEM(i+1, j-1) < this_elevation)
// //         FLOWDIR(i,j) += 1000000;
// //     }

// //     if(DEM(i, j-1) < this_elevation)
// //       FLOWDIR(i,j) += 10000000;
// //   }

// //   for(size_t j = 1; j < NCOLS - 1; j++)
// //   {

// //     size_t i = 0;
// //     float this_elevation = DEM(i,j);
    
// //     if(DEM(i,j) == NODATAVALUE)
// //       continue;

// //     if(j>0)
// //     {
// //       if(DEM(i+1, j-1) < this_elevation)
// //         FLOWDIR(i,j) += 1000000;
// //       if(DEM(i, j-1) < this_elevation)
// //         FLOWDIR(i,j) += 10000000;
// //     }

// //     if(j < NCOLS - 1)
// //     {
// //       if(DEM(i, j+1) < this_elevation)
// //         FLOWDIR(i,j) += 1000;
// //       if(DEM(i+1, j+1) < this_elevation)
// //         FLOWDIR(i,j) += 10000;
// //     }

// //     if(DEM(i+1, j) < this_elevation)
// //       FLOWDIR(i,j) += 100000;

// //     i = NROWS - 1;
// //     this_elevation = DEM(i,j);
// //     if(DEM(i,j) == NODATAVALUE)
// //       continue;

// //     if(j > 0)
// //     { 
// //       if(DEM(i-1, j-1) < this_elevation)
// //         FLOWDIR(i,j) += 1;
// //       if(DEM(i, j-1) < this_elevation)
// //         FLOWDIR(i,j) += 10000000;
// //     }
// //     if(j < NCOLS - 1)
// //     {
// //       if(DEM(i-1, j+1) < this_elevation)
// //         FLOWDIR(i,j) += 100;
// //       if(DEM(i, j+1) < this_elevation)
// //         FLOWDIR(i,j) += 1000;
// //     }
// //     if(DEM(i-1, j) < this_elevation)
// //       FLOWDIR(i,j) += 10;

// //   }
// //   std::cout << "DEBUG::done with edge flowdir" << std::endl;

// // }

// // //-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
// // // This function converts flow direction to list of row-col of receivers
// // // -> DEM: numpy array
// // //-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
// // void cppintail::flowdir_to_receiver_indices(int nodeID, std::vector<int>& receiver_nodes)
// // {
// //   int row,col; this->node_to_row_col(nodeID,row,col);
// //   std::vector<int> receiver_rows, receiver_cols;
// //   this->flowdir_to_receiver_indices( row,  col,  receiver_rows,  receiver_cols);
// //   receiver_nodes = std::vector<int>(receiver_rows.size());
// //   for(size_t i =0; i<receiver_rows.size();i++)
// //   {
// //     int nodeID,row = receiver_rows[i], col = receiver_cols[i];
// //     nodeID = this->row_col_to_node(row,col);
// //     receiver_nodes[i] = nodeID;
// //   }

// // }
// // void cppintail::flowdir_to_receiver_indices(int row, int col, std::vector<int>& receiver_rows, std::vector<int>& receiver_cols)
// // {

// //   // Making sure the vectors are empty
// //   receiver_rows = std::vector<int>(8);
// //   receiver_cols = std::vector<int>(8);  
// //   int this_code = FLOWDIR(row,col);
// //   if(this_code - 10000000 >= 0)
// //   {
// //     receiver_rows.emplace_back(row - 1);
// //     receiver_cols.emplace_back(col - 1);
// //     this_code -= 10000000;
// //   }

// //   if(this_code - 1000000 >= 0)
// //   {
// //     receiver_rows.emplace_back(row - 1);
// //     receiver_cols.emplace_back(col);
// //     this_code -= 1000000;
// //   }

// //   if(this_code - 100000 >= 0)
// //   {
// //     receiver_rows.emplace_back(row - 1);
// //     receiver_cols.emplace_back(col + 1);
// //     this_code -= 100000;
// //   }

// //   if(this_code - 10000 >= 0)
// //   {
// //     receiver_rows.emplace_back(row );
// //     receiver_cols.emplace_back(col + 1);
// //     this_code -= 10000;
// //   }

// //   if(this_code - 1000 >= 0)
// //   {
// //     receiver_rows.emplace_back(row + 1);
// //     receiver_cols.emplace_back(col + 1);
// //     this_code -= 1000;
// //   }

// //   if(this_code - 100 >= 0)
// //   {
// //     receiver_rows.emplace_back(row + 1);
// //     receiver_cols.emplace_back(col);
// //     this_code -= 100;
// //   }

// //   if(this_code - 10 >= 0)
// //   {
// //     receiver_rows.emplace_back(row + 1);
// //     receiver_cols.emplace_back(col - 1);
// //     this_code -= 10;
// //   }

// //   if(this_code - 1 >= 0)
// //   {
// //     receiver_rows.emplace_back(row + 1);
// //     receiver_cols.emplace_back(col);
// //     this_code -= 1;
// //   }

// //   receiver_rows.shrink_to_fit();
// //   receiver_cols.shrink_to_fit();

// //   std::vector<int> new_vecrow, new_vecol;

// //   for (size_t i = 0; i < receiver_rows.size(); i++ )
// //   {
// //     if(receiver_rows[i] < 0 || receiver_cols[i] < 0 || receiver_rows[i] >= NROWS || receiver_cols[i] >= NCOLS)
// //       continue;
// //     else
// //     {
// //       new_vecrow.push_back(receiver_rows[i] );
// //       new_vecol.push_back(receiver_cols[i]);
// //     }
// //   }

// //   receiver_rows = new_vecrow;
// //   receiver_cols = new_vecol;


// // }



// // void cppintail::find_nodes_with_no_donors( xt::pytensor<float,2>& DEM)
// // {
// //   // Attribute containing the no donors nodes
// //   no_donor_nodes = std::vector<int>(NROWS * NCOLS);
// //   std::vector<int> local_ndonors(NROWS * NCOLS,0);
// //   // std::cout << "DEBUG:HERE1" << std::endl;
// //   // Looping through the thingy
// //   int incrementer = 0;
// //   for(size_t i = 0; i < NROWS; i++)
// //   for(size_t j = 0; j < NCOLS; j++)
// //   {
// //     // Ignorign no data
// //     if(DEM(i,j) == NODATAVALUE)
// //       continue;

// //     // Getting the receivers
// //     std::vector<int> receiver_rows,receiver_cols;
// //     // std::cout << "DEBUG:HERE1.5 || " << i << "||" << j << std::endl;
// //     this->flowdir_to_receiver_indices(int(i), int(j), receiver_rows,  receiver_cols);
// //     // std::cout << "DEBUG:HERE1.6 || " << i << "||" << j << std::endl;

// //     // Incrementing the receivers
// //     for(size_t od=0; od<receiver_rows.size(); od++)
// //     {
// //       // std::cout << "DEBUG:HERE1.7 || " << receiver_rows[od] << "||" << receiver_cols[od] << std::endl;
// //       local_ndonors[row_col_to_node(receiver_rows[od],receiver_cols[od])] += 1; 
// //     }
// //   }
// //   // std::cout << "DEBUG:HERE2" << std::endl;


// //   // hunting for the no donor nodes
// //   for(size_t i = 0; i < local_ndonors.size(); i++)
// //   {
// //     if(local_ndonors[i] == 0)
// //     {
// //       no_donor_nodes[incrementer]  = int(i);
// //     }
// //   }
// //   // std::cout << "DEBUG:HERE3" << std::endl;


// //   no_donor_nodes.shrink_to_fit();

// // }







// // struct tempNode
// // {
// //   /// @brief Elevation data.
// //   float Zeta;
// //   /// @brief Row index value.
// //   int NodeIndex;
// // };

// // bool operator>( const tempNode& lhs, const tempNode& rhs )
// // {
// //   return lhs.Zeta > rhs.Zeta;
// // }
// // bool operator<( const tempNode& lhs, const tempNode& rhs )
// // {
// //   return lhs.Zeta < rhs.Zeta;
// // }


// // void cppintail::Initialise_MF_stacks(xt::pytensor<float,2>& DEM)
// // {

// //   // Initialise my stack to the maximum possible size (data - no data)
// //   MF_stack.clear();
// //   MF_stack.reserve(NROWS * NCOLS);

// //   // Sorting the node 
// //   std::priority_queue< tempNode, std::vector<tempNode>, std::greater<tempNode> > PriorityQueue;
// //   for(size_t row = 0; row<NROWS; row++)
// //   for(size_t col = 0; col<NCOLS; col++)
// //   {
// //     tempNode this_node;
// //     int this_nodeID = this->row_col_to_node(row, col);
// //     if(this_nodeID == NODATAVALUE)
// //       continue;

// //     this_node.Zeta = DEM(row,col);
// //     this_node.NodeIndex =  this_nodeID;
// //     PriorityQueue.push(this_node);
// //   }

// //   while(!PriorityQueue.empty())
// //     MF_stack.emplace_back(PriorityQueue.top().NodeIndex);

// //   MF_stack.shrink_to_fit();
// // }

// // void cppintail::compute_DA_slope_exp( double slexponent, xt::pytensor<float,2>& DEM)
// // {

// //   Drainage_area = xt::zeros<double>({NROWS,NCOLS});
  
// //   for(int i = int(MF_stack.size()); i >= 0; i++)
// //   {
// //     int this_node = MF_stack[i];
// //     int row,col; node_to_row_col(this_node,row,col);
// //     // Adding the first thingy
// //     Drainage_area(row,col) += XRES*YRES;
// //     // Getting the neighbors
// //     std::vector<int> neighrow, neighcol;
// //     this->flowdir_to_receiver_indices(row, col, neighrow, neighcol);
    
// //     if(neighrow.size() == 0)
// //       continue;

// //     std::vector<double> slope_rep(neighrow.size());
// //     double max_slope = std::numeric_limits<double>::min(); 
// //     for(size_t tn = 0; tn < neighrow.size(); tn++)
// //     {
// //       int nrow = neighrow[tn];
// //       int ncol = neighcol[tn];
// //       double dx = std::sqrt( std::pow( std::abs(nrow - row) * YRES,2) + std::pow(std::abs(ncol - col) * YRES,2) );
// //       double dz = DEM(row,col) - DEM(nrow,ncol);

// //       double this_slope = std::pow(dz/dx, slexponent);
// //       slope_rep[tn] = this_slope;
// //       if(max_slope<this_slope)
// //         max_slope = this_slope;
// //     }

// //     for(size_t tn = 0; tn < neighrow.size(); tn++)
// //     {
// //       slope_rep[tn] = slope_rep[tn]/max_slope;
// //       int nrow = neighrow[tn];
// //       int ncol = neighcol[tn];
// //       Drainage_area(nrow,ncol) = Drainage_area(nrow,ncol) + slope_rep[tn] * Drainage_area(row,col);
// //     }
// //   }
// //   // Done
// // }


// //  //###############################################  
// //  // Duck transition to general functions
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


// // //##################################################
// // //############# Stack stuff ########################
// // //##################################################
// // // Adapted from xarray-topo




#endif