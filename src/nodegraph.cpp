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

// Second version of the depth first algorithm where the original set of nodes is fixed, eg full topological sorting
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
  Mstack = xt::adapt(multiple_stack_fastscape( n_element, graph, this->not_in_stack));

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


std::vector<int> multiple_stack_fastscape(int n_element, std::vector<Vertex>& graph, std::vector<int>& not_in_stack)
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
    std::cout << "Important: If this happens at the start of the model with a random surface for few timesteps this is not critical." << std::endl;
    std::cout << "If it happens in the middle of a run with a mature mountain this is a problem." << std::endl;

    // throw std::runtime_error("stack underpopulated somehow:");
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