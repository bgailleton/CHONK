//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#ifndef depressiontree_HPP
#define depressiontree_HPP

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

#include "chonkutils.hpp"
// All the xtensor requirements
#include "xtensor-python/pyarray.hpp" // manage the I/O of numpy array
#include "xtensor-python/pytensor.hpp" // same
#include "xtensor-python/pyvectorize.hpp" // Contain some algorithm for vectorised calculation (TODO)
#include "xtensor/xadapt.hpp" // the function adapt is nice to convert vectors to numpy arrays
#include "xtensor/xmath.hpp" // Array-wise math functions
#include "xtensor/xarray.hpp"// manages the xtensor array (lower level than the numpy one)
#include "xtensor/xtensor.hpp" // same
class DepressionTree
{

	//  ___________________
	// |                   |
	// |     Attributes    |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	// Size: number of nodes in the landscapes
	std::vector<int> node2tree;
	std::vector<int> node2outlet;
	std::vector<double> potential_volume;

	// Size: number of depressions in the tree
	//# tree connections
	std::vector<std::vector<int> > treeceivers;
	std::vector<int> parentree;
	std::vector<std::priority_queue< PQ_helper<int,int>, std::vector<PQ_helper<int,int> >, std::greater<PQ_helper<int,int> > > > fillers;
	std::vector<std::vector<int> > nodes;

	//# Node connections 
	std::vector<int> internode;
	std::vector<int> tippingnode;
	std::vector<int> externode;
	std::vector<int> pitnode;

	//# Depression characteristics
	std::vector<double> volume;
	std::vector<double> volume_sed;
	std::vector<double> volume_water;
	std::vector<double> hw_max;
	std::vector<double> hw;
	int indexer = 0;


  //  ___________________
	// |                   |
	// |   Constructors    |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	DepressionTree() {;};
	DepressionTree(int n_elements) {this->node2tree = std::vector<int>(n_elements, -1);this->node2outlet = std::vector<int>(n_elements, -1);};


	//  ___________________
	// |                   |
	// |   Adding Deps.    |
	// |   to the Tree     |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	// Registering depression
	void register_new_depression(xt::pytensor<double,1>& elevation,int pitnode, std::vector<int> children)
	{
		this->treeceivers.emplace_back(children);
		this->parentree.emplace_back(-1);
		this->nodes.emplace_back(std::vector<int>());
		this->internode.emplace_back(-1);
		this->tippingnode.emplace_back(-1);
		this->externode.emplace_back(-1);
		this->pitnode.emplace_back(-1);
		this->volume.emplace_back(0);
		this->volume_sed.emplace_back(0);
		this->volume_water.emplace_back(0);
		this->hw_max.emplace_back(elevation[pitnode]);
		this->hw.emplace_back(0);
		this->fillers.emplace_back(std::vector<std::priority_queue< PQ_helper<int,int>, std::vector<PQ_helper<int,int> >, std::greater<PQ_helper<int,int> > > >());
	}
	
	void register_new_depression(xt::pytensor<double,1>& elevation, int pitnode) {this->register_new_depression(xt::pytensor<double,1>& elevation,int pitnode, {-1,-1});}



  //  ___________________
	// |                   |
	// | Linking functions |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	void parenthood(int parent, std::vector<int> children)
	{
		this->treeceivers[parent] = children;
		for(auto i:children)
			this->parentree[i] = parent; 
	}

	void linkhood(int node,int in, int tip, int out) {this->internode[node] = in; this->tippingnode[node] = tip; this->externode[node] = out;}; 


  //  ___________________
	// |                   |
	// |     Navigation    |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	std::vector<int> get_all_children(int node, bool include_node = false)
	{
		// Initialising the output and reserving an arbitrary size
		std::vector<int> output; output.reserve(std::round(this->parentree.size()/2));
		std::queue<int> children; children.emplace(node);
		while(children.empty() == false)
		{
			int next = children.front();  children.pop();
			if( ((next == node) && include_node)  || (next != node))
				output.emplace_back(next);


			for(auto i : this->treeceivers[next])
			{
				if(i != -1)
					children.emplace(next);
			}
		}
		return output;

	}

	int get_ultimate_parent(int node)
	{ 
		while(this->parentree[node] != -1)
			{node = this->parentree[node];}
	  return node;
  }

	std::vector<int> get_all_nodes(int node)
	{
		std::vector<int> alldeps = this->get_all_children( node, true), output;

		size_t totsize = 0;
		for(auto i: alldeps)
		{
			totsize += this->nodes[i].size();
		}

		output.reserve(totsize);
		
		for(auto i: alldeps)
		{
			for(auto j:this->nodes[i])
			{
				output.emplace_back(j);
			}
		}

		return output;
	}
































};


#endif