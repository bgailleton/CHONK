//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
/*
Header-only code for the binary tree sorting out the depressions in the landscapes
On itself it only contains the data structure and functions to manage the tree,
as well as functions to navigate through it or inserting elements.
The actual building happens in the nodegraph file
B.G. - June 2021

     
Example of tree Structure:
        6
        /\
       /  \
      /    \
     /      5
		/	      /\
   4       /  \
  /\      2    3
 /  \
0    1

*/
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


// Single class hosting everything
class DepressionTree
{

// Everything is public cause I have bad coding habits but it's fine

public:
	//  ___________________
	// |                   |
	// |     Attributes    |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	// Size: number of nodes in the landscapes
	// index node ID -> value: index in the depression tree (-1 if no depression)
	std::vector<int> node2tree;
	// index node ID -> value: node id of the outlet (DEPRECATED?? I think)
	std::vector<int> node2outlet;
	// index node ID -> value: the potential volume the depression can fit in. 
	// Constructed in ascedeing order from pit bottom 
	std::vector<double> potential_volume;

	// Size: number of depressions in the tree
	// index: depression ID -> value: the two children in the binary tree ({-1,-1} if no children)
	std::vector<std::vector<int> > treeceivers; 
	// index: depression id -> value: the parent in the binary tree, -1 if no parent
	std::vector<int> parentree;
	// index = depression ID -> value: level of the depression (increment by 1 from the bottom depression to the top of each local tree, maximum value prevail in case of different level merging)
	std::vector<int> level;
	// index: depression ID -> value: priority queue utilised by the filling processes
	// When a parent depression is created it merges their PQ 
	std::vector<std::priority_queue< PQ_helper<int, double>, std::vector<PQ_helper<int, double> >, std::greater<PQ_helper<int, double> > > > filler;
	// index: depression ID -> value: vector of nodes in the depression
	std::vector<std::vector<int> > nodes;
	// index: depression ID -> value: vector of proportion of labels in the sediment flux
	std::vector<std::vector<double> > label_prop;
	// index Depression ID -> value: I am not sure...
	std::vector<bool> active;

	//# Node connections 
	// index: depression ID -> value: an internode connected to the outlet
	std::vector<int> internode;
	// index: depression ID -> value: tipping node, or outlet of the depression. if twin, will be the same for 2 depressions and that is how the tree is built.
	std::vector<int> tippingnode;
	// index: depression ID -> value: a receiver of the outlet outside of the current basin 
	std::vector<int> externode;
	// index: depression ID -> value: pit of the depression (ie bottom node). THis is mostly important for childless depressions, the parent depressions have a random pit being one of its children pits
	std::vector<int> pitnode;

	//# Depression characteristics
	// index: depression ID -> value: total volume of the depression 
	std::vector<double> volume;
	// index: depression ID -> value: volume of actual sediment hosted in the depression 
	std::vector<double> volume_sed;
	// index: depression ID -> value: volume of actual water hosted in the depression 
	std::vector<double> volume_water;
	// index: depression ID -> value: maximum water height of the depression (in absolute elevation)
	std::vector<double> hw_max;
	// index: depression ID -> value: actual water height of the depression (in absolute elevation)
	std::vector<double> hw;
	// indexer used to do stuff, not sure. Might be deprecated.
	int indexer = 0;

	// Checkers
	// These checker are important during the process phase of the model: a master depression cannot be processed
	// before all of its level0 children have been.
	// index: depression ID -> value: N level 0 depression in this whole system
	std::vector<int> n_0level_children_in_total;
	// index: depression ID -> value: N level 0 depression in this whole system that have been processed
	std::vector<int> n_0level_children_in_total_done;


  //  ___________________
	// |                   |
	// |   Constructors    |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	// Default constructor, does not do much and should not be used
	DepressionTree() {;};
	// Initiate a depression tree and create the global vector to the full size.
	DepressionTree(int n_elements) {this->node2tree = std::vector<int>(n_elements, -1);this->node2outlet = std::vector<int>(n_elements, -1);this->potential_volume = std::vector<double>(n_elements, -1);};


	//  ___________________
	// |                   |
	// |   Adding Deps.    |
	// |   to the Tree     |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	// Registering depression: creating a new depression in the tree, it mostly makes sure the size of all the vectors are rightly expended
	// Also registers the right pit node to the depression 
	void register_new_depression(xt::pytensor<double,1>& elevation, int pitnode, std::vector<int> children)
	{
		// eventually registering children there
		this->treeceivers.emplace_back(children);
		// No parent if just created
		this->parentree.emplace_back(-1);
		// level 0 by default, get calculated dynamically in the building process
		this->level.emplace_back(0);
		// no nodes yet, the first one gets added later
		this->nodes.emplace_back( std::vector<int>()) ;

		//same here, nothing by defautl, gets calculated statically at the end
		this->n_0level_children_in_total.emplace_back(0);
		this->n_0level_children_in_total_done.emplace_back(0);

		// No label at first
		this->label_prop.emplace_back(std::vector<double>());
		// connecting nodes are calculated when a depression outlets/merges
		this->internode.emplace_back(-1);
		this->tippingnode.emplace_back(-1);
		this->externode.emplace_back(-1);
		// Pit node given at first
		this->pitnode.emplace_back(pitnode);
		// No volumes at first
		this->volume.emplace_back(0);
		this->volume_sed.emplace_back(0);
		this->volume_water.emplace_back(0);
		// Initial hw is the one of the pits
		this->hw_max.emplace_back(elevation[pitnode]);
		this->hw.emplace_back(elevation[pitnode]);
		// Empty PQ
		this->filler.emplace_back(std::priority_queue< PQ_helper<int,double>, std::vector<PQ_helper<int,double> >, std::greater<PQ_helper<int,double> > >());
		// whatever that is, that is false RN
		this->active.emplace_back(false);
	}

	// registering new depression without bothering about children
	void register_new_depression(xt::pytensor<double,1>& elevation, int pitnode) 
	{
		this->register_new_depression(elevation,pitnode, {-1,-1});
	}

  //  ___________________
	// |                   |
	// | Linking functions |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	// register parent-children relationship. Sort of birth certificate.
	void parenthood(int parent, std::vector<int> children)
	{
		// Children to parent
		this->treeceivers[parent] = children;
		// Parent to children 
		for(auto i:children)
			this->parentree[i] = parent; 
	}

	// linking depression to nodes
	void linkhood(int node,int in, int tip, int out) {this->internode[node] = in; this->tippingnode[node] = tip; this->externode[node] = out;}; 

	// Static compilation of depressioj level, to be run after the building of the tree.
	// Iterates through depressions and for each orphan goes through all children and increment for each level 0 ones
	void compile_n_0_level_children()
	{
		for(int i = 0; i < int(this->parentree.size()); i++)
		{
			if(this->parentree[i] == -1)
			{
				std::vector<int> chilll = this->get_all_children(i,true);
				for(auto U:chilll)
				{
					if(this->level[U] == 0)
						this->n_0level_children_in_total[i] ++;
				}
			}
		}
	}


  //  ___________________
	// |                   |
	// |     Navigation    |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	// Simple Breadth first traversal to get all children, include_node determinesif the mother dep must be included or not in the outputs
	std::vector<int> get_all_children(int node, bool include_node = false)
	{
		// Initialising the output and reserving an arbitrary capacity, it does not matter much
		std::vector<int> output; output.reserve(std::round(this->parentree.size()/2));
		std::queue<int> children; children.emplace(node);
		while(children.empty() == false)
		{
			int next = children.front();  
			children.pop();
			if( ((next == node) && include_node)  || (next != node))
				output.emplace_back(next);
			for(size_t k=0; k <  this->treeceivers[next].size(); k++)
			{
				int i = this->treeceivers[next][k];
				if(i != -1)
				{
					children.emplace(i);
				}
			}
		}
		return output;
	}

	// Returns the topmost parent linked to a depression
	int get_ultimate_parent(int dep)
	{ 
		if(dep == -1)
			return dep;
		while(this->parentree[dep] != -1)
			{dep = this->parentree[dep];}
	  return dep;
  }

  // Returns the twin of a depression if it has one, or -1
  int get_twin(int dep)
  {
  	if(this->parentree[dep] == -1) {return -1;}
  	for(auto i:this->treeceivers[this->parentree[dep]]){if(i != dep) {return i;} ;};
  	return -1;
  }

  // Returns all the ndoes of the depression (includes the ones from the child depressions)
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

	// Same than above but takes time and effort to sort the depressions by elevation of their pit for some reason. I think I am not using this anymorel.
	std::vector<int> get_all_nodes_bottom2top(int node, xt::pytensor<double,1>& elevation)
	{

		std::priority_queue< PQ_helper<int,double>, std::vector<PQ_helper<int,double> >, std::greater<PQ_helper<int,double> > > sorter;
		std::vector<int> alldeps = this->get_all_children( node, true), output;
		size_t totsize = 0;
		for(auto i: alldeps)
		{
			totsize += this->nodes[i].size();
			if(this->nodes[i].size()>0)
				sorter.emplace(PQ_helper<int,double>(i,elevation[this->nodes[i][0]]));
		}
		output.reserve(totsize);
		while(sorter.empty() == 0)
		{
			int i = sorter.top().node;
			sorter.pop();
			for(auto j:this->nodes[i])
			{
				output.emplace_back(j);
			}
		}
		return output;
	}

	// Returns a topological order of the treebased on the level of the depression in order to process the 0 level first and their ultimate parents last
	std::vector<int> get_treestack()
	{
		std::priority_queue< PQ_helper<int,int>, std::vector<PQ_helper<int,int> >, std::greater<PQ_helper<int,int> > > sorter;
		for(size_t i=0; i<this->treeceivers.size(); i++)
			sorter.emplace(PQ_helper<int,int>(i,this->level[i]));
		std::vector<int> stack(this->treeceivers.size());
		while(sorter.size()>0)
		{
			stack.emplace_back(sorter.top().node);
			sorter.pop();
		}
		return stack;
	}

	// Same than above but just for a local tree
	std::vector<int> get_local_treestack(int dep)
	{
		std::priority_queue< PQ_helper<int,int>, std::vector<PQ_helper<int,int> >, std::greater<PQ_helper<int,int> > > sorter;
		// std::cout << "blag1 " << std::endl;
		auto daft = this->get_ultimate_parent(dep);
		// std::cout << "blag2 " << daft << std::endl;
		std::vector<int> these_seps = this->get_all_children(daft, true);
		// std::cout << "blag2 " << these_seps.size() << std::endl;
		for(size_t i=0; i<these_seps.size(); i++)
			sorter.emplace(PQ_helper<int,int>(these_seps[i],this->level[these_seps[i]]));
		std::vector<int> stack; stack.reserve(these_seps.size());
		while(sorter.size()>0)
		{
			stack.emplace_back(sorter.top().node);
			sorter.pop();
		}
		return stack;
	}

	// Getting all top level depressions, orphan, parentestest, which ever term you prefer
	std::vector<int> get_all_parentfree_depressions()
	{
		std::vector<int> output;
		for (int i = 0; i< int(this->parentree.size()); i++ )
		{
			if(parentree[i] == -1)
				output.emplace_back(i);
		}
		return output;
	}

	// retunrs the last ID used 
	int get_last_id(){return int(this->parentree.size()) - 1;}

	// check if a depression has chilfren
	bool has_children(int dep)
	{
		for(auto i: this->treeceivers[dep])
		{
			if(i != -1)
				return true;
		}
		return false;
	}

	// returns the number of depressions in the whole tree
	int get_n_dep(){return int(this->parentree.size());}

  //  ___________________
	// |                   |
	// | Filling helpers   |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ
	// ; ;

	//returns a is in queue raster for the filling process
	std::vector<bool> get_isinQ4dep(int node)
	{
		std::vector<bool> is_in_queue(this->node2tree.size(), false);
		std::vector<PQ_helper<int,double> >&  vecfrompq = Container(this->filler[node]);
		for(auto& erm: vecfrompq)
			is_in_queue[erm.node] = true;
		return is_in_queue;
	}

	// Check if the depression is the direct child of another 
	bool is_direct_child_of(int is_child, int of){if(this->parentree[is_child] == of) {return true;}else{return false;};}
	// Check if the depression is the direct child of another, from node ID (no depression = false)
	bool is_direct_child_of_from_node(int is_child, int of)
	{
		if(this->node2tree[is_child] == -1 || this->node2tree[of] == -1){return false;}
	  if(this->parentree[this->node2tree[is_child]] == this->node2tree[of]) {return true;}
	  else{return false;};
	}

  // Check wether the thingy is a child (direct or not) of another thingy
	bool is_child_of(int is_child, int of)
	{
		if(is_child == -1 || of == -1){return false;}
		std::queue<int> children; children.emplace(of);
		while(children.empty() == false)
		{
			int next = children.front();  children.pop();
			for(auto i : this->treeceivers[next])
			{
				if(i != -1)
					children.emplace(i);
				if(i == is_child)
					return true;
			}
		}
		return false;
	}

	// Merge children into parents, to transmit info upwardp
	void merge_children_to_parent(std::vector<int> children, int parent, int outlet_node, std::vector<int>& neightbors, xt::pytensor<double,1>& elevation)
	{

		// parent has these children
		this->treeceivers[parent] = children;
		// base hw of parent is the one of children
		this->hw_max[parent] = this->hw_max[children[0]];

		// Level is set to max of level of each child + 1
		this->level[parent] = std::max(this->level[children[0]], this->level[children[1]]) + 1;

		// DEBUG STATEMENT
		if(parent == children[0] || parent == children[1])
			throw std::runtime_error("PARENTAL ISSUE");

		// Hacking the underlying container of the PQ of children, and merging them into a single one for the parent
		std::vector<PQ_helper<int,double> > c1 = Container(this->filler[children[0]]);
		std::vector<PQ_helper<int,double> > c2 = Container(this->filler[children[1]]);
		std::vector<PQ_helper<int,double> > concat; concat.reserve(c1.size() + c2.size());
		for(auto n:c1) 
			concat.emplace_back(n);
		for(auto n:c2) 
			concat.emplace_back(n);

		// initialising the parent PQ with the main one
		this->filler[parent] = std::priority_queue< PQ_helper<int, double>, std::vector<PQ_helper<int, double> >, std::greater<PQ_helper<int, double> > >(concat.begin(), concat.end());

		// Children are merging which means I need to determine their externodeand make sure they are in each other
		// By convention, picking the lowest one of the neightbour of the tipping node
		int lower_elev_c1,lower_elev_c2;
		double vlower_elev_c1 = 1e36,vlower_elev_c2 = 1e36;
		// looping thourgh neighbours and getting it
		for(auto n:neightbors)
		{
			int tdep = this->node2tree[n];
			if(tdep == -1 )
				continue;
			tdep = this->get_ultimate_parent(tdep);

			if(elevation[n] >= vlower_elev_c1 && tdep == children[0])
			{
				lower_elev_c1 = n;
				vlower_elev_c1 = elevation[n];
			}
			if(elevation[n] >= vlower_elev_c2 && tdep == children[1])
			{
				lower_elev_c2 = n;
				vlower_elev_c2 = elevation[n];
			}
		}

		// actually saving them
		this->externode[children[0]] = lower_elev_c2;
		this->externode[children[1]] = lower_elev_c1;



		// each children needs to know their parents	
		for(auto i:children)
		{
			// Your parent is
			this->parentree[i] = parent;
			// Your filler is reinnit
			this->filler[i] = std::priority_queue< PQ_helper<int, double>, std::vector<PQ_helper<int, double> >, std::greater<PQ_helper<int, double> > >();
			// Volume transferred to parent
			this->volume[parent] += this->volume[i];
			// my tipping node is the one of me twin
			this->tippingnode[i] = outlet_node;
		}

		// taking care of the volume incrementation for the new parent
		this->potential_volume[outlet_node] = this->volume[parent];

	}



  //  ___________________
	// |                   |
	// |   Postprocessing  |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	// return a vector depression ID -> ultimate parent
	std::vector<int> dep2top()
	{
		std::vector<int> output(this->treeceivers.size(),-1);
		for(int i = 0; i < int(this->treeceivers.size()); i++)
			output[i] = this->get_ultimate_parent(i);
		return output;
	}	

	// Return a vector node ID -> ultimate parent -> -1 if not in depression
	std::vector<int> node2top()
	{
		std::vector<int> output(this->node2tree.size(),-1), deptop = this->dep2top();
		for(int i = 0; i < int(this->node2tree.size()); i++)
		{
			if(this->node2tree[i] == -1)
				continue;
			output[i] = deptop[this->node2tree[i]];
		}
		return output;
	}	


	double get_volume_of_children(int dep)
	{
		double totvol = 0;
		for(auto ch: this->treeceivers[dep])
		{
			if(ch>=0)
				totvol += this->volume[ch];
		}
		return totvol;
	}

  //  ___________________
	// |                   |
	// |      other        |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	// Print the tree in a rather rudimentary way
	void printree()
	{
		std::string gabul = "Printing the depression tree... \n";
		std::cout << gabul << std::endl;

		for (size_t i=0; i<this->parentree.size() ; i++)
		{
			std::cout << "Dep. " << i << ": children: {" << this->treeceivers[i][0] << "," <<  this->treeceivers[i][1] << "} -> pot. V:" << this->volume[i] << std::endl;
		}

	}

	// sum the total volume of all master description
	double get_sum_of_all_volume_full_lake()
	{
		double tot = 0;
		for (size_t i=0 ; i<this->parentree.size(); i++)
		{
			if(this->parentree[i] == -1)
				tot += this->volume[i];
		}
		std::cout << tot << std::endl;
		return tot;
	}

};




  //  ___________________
	// |                   |
	// |      Done!        |
	// |___________________| 
  //      ||  (\__/)
  //       \\ (•ㅅ•) 
  //        ||C 　 \


























#endif