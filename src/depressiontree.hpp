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
public:
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
	std::vector<int> level;
	std::vector<std::priority_queue< PQ_helper<int, double>, std::vector<PQ_helper<int, double> >, std::greater<PQ_helper<int, double> > > > filler;
	std::vector<std::vector<int> > nodes;
	std::vector<std::vector<double> > label_prop;
	std::vector<bool> active;

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
	DepressionTree(int n_elements) {this->node2tree = std::vector<int>(n_elements, -1);this->node2outlet = std::vector<int>(n_elements, -1);this->potential_volume = std::vector<double>(n_elements, -1);};


	//  ___________________
	// |                   |
	// |   Adding Deps.    |
	// |   to the Tree     |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	// Registering depression
	void register_new_depression(xt::pytensor<double,1>& elevation, int pitnode, std::vector<int> children)
	{
		this->treeceivers.emplace_back(children);
		this->parentree.emplace_back(-1);
		this->level.emplace_back(0);
		this->nodes.emplace_back(std::vector<int>());
		this->label_prop.emplace_back(std::vector<double>());
		this->internode.emplace_back(-1);
		this->tippingnode.emplace_back(-1);
		this->externode.emplace_back(-1);
		this->pitnode.emplace_back(-1);
		this->volume.emplace_back(0);
		this->volume_sed.emplace_back(0);
		this->volume_water.emplace_back(0);
		this->hw_max.emplace_back(elevation[pitnode]);
		this->hw.emplace_back(0);
		this->filler.emplace_back(std::priority_queue< PQ_helper<int,double>, std::vector<PQ_helper<int,double> >, std::greater<PQ_helper<int,double> > >());
		this->active.emplace_back(false);
	}

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
		if(node == -1)
			return node;
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

	std::vector<int> get_all_nodes_bottom2top(int node, xt::pytensor<double,1>& elevation)
	{

		std::priority_queue< PQ_helper<int,double>, std::vector<PQ_helper<int,double> >, std::greater<PQ_helper<int,double> > > sorter;
		std::vector<int> alldeps = this->get_all_children( node, true), output;

		size_t totsize = 0;
		for(auto i: alldeps)
		{
			totsize += this->nodes[i].size();
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

	std::vector<int> get_local_treestack(int dep)
	{
		std::priority_queue< PQ_helper<int,int>, std::vector<PQ_helper<int,int> >, std::greater<PQ_helper<int,int> > > sorter;
		std::vector<int> these_seps = this->get_all_children(this->get_ultimate_parent(dep));
		for(size_t i=0; i<these_seps.size(); i++)
			sorter.emplace(PQ_helper<int,int>(these_seps[i],this->level[these_seps[i]]));

		std::vector<int> stack(these_seps.size());
		while(sorter.size()>0)
		{
			stack.emplace_back(sorter.top().node);
			sorter.pop();
		}
		return stack;
	}


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


	int get_last_id(){return int(this->parentree.size()) - 1;}

	bool has_children(int dep)
	{
		for(auto i: this->treeceivers[dep])
		{
			if(i != -1)
				return true;
		}
		return false;
	}

	int get_n_dep(){return int(this->parentree.size());}

  //  ___________________
	// |                   |
	// | Filling helpers   |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ

	std::vector<bool> get_isinQ4dep(int node)
	{
		std::vector<bool> is_in_queue(this->node2tree.size(), false);
		std::vector<PQ_helper<int,double> >&  vecfrompq = Container(this->filler[node]);
		for(auto& erm: vecfrompq)
			is_in_queue[erm.node] = true;
		return is_in_queue;
	}

	bool is_direct_child_of(int is_child, int of){if(this->parentree[is_child] == of) {return true;}else{return false;};}
	bool is_direct_child_of_from_node(int is_child, int of)
	{
		if(this->node2tree[is_child] == -1 || this->node2tree[of] == -1){return false;}
	  if(this->parentree[this->node2tree[is_child]] == this->node2tree[of]) {return true;}
	  else{return false;};
	}


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

	void merge_children_to_parent(std::vector<int> children, int parent, int outlet_node, std::vector<int>& neightbors, xt::pytensor<double,1>& elevation)
	{

		this->treeceivers[parent] = children;
		this->level[parent] = std::max(this->level[children[0]], this->level[children[0]]) + 1;
		if(parent == children[0] || parent == children[1])
			throw std::runtime_error("PARENTAL ISSUE");

		std::vector<PQ_helper<int,double> > c1 = Container(this->filler[children[0]]);
		std::vector<PQ_helper<int,double> > c2 = Container(this->filler[children[1]]);
		std::vector<PQ_helper<int,double> > concat; concat.reserve(c1.size() + c2.size());
		for(auto n:c1) 
			concat.emplace_back(n);
		for(auto n:c2) 
			concat.emplace_back(n);

		// merging the PQs
		this->filler[parent] = std::priority_queue< PQ_helper<int, double>, std::vector<PQ_helper<int, double> >, std::greater<PQ_helper<int, double> > >(concat.begin(), concat.end());


		int lower_elev_c1,lower_elev_c2;
		double vlower_elev_c1 = 1e36,vlower_elev_c2 = 1e36;
		for(auto n:neightbors)
		{
			int tdep = this->node2tree[n];
			if(tdep == -1 )
				continue;
			tdep = this->get_ultimate_parent(tdep);

			if(elevation[n] >= vlower_elev_c1 && tdep == children[0])
				lower_elev_c1 = n;
			if(elevation[n] >= vlower_elev_c2 && tdep == children[1])
				lower_elev_c2 = n;
		}

		this->externode[children[0]] = lower_elev_c2;
		this->externode[children[1]] = lower_elev_c1;

		for(auto i:children)
		{
			this->parentree[i] = parent;
			this->filler[i] = std::priority_queue< PQ_helper<int, double>, std::vector<PQ_helper<int, double> >, std::greater<PQ_helper<int, double> > >();
			this->volume[parent] += this->volume[i];
			this->tippingnode[i] = outlet_node;
		}

	}



  //  ___________________
	// |                   |
	// |   Postprocessing  |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ


	std::vector<int> dep2top()
	{
		std::vector<int> output(this->treeceivers.size(),-1);
		for(int i = 0; i < int(this->treeceivers.size()); i++)
			output[i] = this->get_ultimate_parent(i);
		return output;
	}	

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

  //  ___________________
	// |                   |
	// |      other        |
	// |___________________| 
	//            (\__/)||
	//            (•ㅅ•) ||
	//            / 　 づ


	void printree()
	{
		std::string gabul = "Printing the depression tree... \n";
		std::cout << gabul << std::endl;

		for (size_t i=0; i<this->parentree.size() ; i++)
		{
			std::cout << "Dep. " << i << ": children: {" << this->treeceivers[i][0] << "," <<  this->treeceivers[i][1] << "} -> pot. V:" << this->volume[i] << std::endl;
		}

	}

};































#endif