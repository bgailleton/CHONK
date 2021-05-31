//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#ifndef CHONK_HPP
#define CHONK_HPP

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
		this->internode.emplace_back(-1);
		this->tippingnode.emplace_back(-1);
		this->externode.emplace_back(-1);
		this->pitnode.emplace_back(-1);
		this->volume.emplace_back(0);
		this->volume_sed.emplace_back(0);
		this->volume_water.emplace_back(0);
		this->hw_max.emplace_back(elevation[pitnode]);
		this->hw.emplace_back(0);
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







































};


#endif