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
#include <thread>

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


// ######################################################################
// ######################################################################
// ######################################################################
// ###################### Model Runner ##################################
// ######################################################################
// ######################################################################

// Initialises the model object, actually Does not do much but is required.
void ModelRunner::create(double ttimestep, std::vector<std::string> tordered_flux_methods, std::string tmove_method)
{
  // Saving all the attributes
  this->timestep = ttimestep;
  this->ordered_flux_methods = tordered_flux_methods;
  this->move_method = tmove_method;
  // By default the lake solver is activated
  this->lake_solver = true;
  // Initialising the labelling stuff
  this->initialise_intcorrespondance();
  this->prepare_label_to_list_for_processes(); 
  this->Qw_in = 0;
  this->Qw_out = 0;
  this->Ql_in = 0;
  this->Ql_out = 0;
}

// initialising the node graph and the chonk network
void ModelRunner::initiate_nodegraph()
{

  std::cout << "initiating nodegraph..." <<std::endl;
  // Creating the nodegraph and preprocessing the depression nodes

  // First jsut sorting out the active node array.
  // Needs cleaning and optimisation cause some routines with bool where somehow buggy
  xt::pytensor<bool,1> active_nodes = xt::zeros<bool>({this->io_int_array["active_nodes"].size()});
  xt::pytensor<int,1>& inctive_nodes = this->io_int_array["active_nodes"];


  // Converting int to bool
  for(size_t i =0; i<inctive_nodes.size(); i++)
  {
    int B = inctive_nodes[i];
    if(B==1)
      active_nodes[i] = true;
    else
      active_nodes[i] = false;
  }

  //Dat is the real stuff:
  // Initialising the graph
  this->graph = NodeGraphV2(this->io_double_array["surface_elevation"], active_nodes,this->io_double["dx"], this->io_double["dy"],
this->io_int["n_rows"], this->io_int["n_cols"]);

  // std::cout << "done, sorting few stuff around ..." << std::endl;
  
  // Chonkification: initialising chonk network
  //# clearing the chonk network
  if(this->chonk_network.size()>0)
  {
    this->chonk_network.clear();
  }
  //# Creating noew empty chonks
  this->chonk_network = std::vector<chonk>();
  this->chonk_network.reserve(size_t(this->io_int["n_elements"]));

  // filling my network with empty chonks
  for(size_t i=0; i<size_t(this->io_int["n_elements"]); i++)
  {
    this->chonk_network.emplace_back(chonk(int(i), int(i), false));
    this->chonk_network[i].initialise_local_label_tracker_in_sediment_flux(this->n_labels);
  }

  // Initialising water balance
  this->Qw_in = 0;
  this->Qw_out = 0;
  this->Ql_in = 0;
  this->Ql_out = 0;

  // Stuff
  // I NEED TO WORK ON THAT PART YO
  // This add previous inherited water from previous lakes
  // Note that it "empties" the lake and reinitialise the depth. If there is still a reason to for the lake, it will form it
  // std::cout << "wat" << std::endl;
  this->process_inherited_water();
  // std::cout << "er" << std::endl;

  // Also initialising the Lake graph
  //# no lakes so far
  lake_network = std::vector<Lake>();
  //# incrementor reset to 0
  lake_incrementor = 0;
  //# no nodes in lakes
  node_in_lake = std::vector<int>(this->io_int["n_elements"], -1);

  std::cout << "done with setting up graph stuff" <<  std::endl;
}

void ModelRunner::run()
{
  // Alright now I need to loop from top to bottom
  std::cout << "Starting the run" << std::endl;
  
  // Keeping track of which node is processed, for debugging and lake management
  is_processed = std::vector<bool>(io_int["n_elements"],false);


  // Aliases for efficiency
  xt::pytensor<int,1>& inctive_nodes = this->io_int_array["active_nodes"];
  xt::pytensor<double,1>&surface_elevation =  this->io_double_array["surface_elevation"];
  // well area of a cell to gain time
  double cellarea = this->io_double["dx"] * this->io_double["dy"];
  // Debug checker
  int underfilled_lake = 0;
  std::cout << "starting iteration" << std::endl;
  // Iterating though all the nodes
  for(int i=0; i<io_int["n_elements"]; i++)
  {
    // Getting the current node in the Us->DS stack order
    int node = this->graph.get_MF_stack_at_i(i);
    // Processing that node
    this->process_node(node, is_processed, lake_incrementor, underfilled_lake, inctive_nodes, cellarea, surface_elevation, true);   
    // std::cout << this->chonk_network[node].get_water_flux() << std::endl; 
    // Switching to the next node in line
  }

  std::cout << "Ending the run" << std::endl;
  // temp debug thingy 
  // if(underfilled_lake>0)
  //   std::cout << "DEBUGINFO::I called the underfilled function " << underfilled_lake << "times" << std::endl;

  // Calling the finalising function: it applies the changes in topography and I think will apply the lake sedimentation
  this->finalise();
  // Done
  // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  std::cout << "done" << std::endl;
}

void ModelRunner::process_node(int& node, std::vector<bool>& is_processed, int& lake_incrementor, int& underfilled_lake,
  xt::pytensor<int,1>& inctive_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep)
{
    // Just a check: if the lake solver is not activated, I have no reason to reprocess node
    if(this->lake_solver == false && is_processed[node])
      return;

    // DEBUG
    // this->print_chonk_info(node);
    // std::cout << "Proc: " << node << " this->chonk_network[0] is " << this->chonk_network[0].get_current_location() << std::endl;

    // if I reach this stage, this node can be labelled as processed
    is_processed[node] = true;

    // manages the fluxes before moving the particule: accumulating DA, precipitation, infiltration, evaporation, ...
    this->manage_fluxes_before_moving_prep(this->chonk_network[node],this->label_array[node] );



    // Uf the lake solving is activated, then I go through the (more than I expected) complex process of filing a lake correctly
    if(this->lake_solver) 
    {
      // 1) checking if my node has a lake ID  
      int lakeid = this->node_in_lake[node];

      // if it has no lake id (ie is not a lake) and is not the bottom of an active depression then I skip that part 
      if(lakeid == -1 && this->graph.is_depression(node) == false)
      {
        goto nolake;
      }
      

      // if It has no preexisting lake AND is the bottom of the depression, I initiate a new lake in the lake network
      if(this->graph.is_depression(node) && lakeid == -1)
      { 
        this->lake_network.push_back(Lake(lake_incrementor));
        lakeid = lake_incrementor;
        lake_incrementor++;
      }

       // If my existing lake has a parent (ie a mother lake that had ingested it beforehand) then I consider the new as the one
      else if(this->lake_network[lakeid].get_parent_lake() >= 0)
      {
        lakeid = this->lake_network[lakeid].get_parent_lake();
      }

      // I have my existing/new lake in the (hopefuly) correct form, I need now to pour water in it
      // Calculating volume: the water flux of my CHONK * timestep (to transform into a volume of water)
      double water_volume = this->chonk_network[node].get_water_flux() * timestep;
      // CHONK directly stores the total sediment volume (already a Volume there)
      double sedvol = this->chonk_network[node].get_sediment_flux();
      
      // pouring sediment into the lake
      this->lake_network[lakeid].pour_sediment_into_lake(sedvol, this->chonk_network[node].get_other_attribute_array("label_tracker") );      
      // Pouring water into the lake, it also finds if there is an outlet and reprocess the water fluxes for dat one
      this->lake_network[lakeid].pour_water_in_lake(water_volume, node, node_in_lake, is_processed, inctive_nodes,lake_network, surface_elevation,graph, cellarea, timestep,chonk_network,this->Ql_out);
    
      // getting the outlet node of my lake. aOOOOOOOOOOOO . will be -9999 if there is no
      int outlet = this->lake_network[lakeid].get_lake_outlet();      

      // OK first step is to check wether this outlet is a lake itself or not
      if(outlet < 0)
        return;

      int lakoutid = node_in_lake[outlet];
      if(lakoutid >= 0)
      {
        if(this->lake_network[lakoutid].get_parent_lake() >= 0)
          lakoutid = this->lake_network[lakoutid].get_parent_lake();
      }

      
      
      // Checking if my node is active (i.e at a boundary or not)
      xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];

      // if in boundary: just transmit the flux and fuck off
      if(active_nodes[outlet] == false)
      {
        chonk& this_chonk = this->lake_network[lakeid].get_outletting_chonk();
        this->chonk_network[outlet].add_to_water_flux(this_chonk.get_water_flux());
        this->chonk_network[outlet].add_to_sediment_flux(this->lake_network[lakeid].get_outletting_chonk().get_sediment_flux(),
             this->lake_network[lakeid].get_outletting_chonk().get_other_attribute_array("label_tracker"));
        return;
      }


      // checking if my final outlet exists AND has been processed before
      if(is_processed[outlet] == false)
      {
        this->chonk_network[outlet].add_to_water_flux(this->lake_network[lakeid].get_outletting_chonk().get_water_flux());
        this->chonk_network[outlet].add_to_sediment_flux(this->lake_network[lakeid].get_outletting_chonk().get_sediment_flux(),
             this->lake_network[lakeid].get_outletting_chonk().get_other_attribute_array("label_tracker"));
        return;
      }

      if(outlet >= 0)
      {

        // std::cout << "OUTLET 0.1 is " << outlet << std::endl;
        // std::cout << "Bulf3.2" << std::endl;
        // Arg it has been, then this is where the code reprocess a bunch of node. It happens in two relatively rare cases:
        // 1) the lake is outletting into an imbricated upstream but underfilled depression, then all the node had been processed to check wether this depression could be filled and plot twist it could not
        // hence the reprocessing of the intermediate nodes to make sure I process them correctly before pouring more water into the dowstream(s) lake
        // 2) Rare but can happen: my outlet does not correspond to the approximation of cordonnier et al. 2019
        // The algorithm approximate lake outlets with a d8 flow routing, which in very rare cases does not correspond to the real case (especially in convoluted random noise landscapes)
        // then usually a small pan of landscape needs to be reprocessed

        // Initiating the vector of nodes affected by this problem
        std::map<int, double > lake_water_corrector, lake_sed_corrector;
        std::map<int, double > lake_water, lake_sed; std::map<int,int> lake_node_entry;
        std::map<int, chonk > lake_chonks;
        std::deque<int> lake_to_process;
        std::vector<bool> lakeinQ(this->lake_network.size(),false);


        // TO DO OPTIMISE THIS
        xt::pytensor<bool,1> temp_bool = xt::zeros<bool>({this->io_int["n_elements"]});
        size_t it = 0;
        for(auto& Uf:active_nodes)
        {
          if(Uf)
            temp_bool[it] = true;
          it++;
        }


        bool there_are_still_lakes_to_reproc = true;

        while(there_are_still_lakes_to_reproc)
        {
           // This steps check wether the lake outflows directly in another lake
          // std::cout << "OUTLET IS " << outlet << std::endl;
          while(outlet >= 0 && is_processed[outlet] && lakoutid > -1)
          {
            // std::cout << "OUTLET 0.01 is " << outlet << std::endl;

            if(lake_network[lakeid].get_parent_lake()>=0)
              lakeid = lake_network[lakeid].get_parent_lake();
            if(lake_network[lakoutid].get_parent_lake()>=0)
              lakoutid = lake_network[lakoutid].get_parent_lake();

            // Pouring sediments into the new lake
            this->lake_network[lakoutid].pour_sediment_into_lake(this->lake_network[lakeid].get_outletting_chonk().get_sediment_flux(),
             this->lake_network[lakeid].get_outletting_chonk().get_other_attribute_array("label_tracker"));

            // Pouring water into the new lake
            std::cout << "bulf?" << std::endl;
            this->lake_network[lakoutid].pour_water_in_lake(this->lake_network[lakeid].get_outletting_chonk().get_water_flux() * this->timestep,
             outlet, node_in_lake, is_processed, inctive_nodes,lake_network, surface_elevation,graph, cellarea, timestep,chonk_network, this->Ql_out);
            std::cout << "bulf!" << std::endl;

            // Getting new outlet
            int lekoutoutlet = this->lake_network[lakoutid].get_lake_outlet();
            std::cout << lekoutoutlet << "||" << outlet << std::endl;

            // Chekcing if the parent lake has been ingested by the old one
            if (lekoutoutlet < 0)
            {
              // No lake, stopping the processing of this node here
              outlet = lekoutoutlet;
              return;
            }
            else
            {
              //It outlets in another lake, I change the referential to this new lake
              lakeid = lakoutid;
              outlet = lekoutoutlet;
              lakoutid = node_in_lake[outlet];
              if(lakoutid >=0)
                if(this->lake_network[lakoutid].get_parent_lake() >= 0)
                  lakoutid = this->lake_network[lakoutid].get_parent_lake();
            }
          }

          // std::cout << "OUTLET 0.2 is " << outlet << std::endl;


          std::vector<int> node_to_reprocess, reproc_in_lakes;
          std::vector<int> nodes_to_recompute_neighbors_at_the_end;


          xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];
          

          // Calling the function that does an adpated graph traversal to detect downstream nodes needing reprocessing
          // It separates the affected nodes into the normal nodes requiring easy deprocessing/reprocessing
          // and lake node requiring merging into a single refilling event
          if(outlet>=0)
          {
            this->find_nodes_to_reprocess(outlet, is_processed, node_to_reprocess, reproc_in_lakes,nodes_to_recompute_neighbors_at_the_end, lakeid);
            // std::cout << "OUTLET 0.3 is " << outlet << std::endl;



            // std::cout << std::endl;
            // std::cout << "Reprocessing::";
            // A bit of optimisation here: finding the ID in the stack of the nodes to reprocess in order to reprocess them in the right order
            int min_stackID = std::numeric_limits<int>::max(), max_stack_id = -1;
            std::unordered_set<int> checker;
            for(auto newnode:node_to_reprocess)
            {

              int tid = this->graph.get_index_MF_stack_at_i(newnode);

              if(tid<min_stackID)
                min_stackID = tid;

              if(tid>max_stack_id)
                max_stack_id = tid;

              // the checker set helps me keeping track of which nodes to process
              checker.insert(newnode);
            }

            // Saving the fluxes pre correction in the lakes, in order to be able to correct it
            for(auto lanode:reproc_in_lakes)
            {


              int tlakeid = this->node_in_lake[lanode];
              if(tlakeid>=0)
              {
                if(this->lake_network[tlakeid].get_parent_lake() >= 0)
                  tlakeid = this->lake_network[tlakeid].get_parent_lake();

                if(lakeinQ[tlakeid] == true)
                {
                  lake_water_corrector[tlakeid] +=  this->chonk_network[lanode].get_water_flux();
                  lake_sed_corrector[tlakeid] += this->chonk_network[lanode].get_sediment_flux();
                }
                else
                {
                  lakeinQ[tlakeid] = true;
                  lake_to_process.push_back(tlakeid);
                
                  lake_water_corrector[tlakeid] =  this->chonk_network[lanode].get_water_flux();
                  lake_sed_corrector[tlakeid] = this->chonk_network[lanode].get_sediment_flux();
                }
                
              }
            }


            // std::cout << "OUTLET 0.4 is " << outlet << std::endl;
            // Cancelling the fluxes from DS to US
            for(int i = max_stack_id; i >= min_stackID; i--)
            {
              // Going through the stack and gathering node
              int inode = this->graph.get_MF_stack_at_i(i);

              // no cancelling of outlet fluxes, already done before
              if(inode == outlet)
                continue;
          
              // only reprocessing the noe if is in the list of course          
              if(checker.find(inode) != checker.end())
              {
                // std::cout << "Cancelling fluxes on " << inode << std::endl;
                // cancelling the fluxes
                this->chonk_network[inode].cancel_split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->timestep);
                // Cancelling the prefluxes (will be readded anyway)
                this->cancel_fluxes_before_moving_prep(this->chonk_network[inode], this->label_array[inode]);
                // I want to reinitialise moving preparation in case the dynamic state changed
                this->chonk_network[inode].reinitialise_moving_prep();
                // because I am reprocessing these nodes, I need to reinitialise their deposition fluxes and erosion fluxes too
                // But not the sediment fluxes, as some sed or water might remain from older thingies
                this->chonk_network[inode].set_erosion_flux_undifferentiated(0.);
                this->chonk_network[inode].set_erosion_flux_only_sediments(0.);
                this->chonk_network[inode].set_erosion_flux_only_bedrock(0.);
                this->chonk_network[inode].set_deposition_flux(0.);
                this->chonk_network[inode].set_sediment_creation_flux(0.);
              }

            }

            // std::cout << "OUTLET 0.5 is " << outlet << std::endl;
            // Last but not least, dealing with the outlet
            // this->cancel_fluxes_before_moving_prep(this->chonk_network[outlet], this->label_array[outlet]);


            chonk& this_chonk = this->lake_network[lakeid].get_outletting_chonk();
            std::vector<int> rec = this_chonk.get_chonk_receivers_copy();
            std::vector<double> slope2rec =  this_chonk.get_chonk_slope_to_recs_copy();
            // water weights
            std::vector<double> wwf = this_chonk.get_chonk_water_weight_copy();
            std::vector<double> wsf = this_chonk.get_chonk_sediment_weight_copy();
            this->chonk_network[outlet].reinitialise_moving_prep();

            this->chonk_network[outlet].external_moving_prep(rec,wwf,wsf,slope2rec);
            this->chonk_network[outlet].set_water_flux(this_chonk.get_water_flux());
            // std::cout << outlet << "<----" << std::endl;

            // std::cout << "OUTLET 0.6 is " << outlet << std::endl;
            std::vector<double> oatlab = this_chonk.get_other_attribute_array("label_tracker");
            // std::cout << "OUTLET 0.61 is " << outlet << std::endl;
            this->chonk_network[outlet].set_sediment_flux(this_chonk.get_sediment_flux(), oatlab);
            // std::cout << "OUTLET 0.62 is " << outlet << std::endl;
            
            this->chonk_network[outlet].set_sediment_creation_flux(0.);
            // std::cout << "OUTLET 0.63 is " << outlet << std::endl;
            this->chonk_network[outlet].set_erosion_flux_undifferentiated(0.);
            // std::cout << "OUTLET 0.64 is " << outlet << std::endl;
            this->chonk_network[outlet].set_erosion_flux_only_sediments(0.);
            // std::cout << "OUTLET 0.65 is " << outlet << std::endl;
            this->chonk_network[outlet].set_erosion_flux_only_bedrock(0.);
            // std::cout << "OUTLET 0.66 is " << outlet << std::endl;
            this->chonk_network[outlet].set_deposition_flux(0.);
            // std::cout << "bo";
            // if(active_nodes[outlet]==1)
            this->process_node_nolake_for_sure(outlet, is_processed, lake_incrementor, underfilled_lake, inctive_nodes, cellarea, surface_elevation, false, false);
            // std::cout << "ris" << "|"; 
            // std::cout << "OUTLET 0.67 is " << outlet << std::endl;


            // Reprocessing the fluxes from US to DS where I am sure there is no lake
            for(int i = min_stackID; i <= max_stack_id; i++)
            {
              // going through the stack
              int inode = this->graph.get_MF_stack_at_i(i);
              if(checker.find(inode) != checker.end())
              {
                // if nodes are in the ones to reprocess, I proceed
                underfilled_lake++;
                // i know they are not in lakes by definition so I call a lighter function, and avoid too much recusrion which tend to break the code somehows
            // std::cout << "ka";
              // std::cout << "reprocessing :: " << inode << std::endl;
                this->process_node_nolake_for_sure(inode, is_processed, lake_incrementor, underfilled_lake, inctive_nodes, cellarea, surface_elevation, true, true);
            // std::cout << "ren" << "|";;
              }
            }
          }
          // std::cout << "OUTLET 0.7 is " << outlet << std::endl;
          // i recomputed the receivers of some nodes, I need to reinitialise it here
          // This was to avoid nodes giving back water to some lakes
          this->graph.compute_receveivers_and_donors( temp_bool, surface_elevation, nodes_to_recompute_neighbors_at_the_end );


          // If no lake to reprocess, well no lake to reprocess
          if(reproc_in_lakes.size() == 0 && lake_to_process.size() == 0)
          {
            return;
          }

          // Preparing to reprocess lakes: keys of these maps are lakeid (index in the lake_network)
          // and values are volume of water, of sediments and the entry node (does not matter which one as long as it is in the lake)
          for(auto& lanode:reproc_in_lakes)
          {
            //Getting the lake id 
            int tlakeid = node_in_lake[lanode];
            // if the lake has a parent, I use it has it has ingested the current lake
            if(this->lake_network[tlakeid].get_parent_lake() >= 0)
              tlakeid = this->lake_network[tlakeid].get_parent_lake();

            if(lake_water.count(tlakeid) == 0)
            {
              lake_water[tlakeid] = 0.;
              lake_sed[tlakeid] = 0.;
              chonk tchonk4lake(-1,-1,false);
              tchonk4lake.initialise_local_label_tracker_in_sediment_flux(this->n_labels);
              lake_chonks[tlakeid] = tchonk4lake;
            }


          }

          // std::cout << "OUTLET 0.8 is " << outlet << std::endl;
          for(auto& lanode:reproc_in_lakes)
          {
            //Getting the lake id 
            int tlakeid = node_in_lake[lanode];
            // if the lake has a parent, I use it has it has ingested the current lake
            if(this->lake_network[tlakeid].get_parent_lake() >= 0)
              tlakeid = this->lake_network[tlakeid].get_parent_lake();

            // adding to the lake the delta water (the post-lake reprocessing will always ADD more water as it happens when more water overflows)
            lake_water[tlakeid] += this->chonk_network[lanode].get_water_flux();
            lake_sed[tlakeid] += this->chonk_network[lanode].get_sediment_flux();

            std::vector<double> label_prop = this->chonk_network[lanode].get_other_attribute_array("label_tracker");
            lake_node_entry[tlakeid] = lanode;
            lake_chonks[tlakeid].add_to_sediment_flux(this->chonk_network[lanode].get_sediment_flux(), label_prop);

          }
          // Finally for each nodes in the system
          // for(auto& toproclake:lake_water)
          // {
          // int tlakeid

          // getting the lakeid
          int tlakeid = lake_to_process.front();
          lake_to_process.pop_front();
          if(lakeinQ[tlakeid] == false)
            throw std::runtime_error("ARG should not happen in lake sover");
          lakeinQ[tlakeid] = false;
          // std::cout << "OUTLET 0.9 is " << outlet << " next lake is " << tlakeid << std::endl;

          // the water flux to add (will be translated to vulume in the next function)
          double water_to_add = ( lake_water[tlakeid] - lake_water_corrector[tlakeid] ) * this->timestep;
          // Sediment flux to add
          double sed_to_add = lake_sed[tlakeid] - lake_sed_corrector[tlakeid];
          // entry node
          int node_id = lake_node_entry[tlakeid];

          auto baluf = lake_chonks[tlakeid].get_other_attribute_array("label_tracker");

          // saving the ingested lakes by this girl
          std::vector<int> prelakingestor = this->lake_network[tlakeid].get_ingested_lakes();

          this->lake_network[tlakeid].pour_sediment_into_lake(sed_to_add,baluf);
          this->lake_network[tlakeid].pour_water_in_lake(water_to_add,node_id, node_in_lake, is_processed, active_nodes, lake_network, surface_elevation, 
            this->graph,this->io_double["dx"]*this->io_double["dy"], this->timestep, this->chonk_network, this->Ql_out);
          
          std::vector<int> postlakingestor = this->lake_network[tlakeid].get_ingested_lakes();

          lakeid = tlakeid;

          // std::cout << "OUTLET 0.10 is " << outlet << std::endl;
          bool need_erasal = true;
          if(postlakingestor.size() > prelakingestor.size())
          {
            lake_water[tlakeid] = 0;
            lake_sed[tlakeid] = 0;
            lake_node_entry[tlakeid] = 0;
            chonk tchonk4lake(-1,-1,false);
            tchonk4lake.initialise_local_label_tracker_in_sediment_flux(this->n_labels);
            lake_chonks[tlakeid] = tchonk4lake;
            lake_sed_corrector[tlakeid] = 0;
            lake_water_corrector[tlakeid] = 0;
            
            for (auto la:postlakingestor)
            {
              if(std::find(prelakingestor.begin(), prelakingestor.end(), la) != prelakingestor.end())
                continue;
              if(lakeinQ[la] == false)
                continue;

              need_erasal = false;

              lake_water[tlakeid] += lake_water[la];
              lake_sed[tlakeid] += lake_sed[la];
              lake_node_entry[tlakeid] += lake_node_entry[la];
              lake_sed_corrector[tlakeid] += lake_sed_corrector[la];
              lake_water_corrector[tlakeid] += lake_water_corrector[la];
              lake_chonks[tlakeid].add_to_sediment_flux(lake_sed[la], lake_chonks[la].get_other_attribute_array("label_tracker"));

              lake_water.erase(la);
              lake_sed.erase(la);
              lake_node_entry.erase(la);
              lake_chonks.erase(la);
              lake_water_corrector.erase(la);
              lake_sed_corrector.erase(la);
              lake_chonks.erase(la);

              lake_to_process.erase(std::remove(lake_to_process.begin(), lake_to_process.end(), la), lake_to_process.end());


            }
          }

          if(need_erasal)
          {
            lake_water.erase(tlakeid);
            lake_sed.erase(tlakeid);
            lake_node_entry.erase(tlakeid);
            lake_chonks.erase(tlakeid);
            lake_water_corrector.erase(tlakeid);
            lake_sed_corrector.erase(tlakeid);
            lake_chonks.erase(tlakeid);
          }
          else
          {
            lake_to_process.push_back(tlakeid);
            lakeinQ[tlakeid] = true;
          }

          // std::cout << "OUTLET 1.5 is " << outlet << std::endl;





          // getting the outlet node of my lake. aOOOOOOOOOOOO . will be -9999 if there is no
          outlet = this->lake_network[lakeid].get_lake_outlet();      
          // std::cout << "OUTLET 2 is " << outlet << " from lake " << lakeid << std::endl;
          // OK first step is to check wether this outlet is a lake itself or not
          if(outlet >= 0)
          {
            lakoutid = node_in_lake[outlet];
            if(lakoutid>=0)
              if(this->lake_network[lakoutid].get_parent_lake() >= 0)
                lakoutid = this->lake_network[lakoutid].get_parent_lake();
          } 
          // std::cout << "OUTLET 3 is " << outlet << std::endl;

          

            // // Simulating the refilling of a lake by reformatting the chonk
            // this->chonk_network[node_id].reset();
            // this->chonk_network[node_id].initialise_local_label_tracker_in_sediment_flux(this->n_labels);
            // this->chonk_network[node_id].set_water_flux(water_to_add);
            // auto baluf = lake_chonks[toproclake.first].get_other_attribute_array("label_tracker");

            // // TODO:: SORT THAT...
            // if(sed_to_add <0)
            //   sed_to_add = 0;
            // this->chonk_network[node_id].set_sediment_flux(sed_to_add, baluf);

            // std::cout << "reproc in lake::"; 
            // this->process_node(node_id, is_processed, lake_incrementor, underfilled_lake, inctive_nodes, cellarea, surface_elevation,false);
            // std::cout << "GABUL_DONE";
          // }

          // return;
        }


      }
      else
        return;

      // end of while loop of reprocessing
      

      
    }
    else
    {
      if(inctive_nodes[node] == 1 && this->graph.is_depression(node))
      {
        int next_node = this->graph.get_Srec(node);
        if(this->graph.is_depression(next_node) == false)
          next_node = this->graph.get_Srec(next_node);

        this->chonk_network[next_node].add_to_water_flux(this->chonk_network[node].get_water_flux());
        if(is_processed[next_node] == true && inctive_nodes[next_node] )
          throw std::runtime_error("FATAL_ERROR::NG24, node " + std::to_string(node) + " gives water to " + std::to_string(next_node) + " but is processed already");
  
        // node = next_node;
        is_processed[node] = true;
        // goto nolake;
      }
      else
        goto nolake;
      return;
    }
    nolake:
    

    // first step is to apply the right move method, to prepare the chonk to move
    if(need_move_prep)
      this->manage_move_prep(this->chonk_network[node]);


    // Fluxes after moving prep are active fluxes such as erosion or other thingies
    this->manage_fluxes_after_moving_prep(this->chonk_network[node],this->label_array[node]);
    // Apply the changes and propagate the fluxes downstream
    this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->io_double_array["surface_elevation_tp1"], io_double_array["sed_height_tp1"], this->timestep);
}

void ModelRunner::process_node_nolake_for_sure(int& node, std::vector<bool>& is_processed, int& lake_incrementor, int& underfilled_lake,
  xt::pytensor<int,1>& inctive_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep, bool need_flux_before_move)
{
    is_processed[node] = true;
    if(need_flux_before_move)
      this->manage_fluxes_before_moving_prep(this->chonk_network[node], this->label_array[node]);
    // first step is to apply the right move method, to prepare the chonk to move
    if(need_move_prep)
      this->manage_move_prep(this->chonk_network[node]);
  
    this->manage_fluxes_after_moving_prep(this->chonk_network[node],this->label_array[node]);

    this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->io_double_array["surface_elevation_tp1"], io_double_array["sed_height_tp1"], this->timestep);
}


void ModelRunner::finalise()
{
  // Finilising the timestep by applying the changes to the thingy
  // First gathering all the aliases 
  xt::pytensor<double,1>& surface_elevation_tp1 = this->io_double_array["surface_elevation_tp1"];
  xt::pytensor<double,1>& surface_elevation = this->io_double_array["surface_elevation"];
  xt::pytensor<double,1>& sed_height_tp1 = this->io_double_array["sed_height_tp1"];
  xt::pytensor<double,1> tlake_depth = xt::zeros<double>({size_t(this->io_int["n_elements"])});


  // First dealing with lake deposition:
   // std::cout <<"1" << std::endl;

  for(auto& loch:this->lake_network)
  {
    if(loch.get_parent_lake()>=0)
      continue;
    loch.drape_deposition_flux_to_chonks(this->chonk_network, surface_elevation, this->timestep);
  }
   // std::cout <<"2" << std::endl;
  // Iterating through all nodes
  for(int i=0; i< this->io_int["n_elements"]; i++)
  {
   // std::cout <<"3.1" << std::endl;
    // Getting the current chonk
    chonk& tchonk = this->chonk_network[i];
    auto this_lab = tchonk.get_other_attribute_array("label_tracker");
    for(auto LAB:this_lab)
      if(std::isfinite(LAB) ==false)
        std::cout << LAB << " << naninf for sedflux" << std::endl;

    // getting the lake stuffs
    if(node_in_lake[i]>=0)
    {
      int lakeid = node_in_lake[i];
      double tdepth = lake_network[lakeid].get_lake_depth_at_node(i,node_in_lake);
      tlake_depth[i] = tdepth;
    }
    else
    {
      tlake_depth[i] = 0;
    }



    // First applying the specific erosion flux
    // std::cout << surface_elevation_tp1[i] << "||";
    surface_elevation_tp1[i] -= tchonk.get_erosion_flux_only_bedrock() * timestep;
    // std::cout << surface_elevation_tp1[i] << "||";

    // Applying elevation changes from the sediments
    double sedcrea = tchonk.get_sediment_creation_flux() * timestep;

    if(std::isfinite(sedcrea) == false)
      throw std::runtime_error("NAN sedcrea finalisation not possible yo");
   // std::cout <<"3.2" << std::endl;

    // std::cout << "a" << std::endl;
    // std::cout << sedcrea << "|" << tchonk.get_erosion_flux_only_sediments() * timestep  << "||";
    if(sedcrea + sed_height_tp1[i] < 0)
    {
      std::cout << "happens??" <<sedcrea << "||" << sed_height_tp1[i] << "||" << this->node_in_lake[i] << std::endl;
      surface_elevation_tp1[i] -= sed_height_tp1[i];
      sed_height_tp1[i] = 0.;
      sed_prop_by_label[i] = std::vector<std::vector<double> >();
    } 
    else
    {
    // std::cout << "b" << std::endl;
     this->add_to_sediment_tracking(i, sedcrea, this_lab, sed_height_tp1[i]);
    // std::cout << "c" << std::endl;

     surface_elevation_tp1[i] += sedcrea;
     sed_height_tp1[i] += sedcrea;
    }

   // std::cout <<"3.3" << std::endl;

    // std::cout << surface_elevation_tp1[i] << std::endl;;

    // double to_remove = tchonk.get_erosion_flux_undifferentiated();
    double tadd = tchonk.get_erosion_flux_undifferentiated() * timestep;
    // std::cout << tadd << "|";
    // std::cout << " Eund is " << tadd << std::endl;
    if(sed_height_tp1[i] < 0)
      tadd = tadd + sed_height_tp1[i];
    // std::cout << "C::" << i << "||" << tadd << "||" << sed_height_tp1[i] << std::endl;
   // std::cout <<"3.4" << std::endl;

    this->add_to_sediment_tracking(i, -1*tadd, this_lab, sed_height_tp1[i]);
    // std::cout << "D::" << i << "||" << tadd << "||" << sed_height_tp1[i] << std::endl;
   // std::cout <<"3.5" << std::endl;

    surface_elevation_tp1[i] -= tadd;
    sed_height_tp1[i] -= tadd;


    if(sed_height_tp1[i]<0)
    {
      // to_remove = std::abs(sed_height_tp1[i]);
      sed_height_tp1[i] = 0;
      sed_prop_by_label[i] = std::vector<std::vector<double> >();
      // surface_elevation_tp1[i] += to_remove;
    }


  }

  this->Ql_out += xt::sum(tlake_depth - this->io_double_array["lake_depth"])() * this->io_double["dx"] * this->io_double["dy"] / this->timestep;
   // std::cout <<"3.6" << std::endl;
  this->io_double_array["lake_depth"] = tlake_depth;
   // std::cout <<"3.7" << std::endl;

  xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];

  for(int i =0; i<this->io_int["n_elements"]; i++)
  {
    if(active_nodes[i] == 0)
    {
      this->Qw_out += this->chonk_network[i].get_water_flux();
    }
  }
}

void ModelRunner::add_to_sediment_tracking(int index, double height, std::vector<double> label_prop, double sed_depth_here)
{
  // trying to remove sediments but no sediments are here
  // Nothing happens then
  if(height == 0)
    return;

  if(height<0 && is_there_sed_here[index] == false)
  {
    return;
  }

  // No sediments previously there -> creating boxes of sediment here
  if(is_there_sed_here[index] == false)
  {
    if(sed_prop_by_label[index].size()>0 )
      throw std::runtime_error("Unexpected Behavior 56");

    int n_strata = std::ceil(std::abs(height / this->io_double["depths_res_sed_proportions"]));
    for(int i = 0; i<n_strata; i++)
    {
      sed_prop_by_label[index].push_back(label_prop);
    }
  // std::cout << "BD" << std::endl;
    if(sed_prop_by_label[index].size() == 0)
    {
      std::cout << index << "||" << node_in_lake[index] << "||" << n_strata << "||" << height << std::endl;
      throw std::runtime_error("Unexpected behaviour SEDBUTNOSED");
    }
    is_there_sed_here[index] = true;
    return;
  }

  // std::cout << "BE" << std::endl;
  for(auto LAB:label_prop)
    if(std::isfinite(LAB) == false)
      std::cout << "NANLAB at label_prop::add_to_sediment_tracking_start" << std::endl;


  // Sediments already in there, getting more complex
  // Getting the proportion of the last box filled and the number of boxes
  double n_boxes = sed_depth_here/this->io_double["depths_res_sed_proportions"];
  double n_strata_already_there;
  double prop_box_filled = std::modf(n_boxes, &n_strata_already_there);
  double n_boxes_to_fill = std::abs(height/this->io_double["depths_res_sed_proportions"]);
  int current_box = int(sed_prop_by_label[index].size()) - 1;

  //TO DO BORIS:: CURRENT BOX SHOULD NEVER BE -1 HERE, WHY IS IT??
  if(current_box < 0)
  {
    is_there_sed_here[index] = false;
    return;
  }


  for(auto stuff:sed_prop_by_label[index][current_box])
    if(std::isfinite(stuff) == false)
      throw std::runtime_error("Unexpected behaviour 463");



  // n_boxes to add or remove
  int delta_boxes = std::floor(n_boxes_to_fill);
  // std::cout << "DB::" << delta_boxes << std::endl;
  // Sediment addition case
  if(height > 0)
  {
    // Calculating the number of boxes to add
    double comparator = n_boxes_to_fill;
    if(n_boxes_to_fill>1)
      comparator = 1;

    for(auto ugh:sed_prop_by_label[index][current_box])
    if(std::isfinite(ugh) != true)
    {
      std::cout << "nan before" << std::endl;
    }
    for(auto ugh:label_prop)
    if(std::isfinite(ugh) != true)
    {
      std::cout << "nan before" << std::endl;
    }

    // filling the current box with the mixed proportions of labels
    sed_prop_by_label[index][current_box] = mix_two_proportions(prop_box_filled, sed_prop_by_label[index][current_box], (comparator - prop_box_filled), label_prop);

    for(auto ugh:sed_prop_by_label[index][current_box])
    if(std::isfinite(ugh) == false)
    {
      throw std::runtime_error("Arg, nan sedafter");
    }

    // adding the boxes
    if(delta_boxes> 0)
    {
      for(int i=0; i<delta_boxes; i++)
        sed_prop_by_label[index].push_back(label_prop);
    }

  }
  else
  {

    // sediments to remove, easier
    if(delta_boxes > 0)
    {

      for(int i=0; i<delta_boxes; i++)
      {
        // NEED TO CHECK WHY IS THIS HAPPENING HERE!!!! IT SHOULD NOT TRY TO POP BACK IF THERE IS NOTHING TO POP BACK 
        if(sed_prop_by_label[index].size() > 0)
          sed_prop_by_label[index].pop_back();
      }
    }

    
  }
  //Done??
  if(sed_prop_by_label[index].size() == 0)
      is_there_sed_here[index] = false;

  int curate = int(sed_prop_by_label[index].size()) -1;
  if(curate>=0)
  {
    for(auto stuff:sed_prop_by_label[index][curate])
      if(std::isfinite(stuff) == false)
        throw std::runtime_error("Unexpected behaviour 465");
  }
}



xt::pytensor<double,2> ModelRunner::get_superficial_layer_sediment_prop()
{
  xt::pytensor<double,2> output = xt::zeros<double>({n_labels,this->io_int["n_elements"]});

  for (int lab = 0; lab < this->n_labels; lab++)
  {
    for(size_t i = 0; i < this->io_int["n_elements"]; i++)
    {
      if(is_there_sed_here[i] && sed_prop_by_label[int(i)].size() != 0)
      {
        output(lab,i) = sed_prop_by_label[int(i)][sed_prop_by_label[int(i)].size() - 1][lab];
      }
      else
        output(lab,i) = 0;
    }
  }
  return output;
}

void ModelRunner::find_nodes_to_reprocess(int start, std::vector<bool>& is_processed, std::vector<int>& nodes_to_reprocess, std::vector<int>& nodes_to_relake,
  std::vector<int>& nodes_to_recompute_neighbors_at_the_end, int lake_to_avoid)
{
  // Alright Now is the time to finally comment this function as it seems to be the reason of mass balance and sediment nan bug :(
  // I have been dreading this moment as this function has been written in one go alongside all the lake reprocessing routines

  // this was a debugging statement marking the start of the function to locate where it first segfaulted
  // std::cout << "start" << std::endl;

  // Some magic to keep a nice balance between cpu and memory optimisation: I initialise a vector assuming I will probably not need to reprocess > 1/4 of the nodes
  // if I eceed this size, I use push_back whihc is slightly slower
  int tsize = int(round(this->io_int["n_elements"]/4));
  // I am initialising various ID to insert and read nodes in this vector
  int insert_id_trav = 0,insert_id_lake = 0,reading_id = -1, insert_id_proc = 1;

  // traversal is my vector storing the nodes to reprocess in a BFS search-like algorithm. I think.
  std::vector<int> traversal;traversal.reserve(tsize);
  // the Q storing the nodes to process
  std::vector<int> tQ;tQ.reserve(tsize);
  // the lake IDs to (re)process
  std::vector<int> travlake;travlake.reserve(tsize);
  // Sets to keep track of nodes already processed in this function. Again, should be more efficient than a full vector as the number of nodes whshould be relatively low
  std::set<int> is_in_Q,is_in_lake;

  // Starting the processing with the starting node ofc
  tQ.emplace_back(start);
  is_in_Q.insert(start);

  // const bool is_in = container.find(element) != container.end();
  // Iterating til' I decide it
  while(true)
  {
    // Reading the next node in the Q
    // if first time, -1 +1 =0 so it works aye
    reading_id++;
    // if I reach the end of me Q, tis the end
    // It works because I reserve some space in it, so I increase its CAPACITY but not its SIZE which grows with emplace_back
    if(reading_id >= tQ.size())
      break;

    // getting the node
    int this_node = tQ[reading_id];
    bool recomputed_neight = false;
    // std::cout << "|Q_NODE:" << this_node;
    // if node is not start, I put it into the traversal, cause it has already been checked
    if(this_node != start)
    {  
      if(insert_id_trav < tsize)
      {
        traversal.emplace_back(this_node);
        insert_id_trav++;
      }
      else
        traversal.push_back(this_node);
    }

    std::vector<int> golog = this->graph.get_MF_receivers_at_node(this_node);
    // iterating through all his neighbors
    for(auto node:golog)
    {
     
      // if the neightbor has not been processed originally no reproc
      if(is_processed[node] == false)
        continue;

      // if the node is already in Q skip
      if(is_in_Q.find(node) != is_in_Q.end())
        continue;
     
      // getting the lake id
      int this_lake_id = node_in_lake[node];

      // If this node is in a lake, but not processed yet by this function and not the starting one
      if(this_lake_id >= 0 && is_in_lake.find(node) == is_in_lake.end()) // && this_node != start)
      {

        // Checking and avoiding dedundancy-> if is in the lake to avoid -> rework the receivers of the outlet and process it
        if(this_lake_id == lake_to_avoid || this->lake_network[this_lake_id].get_parent_lake() == lake_to_avoid)
        {
          std::vector<int> new_rec;
          std::vector<int> datiznogoud = this->graph.get_MF_receivers_at_node(this_node);
          for(auto recnode:datiznogoud)
          {
            if(recnode != node)
              new_rec.push_back(recnode);
          }
          // std::cout << "IT HAPPENS" << std::endl;
          this->graph.update_receivers_at_node(this_node, new_rec);
          recomputed_neight = true;
          continue;
        }

        // if it passes this test, then the node is in a lake to reprocess

        is_in_lake.insert(node);
        if(insert_id_lake < tsize )
        {

          travlake.emplace_back(node);
          insert_id_lake++;
        }
        else
          travlake.push_back(node);

        // I do not want to reprocess this node, so I am putting "in the Q" so that it gets ignored now. I know I have to reprocess it
        is_in_Q.insert(node);
        
        continue;
      }

      // if is in lake to avoid at start, I stop here
      if(node == start && (this_lake_id == lake_to_avoid || this->lake_network[this_lake_id].get_parent_lake() == lake_to_avoid) )
        continue;

      // otherwise, I'll need to check this node and put it in teh traversal yolo
      is_in_Q.insert(node);
      if(insert_id_proc < tsize )
      {
        tQ.emplace_back(node);
        insert_id_proc++;
      }
      else
        tQ.push_back(node);

    }
    if(recomputed_neight)
      nodes_to_recompute_neighbors_at_the_end.push_back(this_node);

  }


  // for (auto bugh:traversal)
  //   if(node_in_lake[bugh] >= 0)
  //     throw std::runtime_error("ERROAR! this nor should not be in this traversal");
  // if(travlake.size()>0)
  //   std::cout << "{" << travlake.size() << "}";

  nodes_to_reprocess = std::move(traversal);
  nodes_to_relake = std::move(travlake);




  // // std::cout << "end" << std::endl;
  // std::map<int,int> node_counter;
  // for(auto n:nodes_to_reprocess)
  // {
  //   node_counter[n] = 0;
  // }
  // for(auto n:nodes_to_reprocess)
  // {
  //   node_counter[n] += 1;
  //   if(node_counter[n] > 1)
  //     throw std::runtime_error("Doubled reprocessing error");
  // }

}



void ModelRunner::manage_fluxes_before_moving_prep(chonk& this_chonk, int label_id)
{

  for(auto method:this->ordered_flux_methods)
  {
    if(method == "move")
      break;
    int this_case = intcorrespondance[method];

    switch(this_case)
    {
      case 5:
        this_chonk.inplace_only_drainage_area(this->io_double["dx"], this->io_double["dy"]);
        this->Qw_in += this->io_double["dx"]* this->io_double["dy"];
        break;
      case 6:
        this_chonk.inplace_precipitation_discharge(this->io_double["dx"], this->io_double["dy"],this->io_double_array["precipitation"]);
        this->Qw_in += this->io_double_array["precipitation"][this_chonk.get_current_location()] * this->io_double["dx"]* this->io_double["dy"];
        break;
      case 7:
        this_chonk.inplace_infiltration(this->io_double["dx"], this->io_double["dy"], this->io_double_array["infiltration"]);
        this->Qw_out += this->io_double_array["infiltration"][this_chonk.get_current_location()] * this->io_double["dx"]* this->io_double["dy"];
        break;
    }

  }
}

void ModelRunner::cancel_fluxes_before_moving_prep(chonk& this_chonk, int label_id)
{
  for(auto method:this->ordered_flux_methods)
  {
    if(method == "move")
      break;
    int this_case = intcorrespondance[method];

    switch(this_case)
    {
      case 5:
        this_chonk.cancel_inplace_only_drainage_area(this->io_double["dx"], this->io_double["dy"]);
        this->Qw_in -= this->io_double["dx"]* this->io_double["dy"];
        break;
      case 6:
        this_chonk.cancel_inplace_precipitation_discharge(this->io_double["dx"], this->io_double["dy"],this->io_double_array["precipitation"]);
        this->Qw_in -= this->io_double_array["precipitation"][this_chonk.get_current_location()] * this->io_double["dx"]* this->io_double["dy"];
        break;
      case 7:
        this_chonk.cancel_inplace_infiltration(this->io_double["dx"], this->io_double["dy"], this->io_double_array["infiltration"]);
        this->Qw_out -= this->io_double_array["infiltration"][this_chonk.get_current_location()] * this->io_double["dx"]* this->io_double["dy"];
        break;
    }

  }
}



void ModelRunner::manage_move_prep(chonk& this_chonk)
{
  int this_case = intcorrespondance[this->move_method];

  switch(this_case)
  {
    case 2:
      this_chonk.move_to_steepest_descent(this->graph, this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["surface_elevation"],  this->io_double_array["surface_elevation_tp1"], this->io_double["dx"], this->io_double["dy"], chonk_network);
      break;
    case 3:
      this_chonk.move_MF_from_fastscapelib(this->graph, this->io_double_array2d["external_weigths_water"], this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["surface_elevation"],  this->io_double_array["surface_elevation_tp1"], this->io_double["dx"], this->io_double["dy"], chonk_network);
      break;
    case 4:
      this_chonk.move_MF_from_fastscapelib_threshold_SF(this->graph, this->io_double["threshold_single_flow"], this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["surface_elevation"],  this->io_double_array["surface_elevation_tp1"], this->io_double["dx"], this->io_double["dy"], chonk_network);
      break;
      
    default:
      std::cout << "WARNING::move method name unrecognised, not sure what will happen now, probably crash" << std::endl;
  }
}

void ModelRunner::manage_fluxes_after_moving_prep(chonk& this_chonk, int label_id)
{
  bool has_moved = false;
  for(auto method:this->ordered_flux_methods)
  {
    int index = this_chonk.get_current_location();
    
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
        this_chonk.active_simple_SPL(this->labelz_list_double["SPIL_n"][label_id], this->labelz_list_double["SPIL_m"][label_id], this->labelz_list_double["SPIL_K"][label_id], this->timestep, this->io_double["dx"], this->io_double["dy"], label_id);
        break;
      case 8:
        std::vector<double> these_sed_props(this->n_labels,0.);
        if(is_there_sed_here[index] && this->sed_prop_by_label[index].size()>0) // I SHOULD NOT NEED THE SECOND THING, WHY DO I FUTURE BORIS????
          these_sed_props = this->sed_prop_by_label[index][this->sed_prop_by_label[index].size() - 1];
      // if(this->node_in_lake[index] >=0)
      // {
      //   std::cout << "NODEINLAKEYO? " << index << "|" << this->node_in_lake[index] << "|" << this-> lake_network[this->node_in_lake[index]].get_lake_outlet() << std::endl;
      //   throw std::runtime_error("node in lake yo should not charlicise");
      // }
        this_chonk.charlie_I(this->labelz_list_double["SPIL_n"][label_id], this->labelz_list_double["SPIL_m"][label_id], this->labelz_list_double["CHARLIE_I_Kr"][label_id], 
  this->labelz_list_double["CHARLIE_I_Ks"][label_id],
  this->labelz_list_double["CHARLIE_I_dimless_roughness"][label_id], this->io_double_array["sed_height"][index], 
  this->labelz_list_double["CHARLIE_I_V"][label_id], 
  this->labelz_list_double["CHARLIE_I_dstar"][label_id], this->labelz_list_double["CHARLIE_I_threshold_incision"][label_id], 
  this->labelz_list_double["CHARLIE_I_threshold_entrainment"][label_id],
  label_id, these_sed_props, this->timestep,  this->io_double["dx"], this->io_double["dy"]);
        
        break;
    }
  }
  return;
}

// Initialise ad-hoc set of internal correspondance between process names and integer
// Again this is a small otpimisation that reduce the need to initialise and call maps for each nodes as maps are slower to access
void ModelRunner::initialise_intcorrespondance()
{
  intcorrespondance = std::map<std::string,int>();
  intcorrespondance["SPIL_Howard_Kerby_1984"] = 1;
  intcorrespondance["D8"] = 2;
  intcorrespondance["MF_fastscapelib"] = 3;
  intcorrespondance["MF_fastscapelib_threshold_SF"] = 4;
  intcorrespondance["drainage_area"] = 5;
  intcorrespondance["precipitation_discharge"] = 6;
  intcorrespondance["infiltration_discharge"] = 7;
  intcorrespondance["CHARLIE_I"] = 8;

}

void ModelRunner::prepare_label_to_list_for_processes()
{
  for(auto method:this->ordered_flux_methods)
  {
    int this_case = intcorrespondance[method];
    switch(this_case)
    {
      // SPIL_full_label
      case 1:
        labelz_list_double["SPIL_m"] = std::vector<double>();
        labelz_list_double["SPIL_n"] = std::vector<double>();
        labelz_list_double["SPIL_K"] = std::vector<double>();
        for(auto& tlab:labelz_list)
        {
          labelz_list_double["SPIL_m"].push_back(tlab.double_attributes["SPIL_m"]);
          labelz_list_double["SPIL_n"].push_back(tlab.double_attributes["SPIL_n"]);
          labelz_list_double["SPIL_K"].push_back(tlab.double_attributes["SPIL_K"]);
        }
        break;
      // Charlie the first
      case 8:
        labelz_list_double["SPIL_m"] = std::vector<double>();
        labelz_list_double["SPIL_n"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_Kr"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_Ks"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_V"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_dimless_roughness"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_dstar"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_threshold_incision"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_threshold_entrainment"] = std::vector<double>();
        for(auto& tlab:labelz_list)
        {
          labelz_list_double["SPIL_m"].push_back(tlab.double_attributes["SPIL_m"]);
          labelz_list_double["SPIL_n"].push_back(tlab.double_attributes["SPIL_n"]);
          labelz_list_double["CHARLIE_I_Kr"].push_back(tlab.double_attributes["CHARLIE_I_Kr"]);
          labelz_list_double["CHARLIE_I_Ks"].push_back(tlab.double_attributes["CHARLIE_I_Ks"]);
          labelz_list_double["CHARLIE_I_V"].push_back(tlab.double_attributes["CHARLIE_I_V"]);
          labelz_list_double["CHARLIE_I_dimless_roughness"].push_back(tlab.double_attributes["CHARLIE_I_dimless_roughness"]);
          labelz_list_double["CHARLIE_I_dstar"].push_back(tlab.double_attributes["CHARLIE_I_dstar"]);
          labelz_list_double["CHARLIE_I_threshold_incision"].push_back(tlab.double_attributes["CHARLIE_I_threshold_incision"]);
          labelz_list_double["CHARLIE_I_threshold_entrainment"].push_back(tlab.double_attributes["CHARLIE_I_threshold_entrainment"]);
        }
        break;
  
      // defaut case means the law has no correspondance so it does not do anything
      default: 
        break;
    }
  }

}

void ModelRunner::initialise_label_list(std::vector<labelz> these_labelz)
{
  this->labelz_list = these_labelz;
  this->n_labels = int(these_labelz.size());
  this->prepare_label_to_list_for_processes();

  // initialising the sediment tracking device
  is_there_sed_here = std::vector<bool>(this->io_int["n_elements"], false);
  sed_prop_by_label = std::map<int, std::vector<std::vector<double> > >() ;
}


//#################################################
//#################################################
//#################################################
//######### Solving depression ####################
//#################################################
//#################################################
//#################################################





// This function processes every existing lakes and check if the new topography has an outlet for them
// if it does , it empty what needs to be emptied at the outlet
// whatever remains after all of this is added to each nodes separatedly which simulates mass-balance
// POSSIBLE OPTIMISATION::Recreating lake objects, but I would need a bit of refactoring the Lake object. Will seee if this is very slow
void ModelRunner::process_inherited_water()
{
  for(auto& tlake:this->lake_network)
  {
    // getting node underwater
    std::vector<int>& unodes = tlake.get_lake_nodes();
    if(unodes.size() == 0 || tlake.get_parent_lake() >= 0)
      continue;


    // This following method was recalculating lake outlet with the new topography
    // // going through all the nodes and gathering the potenial outlets
    // // With the new topography, we can imagine a rare case whenre several breaches are opened in the lake within one timestep
    // // It is unlikely that they would happened at the same time in real life so I just decide to drain it to the steepest descent one assuming it would arrive first
    // // I have to say that this might be controversial, but also would only happen in very rare cases, even never to be fair
    // std::vector<int> outlets, outlets_origin;

    xt::pytensor<double,1>& surface_elevation = this->io_double_array["surface_elevation"];
    // xt::pytensor<double,1>& lake_depth = this->io_double_array["lake_depth"];

    // for(auto node:unodes)
    // {
    //     std::vector<int>& recs = graph.get_MF_receivers_at_node_no_rerouting(node);
    //     std::vector<int>& dons = graph.get_MF_donors_at_node(node);
    //     std::vector<int> neightbors;
    //     neightbors.reserve(neightbors.size() + dons.size());

    //     for (auto r:recs)
    //       neightbors.emplace_back(r);
    //     for (auto d:dons)
    //       neightbors.emplace_back(d);

    //   double elenode = surface_elevation[node] + tlake.get_lake_depth_at_node(node, node_in_lake);


    //   for(auto nenode:neightbors)
    //   {
    //     int this_lake_id = tlake.get_lake_id();
    //     if(this->node_in_lake[nenode] == this_lake_id)
    //       continue;
    //     double this_depth = 0.;

    //     if(this_lake_id >= 0)
    //       this_depth = this->lake_network[this_lake_id].get_lake_depth_at_node(nenode, node_in_lake);

    //     double tselev = this_depth + surface_elevation[nenode];
    //     if(tselev < elenode)
    //     {
    //       if(std::find(outlets.begin(), outlets.end(), nenode) != outlets.end())
    //         continue;
    //       outlets.push_back(nenode);
    //       outlets_origin.push_back(node);
    //     }
    //   }
    // }

    // if(outlets.size() > 0)
    // {
    //   double min_elev = std::numeric_limits<double>::max();
    //   int toutlet = -9999;
    //   int index_toutlet = -9999;
    //   for(size_t j=0; j<outlets.size(); j++)
    //   {
    //     int node = outlets[j];
    //     double this_elev = surface_elevation[node];
    //     // if(this->node_in_lake[nenode]>=0)
    //     //   this_elev += this->lake_network[this[node_in_lake[node]]].get_lake_depth_at_node(node, node_in_lake);
    //     if(this_elev < min_elev)
    //     {
    //       toutlet = node;
    //       index_toutlet = int(j);
    //       min_elev = this_elev;
    //     }
    //   }
    //   // Now I need to remove the water from the lake
    //   double volume_to_transfer = 0;
    //   min_elev = surface_elevation[outlets_origin[index_toutlet]];
    //   for(auto node:unodes)
    //   {
    //     double this_elevation  = surface_elevation[node] + tlake.get_lake_depth_at_node(node, node_in_lake);
    //     double delta_elevation = this_elevation - min_elev;

    //     // correcting the new remaining lake depth. I am not jsut applying the delta elevation because there can be some lake-bed elevation change too!
    //     lake_depth[node] = surface_elevation[node] -  min_elev;
    //     double tvolume = delta_elevation * this->io_double["dx"] * this->io_double["dy"];
    //     tlake.set_lake_depth_at_node(node, surface_elevation[node] -  min_elev);
    //     volume_to_transfer += tvolume;
    //   }

    //   tlake.set_lake_volume(tlake.get_lake_volume() - volume_to_transfer);
    //   this->chonk_network[toutlet].add_to_water_flux(volume_to_transfer/(timestep));
    //   this->Ql_in += volume_to_transfer/(this->timestep);



    // And finally refeeding the water to the rest
    // for(auto node:unodes)
    // {
    //   double mandragore = tlake.get_lake_depth_at_node(node, this->node_in_lake) * this->io_double["dx"] * this->io_double["dy"] / this->timestep;
    //   this->chonk_network[node].add_to_water_flux(mandragore );
    //   // this->Ql_in += mandragore;
    //   this->Ql_out -= mandragore;
    // }

    double minelev = std::numeric_limits<double>::max();
    int minnodor = -9999;
    double sumwat = 0;
    for( auto node:tlake.get_lake_nodes())
    {
      if(surface_elevation[node] < minelev)
      {
        minnodor = node;
        minelev = surface_elevation[node];
      }
      sumwat += tlake.get_lake_depth_at_node(node, node_in_lake) * this->io_double["dx"] * this->io_double["dy"] / this->timestep ;
    }

    this->chonk_network[minnodor].add_to_water_flux(sumwat);

  }

  // Done with reshuffling the water

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
    output[tchonk.get_current_location()] = tchonk.get_erosion_flux_undifferentiated() + tchonk.get_erosion_flux_only_sediments() + tchonk.get_erosion_flux_only_bedrock() ;
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_erosion_bedrock_only_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_erosion_flux_only_bedrock() ;
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_erosion_sed_only_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_erosion_flux_only_sediments() ;
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_sediment_creation_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_sediment_creation_flux() ;
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

std::vector<xt::pytensor<double,1> > ModelRunner::get_label_tracking_results()
{
  std::vector<xt::pytensor<double,1> > output;
  for(int i=0; i<this->n_labels; i++)
  {
    xt::pytensor<double,1> temp = xt::zeros_like(io_double_array["surface_elevation"]);
    output.push_back(temp);
  }

  for( int i=0; i< io_int["n_elements"];i++)
  {
    chonk& tchonk =this->chonk_network[i];
    for(int j=0; j<this->n_labels; j++)
    {
      output[j][i] = tchonk.get_other_attribute_array("label_tracker")[j];
    }
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
// Functions managing lakes #######################################################
//#################################################################################
//#################################################################################
//#################################################################################


void Lake::ingest_other_lake(
   Lake& other_lake,
   std::vector<int>& node_in_lake, 
   std::vector<bool>& is_in_queue,
   std::vector<Lake>& lake_network
   )
{
  // getting the attributes of the other lake
  std::vector<int>& these_nodes = other_lake.get_lake_nodes();
  std::vector<int>& these_node_in_queue = other_lake.get_lake_nodes_in_queue();
  std::unordered_map<int,double>& these_depths = other_lake.get_lake_depths();
  std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> >& this_PQ = other_lake.get_lake_priority_queue();
  // merging them into this lake while node forgetting to label them as visited and everything like that
  for(auto node:these_nodes)
  {
    this->nodes.push_back(node);
    node_in_lake[node] = this->lake_id;
  }

  for(auto node:these_node_in_queue)
  {
    this->node_in_queue.push_back(node);
    is_in_queue[node] = true;
  }

  this->depths.insert(these_depths.begin(), these_depths.end());

  this->n_nodes += other_lake.get_n_nodes();

  // Transferring the PQ (not ideal but meh...)
  while(this_PQ.empty() == false)
  {
    nodium this_nodium = this_PQ.top();
    is_in_queue[this_nodium.node] = true;
    this->depressionfiller.emplace(this_nodium);
    this_PQ.pop();
  }
  this->pour_sediment_into_lake(other_lake.get_volume_of_sediment(), other_lake.get_lake_lab_prop());

  this->volume += other_lake.get_lake_volume();

  // Deleting this lake and setting its parent lake
  int save_ID = other_lake.get_lake_id();


  Lake temp = Lake(save_ID);
  other_lake = temp;
  other_lake.set_parent_lake(this->lake_id);
  this->ingested_lakes.push_back(other_lake.get_lake_id());

  for(auto lid:other_lake.get_ingested_lakes())
    lake_network[lid].set_parent_lake(this->lake_id);


  return;
}

void Lake::pour_sediment_into_lake(double sediment_volume, std::vector<double> label_prop)
{
  if(this->volume_of_sediment<0)
    std::cout << "NEG BEFORE POURING SED" << std::endl;
  if(sediment_volume<0)
    std::cout << "NEG POURED" << std::endl;

  std::vector<double> coplaklab = this->lake_label_prop;
  std::vector<double> copaddlab = label_prop;
  
  if(this->lake_label_prop.size()>0)
  {
    // std::cout << "bo";
    for(auto lb:this->lake_label_prop)
      if(std::isfinite(lb) == false)
        std::cout << "NANINF in lake already" << std::endl;
    for(auto lb:label_prop)
      if(std::isfinite(lb) == false)
        std::cout << "NANINF coming in" << std::endl;
    if(this->volume_of_sediment > 0)
    {
      // std::cout << "go";
      this->lake_label_prop = mix_two_proportions(this->volume_of_sediment, this->lake_label_prop, sediment_volume, label_prop);
      // std::cout << "ris";
    }
    else
    {
      this->lake_label_prop = label_prop;
    }

      
    for(auto lb:this->lake_label_prop)
      if(std::isfinite(lb) == false)
      {
        std::cout << "NANINF in lake after::" << this->volume_of_sediment << "||" << sediment_volume << std::endl;
        for(auto yu:this->lake_label_prop)
          std::cout << yu << "|";
        std::cout << std::endl;
        for(auto yu:label_prop)
          std::cout << yu << "|";
        std::cout << std::endl;
        for(auto yu:copaddlab)
          std::cout << yu << "|";
        std::cout << std::endl;
        for(auto yu:coplaklab)
          std::cout << yu << "|";
        std::cout << std::endl;
        
        throw std::runtime_error("NANINF in lake sed pouring");
      }
    // std::cout << "ris";
  }
  else
    this->lake_label_prop = label_prop;

  this->volume_of_sediment += sediment_volume;
  if(this->volume_of_sediment<0)
  {
    std::cout << "NEG AFTER POURING SED " << sediment_volume << std::endl;
    // throw std::runtime_error("sedneg issue number 5");
  }

}

// Give the deposition fluxe from lakes to the 
void Lake::drape_deposition_flux_to_chonks(std::vector<chonk>& chonk_network, xt::pytensor<double,1>& surface_elevation, double timestep)
{

  if(this->volume == 0)
    return;

  double ratio_of_dep = this->volume_of_sediment/this->volume;
  std::cout << this->volume_of_sediment << "||" << this->volume << std::endl;

  // NEED TO DEAL WITH THAT BOBO
  if(ratio_of_dep>1)
    ratio_of_dep = 1;
  //   throw std::runtime_error("MORE_SEDIMENT_THAN_WATER_IN_LAKE_FINALISATION::" + std::to_string(ratio_of_dep) + "||" + std::to_string(this->volume));

  for(auto& no:this->nodes)
  {
    double pre = chonk_network[no].get_sediment_creation_flux();
    if(std::isfinite(pre) ==false)
     throw std::runtime_error("already naninf before draping");// std::cout <<  chonk_network[no].get_sediment_creation_flux() << "||";
    

    double slangh = ratio_of_dep * (this->water_elevation - surface_elevation[no]) / timestep;
    chonk_network[no].add_sediment_creation_flux(slangh);

    pre = chonk_network[no].get_sediment_creation_flux();
    if(std::isfinite(pre) == false)
    {
      std::cout << ratio_of_dep << "||" << (this->water_elevation - surface_elevation[no]) << "||" << timestep << std::endl;
      throw std::runtime_error(" naninf after draping");
    }
  }


}

// Fill a lake with a certain amount of water for the first time of the run
void Lake::pour_water_in_lake(
  double water_volume,
  int originode,
  std::vector<int>& node_in_lake,
  std::vector<bool>& is_processed,
  xt::pytensor<int,1>& active_nodes,
  std::vector<Lake>& lake_network,
  xt::pytensor<double,1>& surface_elevation,
  NodeGraphV2& graph,
  double cellarea,
  double dt,
  std::vector<chonk>& chonk_network,
  double& Ql_out
  )
{ 

  // std::cout << "Pouring " << water_volume << " water (rate = " << water_volume/dt << ") into " << this->lake_id << " from node " << originode << std::endl;


  // std::cout << "Entering water volume is " << water_volume << " hence water flux is " <<  water_volume/dt << std::endl;
  double save_entering_water = water_volume;
  double save_preexistingwater = this->volume;
  int n_labels = int(chonk_network[originode].get_other_attribute_array("label_tracker").size());

  // Some ongoing debugging
  // if(originode == 8371)
  //   std::cout << "processing the problem node W:" << chonk_network[originode].get_water_flux() << std::endl;

  // first cancelling the outlet to make sure I eventually find a new one, If I pour water into the lake I might merge with another one, etc
  this->outlet_node = -9999;

  double sum_this_fill = 0;

  // no matter if I am filling a new lake or an old one:
  // I am filling a vector of nodes already in teh system (Queue or lake)
  std::vector<bool> is_in_queue(node_in_lake.size(),false);
  for(auto nq:this->node_in_queue)
    is_in_queue[nq] = true;

  // First checking if this node is in a lake, if yes it means the lake has already been initialised and 
  // We are pouring water from another lake 
  if(node_in_lake[originode] == -1)
  {
    // NEW LAKE
    // Emplacing the node in the queue, It will be the first to be processed
    if(originode == 0)
      throw std::runtime_error("0 is originode...");

    depressionfiller.emplace( nodium( originode, surface_elevation[originode] ) );
    // This function fills an original lake, hence there is no lake depth yet:
    this->water_elevation = surface_elevation[originode];
    is_in_queue[originode] = true;
    // this->nodes.push_back(originode);
    // this->n_nodes ++;
  }



  // I am processing new nodes while I still have water OR still nodes upstream 
  // (if an outlet in encountered, the loop is breaked anually)
  // UPDATE_09_2020:: No idea what I meant by anually.
  while(depressionfiller.empty() == false && water_volume > 0 )
  {
    std::cout << this->water_elevation << "||" << water_volume << std::endl;
    // Getting the next node and ...
    // (If this is the first time I fill a lake -> the first node is returned)
    // (If this is another node, it is jsut the next one the closest elevation to the water level)
    nodium next_node = this->depressionfiller.top();

    // ... removing it from the priority queue 
    this->depressionfiller.pop();


    // Initialising a dummy outlet, if this outlet becomes something I shall break the loop
    int outlet = -9999;

    // Adding the upstream neighbors to the queue and checking if there is an outlet node
    outlet = this->check_neighbors_for_outlet_or_existing_lakes(next_node, graph, node_in_lake, lake_network, surface_elevation, is_in_queue, active_nodes, chonk_network);

    // If I have an outlet, then the outlet node is positive
    if(outlet >= 0)
    {
      this->water_elevation = next_node.elevation;

      // I therefore save it and break the loop
      // std::cout << "{" << outlet << "|" << this->outlet_node << "}";
      this->outlet_node = outlet;
      // and readding the node to the depression
      this->depressionfiller.emplace(next_node);


      // break;
    }



    //Decreasing water volume by filling teh lake
    double dV = this->n_nodes * cellarea * ( next_node.elevation - this->water_elevation );


    sum_this_fill += dV;
    // I SHOULD NOT HAVE TO DO THAT!!!! PROBABLY LINKED TO NUMERICAL UNSTABILITIES BUT STILL
    if(dV > - 1e-3 && dV < 0)
      dV = 0;

    if(dV<0)
    {
      std::cout << "Arg should not have neg filling::" << dV << std::endl;
      dV = 0;
      // std::string ljsdfld;
      // if(is_processed[next_node.node] == true)
      //   ljsdfld = "true";
      // else
      //   ljsdfld = "false";
      // std::cout << "DV::" << dV << " :: " << node_in_lake[next_node.node]  << " :: " << this->n_nodes << "::" << this->water_elevation << " :: " << next_node.elevation << " :: " << ljsdfld << "::" << outlet << std::endl;
      // throw std::runtime_error("negative dV lake filling");
    }

    water_volume -= dV;
    

    this->volume += dV;


    // The water elevation is the elevation of that Nodium object
    // (if 1st node -> elevation of the bottom of the depression)
    // (if other node -> lake water elevation)
    this->water_elevation = next_node.elevation;

    if(outlet >= 0)
      break;
        // Otehr wise, I do not have an outlet and I can save this node as in depression
    this->nodes.push_back(next_node.node);
    node_in_lake[next_node.node] = this->lake_id;
    this->n_nodes ++;
    // At this point I either have enough water to carry on or I stop the process
  }


  double local_balance = (save_entering_water - water_volume)/dt -  sum_this_fill/dt;

  // if(save_entering_water = save_preexistingwater)

  // std::cout << "LOCAL BALANCE SHOULD BE 0::" << local_balance << std::endl;
  // std::cout << "After raw filling lake water volume is " << water_volume << " hence water flux is " <<  water_volume/dt << std::endl;
  // std::cout << "Outletting in " << this->outlet_node << std::endl;;

  // if(this->outlet_node>=0)
  // {
  //   if(node_in_lake[this->outlet_node]>=0)
  //   {
  //     if(node_in_lake[this->outlet_node] == this->lake_id || node_in_lake[this->outlet_node] == this->get_parent_lake())
  //     {

  //       std::cout << "SSDF" << node_in_lake[this->outlet_node] << "||" << this->lake_id <<  "||" << this->nodes.size() << "||" << this->get_parent_lake() << std::endl;
  //       throw std::runtime_error("Fatal Error:: outletinlake");
  //     }
  //   }
  // }




  // checking that I did not overfilled my lake:
  if(water_volume < 0 && this->outlet_node <0)
  {
    double extra = abs(water_volume);
    // this->n_nodes -= 1;
    int extra_node = this->nodes[this->nodes.size() - 1];
    this->depressionfiller.emplace(nodium(extra_node,surface_elevation[extra_node]));
    this->nodes.erase(this->nodes.begin() + this->nodes.size() - 1);
    sum_this_fill -= extra;
    double dZ = extra / this->n_nodes / cellarea;


    this->water_elevation -= dZ;
    water_volume = 0;
    this->volume -= extra;
  }




  // Ql_out += sum_this_fill/dt;

  // std::cout << "Water balance: " << this->volume - save_preexistingwater + water_volume << "should be equal to " << save_entering_water << std::endl;


  // Labelling the node in depression as belonging to this lake and saving their depth
  for(auto Unot:this->nodes)
  {
    // std::cout << Unot << "|";
    this->depths[Unot] = this->water_elevation - surface_elevation[Unot];
    node_in_lake[Unot] = this->lake_id;
    double temp_watflux = chonk_network[Unot].get_water_flux();
    double temp_sedflux = chonk_network[Unot].get_sediment_flux();
    std::vector<double> oatlab = chonk_network[Unot].get_other_attribute_array("label_tracker");
    chonk_network[Unot].reset();
    chonk_network[Unot].set_water_flux(temp_watflux);
    chonk_network[Unot].set_sediment_flux(temp_sedflux,oatlab);
    chonk_network[Unot].initialise_local_label_tracker_in_sediment_flux(n_labels);
    // std::cout <<  chonk_network[Unot].get_sediment_creation_flux() << "||";
  }
    // std::cout << " in " << this->lake_id << std::endl;

  // std::cout << "Water volume left: " << water_volume << std::endl;
  // Transmitting the water flux to the SS receiver not in the lake
  if(water_volume > 0 && this->outlet_node >= 0)
  {

    // If the node is inactive, ie if its code is 0, the fluxes can escape the system and we stop it here
  
    // Otherwise: calculating the outflux: water_volume_remaining divided by the time step
    double out_water_rate = water_volume/(dt);

    // Getting all the receivers and the length to the oulet
    std::vector<int> receivers = graph.get_MF_receivers_at_node(this->outlet_node);
    std::vector<double> length = graph.get_MF_lengths_at_node(this->outlet_node);
    // And finding the steepest slope 
    int SS_ID = -9999; 
    double SS = -9999; // hmmmm I may need to change this name
    for(size_t i=0; i<receivers.size(); i++)
    {
      int nodelakeid = node_in_lake[receivers[i]];

      if(nodelakeid > -1)
      {
        if( nodelakeid == this->lake_id  || lake_network[nodelakeid].get_parent_lake() == this->lake_id)
          continue;
      } 

      double this_slope = (surface_elevation[this->outlet_node] - surface_elevation[receivers[i]])/length[i];


      if(this_slope>SS )
      {
        SS = this_slope;
        SS_ID = receivers[i];
      }
    }

    if(SS_ID < 0)
    {
      int sr = graph.get_Srec(this->outlet_node);
      int lsr = node_in_lake[sr];
      if(sr != this->outlet_node && lsr != this->lake_id && lake_network[lsr].get_parent_lake() != this->lake_id)
      {
        SS_ID = sr;
        SS = 0;
      }
      // std::cout << "Warning::lake outlet is itself a lake bottom? is it normal?" << std::endl;
      // yes it can be: flat surfaces
      else
      {
        // throw std::runtime_error(" The lake has an outlet with no downlslope neighbors ??? This is not possible, check Lake::initial_lake_fill or warn Boris that it happened");

        SS_ID = this->outlet_node;
        SS = 0.;
      }
    }

    // resetting the outlet CHONK
    // if(is_processed[this->outlet_node])
    // {
    //   chonk_network[this->outlet_node].cancel_split_and_merge_in_receiving_chonks(chonk_network,graph,dt);
    // }

    this->outlet_chonk = chonk(-1, -1, false); //  this is creating a "fake" chonk so its id is -1
    this->outlet_chonk.reinitialise_moving_prep();
    this->outlet_chonk.initialise_local_label_tracker_in_sediment_flux( n_labels );
    // forcing the new water flux

    this->outlet_chonk.set_water_flux(out_water_rate);

    // Forcing receivers
    // std::cout << "SS ID for routletting is " << SS_ID << std::endl;
    std::vector<int> rec = {SS_ID};
    std::vector<double> wwf = {1.};
    std::vector<double> wws = {1.};
    std::vector<double> Strec = {SS};
    this->outlet_chonk.external_moving_prep(rec,wwf,wws,Strec);

    if(this->volume_of_sediment > this->volume)
    {
      double outsed = this->volume_of_sediment - this->volume;
      this->volume_of_sediment -= outsed;
      this->outlet_chonk.set_sediment_flux(0., this->lake_label_prop);
      this->outlet_chonk.add_to_sediment_flux(outsed, this->lake_label_prop);
    }
    else
    {
      std::vector<double> baluf_2 (chonk_network[originode].get_other_attribute_array("label_tracker").size(),0.);
      this->outlet_chonk.set_sediment_flux(0.,baluf_2);
    }

    if(active_nodes[this->outlet_node] == 0)
    {
      chonk_network[this->outlet_node] = this->outlet_chonk;
      chonk_network[this->outlet_node].set_current_location(this->outlet_node);
      chonk_network[this->outlet_node].reinitialise_moving_prep();
    }

    // ready for re calculation, but it needs to be in the env object
  

  }
  // std::cout << "done with lake " << this->lake_id << " outlet is " << this->outlet_node << std::endl;


  return;
}

// This function checks all the neighbours of a pixel node and return -9999 if there is no receivers
// And the node index if there is a receiver
// 
int Lake::check_neighbors_for_outlet_or_existing_lakes(
  nodium& next_node, 
  NodeGraphV2& graph, 
  std::vector<int>& node_in_lake, 
  std::vector<Lake>& lake_network,
  xt::pytensor<double,1>& surface_elevation,
  std::vector<bool>& is_in_queue,
  xt::pytensor<int,1>& active_nodes,
  std::vector<chonk>& chonk_network
  )
{

  // Getting all neighbors: receivers AND donors
  // Ignore dummy, is just cause I is too lazy to overlaod my functions correctly
  // TODO::Overload your function correctly
  std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(next_node.node, active_nodes, neightbors, dummy);

  // std::cout << "IS THIS EVEN CALLED???:: " << next_node.node << std::endl;

  // No outlet so far
  int outlet = -9999;
  bool has_eaten = false;

  std::vector<int> rec_of_node = chonk_network[next_node.node].get_chonk_receivers_copy();

  // Checking all neighbours
  for(auto node : neightbors)
  {
    // std::cout << "From node " << next_node.node << " checking " << node << std::endl;
    // First checking if the node is already in the queue
    // If it is, well I do not need it right?
    // Right.
    if(is_in_queue[node]  || node  == next_node.node)
      continue;

    // Check if the neighbour is a lake, if it is, I am gathering the ID and the depths
    int lake_index = -1;
    if(node_in_lake[node] > -1)
    {
      lake_index = node_in_lake[node];
      // If my node is not in the queue but in the same lake (this happens when refilling the lake with more water)
      if(lake_index == this->lake_id || lake_network[lake_index].get_parent_lake() == this->lake_id )
        continue;
    }


    
    // lake depth
    double this_depth = 0.;
    // getting potentially inherited lake depth
    if(lake_index >= 0)
    {
      if(lake_network[lake_index].get_parent_lake()>=0)
        lake_index = lake_network[lake_index].get_parent_lake();
      this_depth = lake_network[lake_index].get_lake_depth_at_node(node, node_in_lake);
    }

    // It gives me the elevation to be considered
    double tested_elevation = surface_elevation[node] + this_depth;

    // However if there is another lake, and that his elevation is mine I am ingesting it
    // if the lake has a lower elevation, I am outletting in it
    // if the lake has greater elevation, I am considering this node as a potential donor
    if(lake_index > -1 && tested_elevation == next_node.elevation)
    {
      // Well, before drinking it I need to make sure that I did not already ddid it
      if(lake_network[lake_index].get_parent_lake() == this->lake_id)
        continue;
      // OK let's try to drink it 
      this->ingest_other_lake(lake_network[lake_index], node_in_lake, is_in_queue,lake_network);
      has_eaten = true; 
      continue;
    }


    
    // If the node is at higher (or same) elevation than me water surface, I set it in the queue
    else if(tested_elevation >= next_node.elevation)
    {
      // std::cout << "depressionfiller ingests " << node << " from " << next_node.node << std::endl;
      this->depressionfiller.emplace(nodium(node,tested_elevation));
      // Making sure I mark it as queued
      is_in_queue[node] = true;
      // Adding the node to the list of nodes in me queue
      this->node_in_queue.push_back(node);
    }

    // Else, if not in queue and has lower elevation, then the current mother node IS an outlet
    else
    {

      int outlake = node_in_lake[next_node.node] ;
      // std::cout << "{" << outlake << "}";
       // In some rare cases the outlet is already labelled as in this lake: for example if I already processed the lake before and readd water or more convoluted situations where I had a single pixeld lake
      if(outlake < 0)
      {
        // std::cout <<"gabul1  " << next_node.node << std::endl;
        outlet = next_node.node;
      }
      else if(outlake != this->lake_id && lake_network[outlake].get_parent_lake() != this->lake_id)
      {
        // std::cout <<"gabul2 " << next_node.node << std::endl;
        outlet = next_node.node;
      }


      // IMPORTANT::not breaking the loop: I want to get all myneighbors in the queue for potential repouring water in the thingy
    }

    // Moving to the next neighbour
  }

  // if a lake ahs been ingested, it makes the outlet ambiguous and need reprocessing of this node
  if(has_eaten)
  {
    this->depressionfiller.emplace(next_node);
    outlet = -9999;
  }

  // // outlet is >= 0 -> tehre is an outlet
  // if(outlet>=0)
  //   std::cout << "POCHTRAC::" << outlet << "||" << node_in_lake[outlet] << "||" << this->lake_id << std::endl;

  if(active_nodes[next_node.node] == false)
  {
    outlet = next_node.node;
    
  }

  return outlet;
}

void  ModelRunner::find_underfilled_lakes_already_processed_and_give_water(int SS_ID, std::vector<bool>& is_processed )
{

  // Legaciatisation
  xt::pytensor<double,1>& surface_elevation = this->io_double_array["surface_elevation"];
  xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];
  double& dt = timestep;
  double cellarea = this->io_double["dx"] * this->io_double["dy"] ;

  int& n_elements = this->io_int["n_elements"];
  // Using an already processed vector to not readd
  std::vector<bool> to_reprocessed(n_elements,false);
  std::vector<int> traversal(n_elements,-9999); // -9999 is nodata
  std::unordered_map<int,double> to_add_in_lakes, to_add_in_lakes_sed_edition;
  std::unordered_map<int, std::vector<double> >to_add_in_lakes_sed_edition_but_label_tracker;
  traversal[0] = SS_ID;
  to_reprocessed[SS_ID] = true;
  // basically here I wneed to reprocess all nodes dowstream of that one!
  // if processed and not in lake -> reprocess
  // else: stop
  // First: graph traversal
  int n_reprocessed = 0;
  int next_test = traversal[0];
  int reading_ID = 1, writing_ID = 1;
  while(next_test != -9999)
  {
    // feeding the queues with the receivers
    std::vector<int> recs = graph.get_MF_receivers_at_node_no_rerouting(next_test);
    for (auto node:recs)
    {
      // if not preprocessed globally or jsut already in the queue yet: boom
      if(to_reprocessed[node] || is_processed[node] == false || node_in_lake[node] >= 0)
        continue;

      to_reprocessed[node] = true;
      traversal[writing_ID] = node;
      writing_ID++;
    }

    // Reading the next node in list
    next_test = traversal[reading_ID];
    reading_ID++;
    // will break here if next node is -9999
  }
  // std::cout << writing_ID << "||";


  //Reinitialising all the concerned nodes but the outlet to repropagate correctly the thingies 
  for(size_t i=1; i<writing_ID; i++)
    chonk_network[traversal[i]] = chonk(traversal[i],traversal[i],false);


  // reprocessing nodes
  for(int i=0; i<n_elements; i++)
  {

 
    int node = graph.get_MF_stack_at_i(i);

    if(to_reprocessed[node] == false)
      continue;


    this->manage_fluxes_before_moving_prep(chonk_network[node], this->label_array[node] );


    this->manage_move_prep(chonk_network[node]);
    // std::cout << "call from reproc" << std::endl;
    // std::cout << "call from reproc" << std::endl;
    this->manage_fluxes_after_moving_prep(chonk_network[node], this->label_array[node]);
    // std::cout << "called from reproc" << std::endl;

    // need to check if some nodes give in lake
    std::vector<int> to_ignore;
    std::vector<int>& crec = chonk_network[node].get_chonk_receivers();
    std::vector<double>& cwawe = chonk_network[node].get_chonk_water_weight();
    std::vector<double>& csedw = chonk_network[node].get_chonk_sediment_weight();
    for(size_t i=0; i<crec.size(); i++)
    {
      if(node_in_lake[crec[i]]>=0)
      {
        to_ignore.push_back(crec[i]);
        int lake_to_consider = node_in_lake[crec[i]];
        to_add_in_lakes[lake_to_consider] += cwawe[i] * chonk_network[crec[i]].get_water_flux() * dt;
        to_add_in_lakes_sed_edition[lake_to_consider] += chonk_network[crec[i]].get_sediment_flux() * csedw[i];
        if(to_add_in_lakes_sed_edition_but_label_tracker.count(lake_to_consider) == 0)
        {
          to_add_in_lakes_sed_edition_but_label_tracker[lake_to_consider] = chonk_network[crec[i]].get_other_attribute_array("label_tracker");
        }
        else
        {
    std::cout << "ka";
          to_add_in_lakes_sed_edition_but_label_tracker[lake_to_consider] = mix_two_proportions(to_add_in_lakes_sed_edition[lake_to_consider],
            to_add_in_lakes_sed_edition_but_label_tracker[lake_to_consider]
            ,chonk_network[crec[i]].get_sediment_flux() * csedw[i], chonk_network[crec[i]].get_other_attribute_array("label_tracker"));
    std::cout << "ren";
        }
      }
    }

    chonk_network[node].split_and_merge_in_receiving_chonks_ignore_some(chonk_network, graph, timestep, to_ignore);
  }

  for(auto x:to_add_in_lakes)
  {
    int lake_to_fill = x.first;
    if(lake_network[lake_to_fill].get_parent_lake()>=0)
      lake_to_fill = lake_network[lake_to_fill].get_parent_lake();
    double water_volume = x.second;
    std::cout << "dar";

    this->lake_network[lake_to_fill].pour_sediment_into_lake(to_add_in_lakes_sed_edition[lake_to_fill], to_add_in_lakes_sed_edition_but_label_tracker[lake_to_fill]);
    std::cout << "de";

    this->lake_network[lake_to_fill].pour_water_in_lake(water_volume,lake_network[lake_to_fill].get_lake_nodes()[0], // pouring water in a random nodfe in the lake, it does not matter
  node_in_lake, is_processed, active_nodes,lake_network, surface_elevation,graph, cellarea, timestep,chonk_network,this->Ql_out);
  
  }

}


double Lake::get_lake_depth_at_node(int node, std::vector<int>& node_in_lake)
{
  return this->depths[node];
}


xt::pytensor<int,1> ModelRunner::get_lake_ID_array_raw()
{
  xt::pytensor<int,1> output = xt::zeros<int>({this->io_int["n_elements"]}) -1;

  for(int i=0; i< this->io_int["n_elements"]; i++)
  {
    if(node_in_lake[i] >= 0)
      output[i] = node_in_lake[i];
  }
  return output;
}


xt::pytensor<int,1> ModelRunner::get_lake_ID_array()
{
  xt::pytensor<int,1> output = xt::zeros<int>({this->io_int["n_elements"]}) -1;

  for(int i=0; i< this->io_int["n_elements"]; i++)
  {
    if(node_in_lake[i] >= 0)
    {
      if(this->lake_network[node_in_lake[i]].get_parent_lake() >= 0)
        output[i] = this->lake_network[node_in_lake[i]].get_parent_lake();
    }
  }
  return output;
}


xt::pytensor<int,1> ModelRunner::get_mstack_checker()
{
  xt::pytensor<int,1>& mstack = this->graph.get_MF_stack_full_adress();
  xt::pytensor<int,1>  output = xt::zeros<int>({this->io_int["n_elements"]}) -1;
  int i = 0;
  for(auto n: mstack)
  {
    output[n] = i;
    i++;
  }
  return output;
}


void ModelRunner::print_chonk_info(int node)
{
  chonk& tchonk = this->chonk_network[node];
  // std::cout << "===== CHONK DEBUGGER =====" std::endl;
  std::cout << "=ID:" << node << "= lake:" <<node_in_lake[node] << "= processed:" << is_processed[node]  <<" =" << std::endl;
  std::cout << "=Water flux: " << tchonk.get_water_flux() << " =" << std::endl;

}

//#################################################################################
//#################################################################################
//#################################################################################
//#################################################################################
// OTher functions, with a clear one-off utility ##################################
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



//////////////////////////////////////////
/////////////////////////////////////////





#endif

