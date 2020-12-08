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

// All the xtensor requirements
#include "xtensor-python/pyarray.hpp" // manage the I/O of numpy array
#include "xtensor-python/pytensor.hpp" // same
#include "xtensor-python/pyvectorize.hpp" // Contain some algorithm for vectorised calculation (TODO)
#include "xtensor/xadapt.hpp" // the function adapt is nice to convert vectors to numpy arrays
#include "xtensor/xmath.hpp" // Array-wise math functions
#include "xtensor/xarray.hpp"// manages the xtensor array (lower level than the numpy one)
#include "xtensor/xtensor.hpp" // same

#include "nodegraph.hpp"



class chonk
{
  public:
    // Default constructor
    chonk() { create(); };
    // Normal constructor
    chonk(int tchonkID, int tcurrent_node, bool tmemory_saver){create(tchonkID, tcurrent_node, tmemory_saver);};
    // Reset function
    void reset();


    // Merge function(s)
    void split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraphV2& graph, xt::pytensor<double,1>& surface_elevation_tp1, xt::pytensor<double,1>& sed_height_tp1, double dt);
    void split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraphV2& graph, double dt);
    void split_and_merge_in_receiving_chonks_ignore_some(std::vector<chonk>& chonkscape, NodeGraphV2& graph, double dt, std::vector<int>& to_ignore);
    void cancel_split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraphV2& graph, double dt);

    // move and split functions
    void move_to_steepest_descent(NodeGraphV2& graph, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
  xt::pytensor<double,1>& surface_elevation, xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres, std::vector<chonk>& chonk_network);
    void move_to_steepest_descent_nodepression(NodeGraphV2& graph, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
  xt::pytensor<double,1>& surface_elevation, xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres, std::vector<chonk>& chonk_network);
    void move_MF_from_fastscapelib(NodeGraphV2& graph, xt::pytensor<double,2>& external_weigth_water_fluxes, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
  xt::pytensor<double,1>& surface_elevation, xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres, std::vector<chonk>& chonk_network);
    void move_MF_from_fastscapelib_threshold_SF(NodeGraphV2& graph, double threshold_Q, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
  xt::pytensor<double,1>& surface_elevation, xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres, std::vector<chonk>& chonk_network);
    // Functions that apply and calculate fluxes
    //#### In place flux applyer (BEFORE move)
    void inplace_only_drainage_area(double Xres, double Yres);
    void inplace_precipitation_discharge(double Xres, double Yres, xt::pytensor<double,1>& precipitation);
    void inplace_infiltration(double Xres, double Yres, xt::pytensor<double,1>& infiltration);
    
    //#### Cancelling counterpart for the reprocessing
    void cancel_inplace_only_drainage_area(double Xres, double Yres);
    void cancel_inplace_precipitation_discharge(double Xres, double Yres, xt::pytensor<double,1>& precipitation);
    void cancel_inplace_infiltration(double Xres, double Yres, xt::pytensor<double,1>& infiltration);


    //#### active flux applyer (AFTER move)
    void active_simple_SPL(double n, double m, double K, double dt, double Xres, double Yres, int label);
    void charlie_I(double n, double m, double K_r, double K_s,
  double dimless_roughness, double this_sed_height, double V_param, 
  double d_star, double threshold_incision, double threshold_sed_entrainment,
  int label, std::vector<double> sed_label_prop, double dt, double Xres, double Yres);



    // Accessors and modifyers
    // # Admin attribute
    int get_current_location() {return current_node;}
    void set_current_location(int cl) { current_node = cl;}
    // # Water flux
    double get_water_flux(){return water_flux;}
    void set_water_flux(double value){water_flux = value;}
    void add_to_water_flux(double value){water_flux += value;}
    // # Erosion flux
    double get_erosion_flux_undifferentiated(){return erosion_flux_undifferentiated;}
    void set_erosion_flux_undifferentiated(double value){erosion_flux_undifferentiated = value;}
    double get_erosion_flux_only_sediments(){return erosion_flux_only_sediments;}
    void set_erosion_flux_only_sediments(double value){erosion_flux_only_sediments = value;}
    double get_erosion_flux_only_bedrock(){return erosion_flux_only_bedrock;}
    void set_erosion_flux_only_bedrock(double value){erosion_flux_only_bedrock = value;}
    // # Deposition flux
    double get_deposition_flux(){return deposition_flux;}
    void set_deposition_flux(double value){deposition_flux = value;}
    void add_deposition_flux(double value){deposition_flux += value;}
    // # sediment_creation flux
    double get_sediment_creation_flux(){return sediment_creation_flux;}
    void set_sediment_creation_flux(double value){sediment_creation_flux = value;}
    void add_sediment_creation_flux(double value){sediment_creation_flux += value;}
    // # Sediment flux
    double get_sediment_flux(){return sediment_flux;}
    void set_sediment_flux_no_tacking(double value){sediment_flux = value;}
    void set_sediment_flux(double value,std::vector<double> label_proportions);
    void add_to_sediment_flux_no_tracking(double value){sediment_flux += value;}
    void add_to_sediment_flux(double value, std::vector<double> label_proportions);
    //# check emptyness 
    bool check_if_empty(){return is_empty;};
    //# Check if depression solved
    bool is_depression_solved_at_this_timestep(){return depression_solved_at_this_timestep;};
    //# other attribute
    void set_other_attribute(std::string key, double val){other_attributes[key] = val;}
    double get_other_attribute(std::string key){return other_attributes[key];}
    std::vector<double> get_other_attribute_array(std::string key){return other_attributes_arrays[key];}
    void set_other_attribute_array(std::string key, std::vector<double> val){ other_attributes_arrays[key] = val;}

    // receivers
    std::vector<int>& get_chonk_receivers(){return receivers;}
    std::vector<double>& get_chonk_slope_to_recs(){return slope_to_rec;}
    // water weights
    std::vector<double>& get_chonk_water_weight(){return weigth_water_fluxes;}
    std::vector<double>& get_chonk_sediment_weight(){return weigth_sediment_fluxes;}
    std::vector<double> get_preexisting_sediment_flux_by_receivers();
    
    // receivers
    std::vector<int> get_chonk_receivers_copy(){return receivers;}
    // water weights
    std::vector<double> get_chonk_water_weight_copy(){return weigth_water_fluxes;}
    std::vector<double> get_chonk_sediment_weight_copy(){return weigth_sediment_fluxes;}
    std::vector<double> get_chonk_slope_to_recs_copy(){return slope_to_rec;}


    // reinitialise moving preparation by clearing all vectors of move
    void reinitialise_moving_prep(){receivers.clear(); weigth_water_fluxes.clear(); weigth_sediment_fluxes.clear(); slope_to_rec.clear(); return;}
    void external_moving_prep(std::vector<int> rec,std::vector<double> wwf,std::vector<double> wws, std::vector<double> strec)
         {receivers = rec; weigth_water_fluxes = wwf; weigth_sediment_fluxes = wws; slope_to_rec = strec; return;}

    // Tracking and labelling functions
    void initialise_local_label_tracker_in_sediment_flux(int n_labels){other_attributes_arrays["label_tracker"] = std::vector<double>(n_labels,0.);}

  protected:
    // Administration attributes
    // The ID of the chonk
    int chonkID;
    // Check if the chonk is a dummy one
    bool is_empty;
    // Current location on the graph
    int current_node;
    // Check if the node has solved a pit and therefore no active flux (erosion,...) should be activated, jsut transfer water and sed if you can 
    bool depression_solved_at_this_timestep;
    // Memory saver: if activated it empty the chonks after it moved (But it makes the tracking/recording slower and requiring more code)
    bool memory_saver;

    // Fluxes
    // Current flux of water in the CHONK (in L^3/T)
    double water_flux;
    // Current erosion flux in H/T
    double erosion_flux_undifferentiated;
    double erosion_flux_only_sediments;
    double erosion_flux_only_bedrock;
    // Current deposition flux in H/T
    double deposition_flux;
    // Current Sediment flux in L^3
    double sediment_flux;
    // sediment creation flux
    double sediment_creation_flux;


    // Movers
    std::vector<int> receivers;
    std::vector<double> weigth_water_fluxes;
    std::vector<double> weigth_sediment_fluxes;
    std::vector<double> slope_to_rec;


    // Trackers
    // I may have a wide range of variables here, variables that I am not sure I will often use.
    // I think that if I systematically add them as attributes, I will loose some performance
    // Instead I well assign a map of attribute so that I can add as many as I need, even if less explicit
    std::map<std::string,double> other_attributes;
    std::map<std::string,std::vector<double> > other_attributes_arrays;


  private:
    void create();
    void create(int tchonkID, int tcurrent_node, bool tmemory_saver);
};

std::vector<double> mix_two_proportions(double prop1, std::vector<double> labprop1, double prop2, std::vector<double> labprop2);
bool double_equals(double a, double b, double epsilon = 0.0001);


#endif