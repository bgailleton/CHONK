#include "pybind11/pybind11.h"
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor/xadapt.hpp"
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>

#include "nodegraph.hpp"
#include "CHONK.hpp"
#include "Environment.hpp"

// Common includes
#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <map>

namespace py = pybind11;

PYBIND11_MODULE(CHONK_cpp, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        CHONK model runner, a tool to set and run the model CHONK

        .. currentmodule:: CHONK_cpp

        .. autosummary::
           :toctree: _generate

    )pbdoc";

// (double ttimestep, double tstart_time, std::vector<std::string> tordered_flux_methods, std::string tmove_method)
    py::class_<ModelRunner>(m, "ModelRunner",py::dynamic_attr())
      .def(py::init<>())
      .def(py::init([](double ttimestep, std::vector<std::string> tordered_flux_methods, std::string tmove_method){return std::unique_ptr<ModelRunner>(new ModelRunner( ttimestep, tordered_flux_methods, tmove_method)); }))
      .def("initiate_nodegraph", &ModelRunner::initiate_nodegraph)
      .def("run", &ModelRunner::run)
      .def("update_int_param",&ModelRunner::update_int_param)
      .def("update_double_param",&ModelRunner::update_double_param)
      .def("update_array_int_param",&ModelRunner::update_array_int_param)
      .def("update_array2d_int_param",&ModelRunner::update_array2d_int_param)
      .def("update_array_double_param",&ModelRunner::update_array_double_param)
      .def("update_array2d_double_param",&ModelRunner::update_array2d_double_param)
      .def("get_int_param", &ModelRunner::get_int_param)
      .def("get_double_param", &ModelRunner::get_double_param)
      .def("get_array_int_param", &ModelRunner::get_array_int_param)
      .def("get_array2d_int_param", &ModelRunner::get_array2d_int_param)
      .def("get_array_double_param", &ModelRunner::get_array_double_param)
      .def("get_array2d_double_param", &ModelRunner::get_array2d_double_param)
      .def("DEBUG_modify_double_array_param_inplace", &ModelRunner::DEBUG_modify_double_array_param_inplace)
      .def("get_water_flux",&ModelRunner::get_water_flux)
      .def("get_sediment_flux",&ModelRunner::get_sediment_flux)
      .def("get_erosion_flux",&ModelRunner::get_erosion_flux)
      .def("get_other_attribute", &ModelRunner::get_other_attribute)
      .def("update_timestep", &ModelRunner::update_timestep)
      .def("DEBUG_check_weird_val_stacks", &ModelRunner::DEBUG_check_weird_val_stacks)
      .def("DEBUG_get_receivers_at_node", &ModelRunner::DEBUG_get_receivers_at_node)
      .def("DEBUG_get_Sreceivers_at_node", &ModelRunner::DEBUG_get_Sreceivers_at_node)
      .def("set_lake_switch", &ModelRunner::set_lake_switch)
      .def("get_broken_nodes", &ModelRunner::get_broken_nodes)
      .def("get_DEBUG_connbas", &ModelRunner::get_DEBUG_connbas)
      .def("get_DEBUG_connode", &ModelRunner::get_DEBUG_connode)
      .def("get_mstree", &ModelRunner::get_mstree)
      .def("get_mstree_translated", &ModelRunner::get_mstree_translated)
      .def("reinitialise_label_list",&ModelRunner::reinitialise_label_list)
      .def("initialise_label_list",&ModelRunner::initialise_label_list)
      .def("update_label_array", &ModelRunner::update_label_array)
      .def("get_label_tracking_results", &ModelRunner::get_label_tracking_results)
      .def("get_ordered_flux_method", &ModelRunner::get_ordered_flux_method)
      .def("update_flux_methods", &ModelRunner::update_flux_methods)
      .def("update_move_method", &ModelRunner::update_move_method)
      .def("add_external_to_double_array", &ModelRunner::add_external_to_double_array)
      .def("get_superficial_layer_sediment_prop", &ModelRunner::get_superficial_layer_sediment_prop)
      .def("get_erosion_bedrock_only_flux", &ModelRunner::get_erosion_bedrock_only_flux)
      .def("get_erosion_sed_only_flux", &ModelRunner::get_erosion_sed_only_flux)
      .def("get_sediment_creation_flux", &ModelRunner::get_sediment_creation_flux)
      .def("get_deposition_flux", &ModelRunner::get_deposition_flux)
      .def("get_lake_ID_array_raw", &ModelRunner::get_lake_ID_array_raw)
      .def("get_lake_ID_array", &ModelRunner::get_lake_ID_array)
      .def("get_mstack_checker", &ModelRunner::get_mstack_checker)
      .def("get_Qw_in", &ModelRunner::get_Qw_in)
      .def("get_Qw_out", &ModelRunner::get_Qw_out)
      .def("get_Ql_in", &ModelRunner::get_Ql_in)
      .def("get_Ql_out", &ModelRunner::get_Ql_out)
      .def("get_flat_mask", &ModelRunner::get_flat_mask)
      .def("get_sed_prop_by_label", &ModelRunner::get_sed_prop_by_label)
      .def("get_sed_prop_by_label_matrice", &ModelRunner::get_sed_prop_by_label_matrice)
      .def("get_debugint", &ModelRunner::get_debugint)
      .def("get_Qs_mass_balance", &ModelRunner::get_Qs_mass_balance)
      .def("set_surface_elevation",&ModelRunner::set_surface_elevation)
      .def("set_surface_elevation_tp1",&ModelRunner::set_surface_elevation_tp1)
      .def("set_topography",&ModelRunner::set_topography)
      .def("set_active_nodes",&ModelRunner::set_active_nodes)
      .def("get_surface_elevation",&ModelRunner::get_surface_elevation)
      .def("get_surface_elevation_tp1",&ModelRunner::get_surface_elevation_tp1)
      .def("get_topography",&ModelRunner::get_topography)
      .def("get_active_nodes",&ModelRunner::get_active_nodes)
      .def("add_external_to_surface_elevation_tp1",&ModelRunner::add_external_to_surface_elevation_tp1)
      .def("set_sed_height", &ModelRunner::set_sed_height)
      .def("set_sed_height_tp1", &ModelRunner::set_sed_height_tp1)
      .def("get_sed_height", &ModelRunner::get_sed_height)
      .def("get_sed_height_tp1", &ModelRunner::get_sed_height_tp1)
      .def_readwrite("CHARLIE_I", &ModelRunner::CHARLIE_I)
      .def_readwrite("CIDRE_HS", &ModelRunner::CIDRE_HS)
      .def_readwrite("tool_effect_rock", &ModelRunner::tool_effect_rock)
      .def_readwrite("tool_effect_sed", &ModelRunner::tool_effect_sed)
      .def_readwrite("external_K", &ModelRunner::external_K)
      .def_readwrite("external_kappa", &ModelRunner::external_kappa)
      .def_readwrite("precipitations_enabled", &ModelRunner::precipitations_enabled)
      .def_readwrite("precipitations", &ModelRunner::precipitations)
      .def_readwrite("lake_evaporation", &ModelRunner::lake_evaporation)
      .def_readwrite("lake_evaporation_rate_spatial", &ModelRunner::lake_evaporation_rate_spatial)
      .def_readwrite("thresholdMF2SF", &ModelRunner::thresholdMF2SF)
      .def_readwrite("lake_depth", &ModelRunner::lake_depth)
      .def_readwrite("depths_res_sed_proportions", &ModelRunner::depths_res_sed_proportions)
      .def_readwrite("initial_carving", &ModelRunner::initial_carving)
      .def("get_fluvial_Qs", &ModelRunner::get_fluvial_Qs)
      .def("get_hillslope_Qs", &ModelRunner::get_hillslope_Qs)
      .def("get_neighbours_for_debugging", &ModelRunner::get_neighbours_for_debugging)
      .def("get_Qsprop_bound", &ModelRunner::get_Qsprop_bound)
      .def("get_stratiprop",&ModelRunner::get_stratiprop)

      

      .def("get_K_calc", &ModelRunner::get_K_calc)
      .def("get_top_depression", &ModelRunner::get_top_depression)
      .def("get_potential_volume", &ModelRunner::get_potential_volume)
      .def("get_sum_of_all_volume_full_lake", &ModelRunner::get_sum_of_all_volume_full_lake)
      .def("get_fluvlabprop",&ModelRunner::get_fluvlabprop)

    ;

    
    m.def("set_DEBUG_switch_nodegraph",set_DEBUG_switch_nodegraph);
    m.def("pop_elevation_to_SS_SF_SPIL", pop_elevation_to_SS_SF_SPIL);

    // py::class_<NodeGraphV2>(m, "NodeGraph",py::dynamic_attr())
    //   .def(py::init<>())
    //   .def(py::init([]( xt::pytensor<double,1>& elevation,xt::pytensor<bool,1>& active_nodes, 
    //     double dx, double dy, int nrows, int ncols){return std::unique_ptr<NodeGraphV2>(new NodeGraphV2( elevation,active_nodes,  dx,  dy, nrows, ncols)); }))
    //   .def("get_MF_stack_full", &NodeGraphV2::get_MF_stack_full)
    //   ;

    py::class_<labelz>(m, "label",py::dynamic_attr())
      .def(py::init<>())
      .def(py::init([](int label_id){return std::unique_ptr<labelz>(new labelz( label_id)); }))
      .def("set_int_attribute", &labelz::set_int_attribute)
      .def("set_double_attribute", &labelz::set_double_attribute)
      .def("set_int_array_attribute", &labelz::set_int_array_attribute)
      .def("set_double_array_attribute", &labelz::set_double_array_attribute)
      .def_readwrite("m",&labelz::m)
      .def_readwrite("n",&labelz::n)
      .def_readwrite("base_K",&labelz::base_K)
      .def_readwrite("Ks_modifyer",&labelz::Ks_modifyer)
      .def_readwrite("Kr_modifyer",&labelz::Kr_modifyer)
      .def_readwrite("dimless_roughness",&labelz::dimless_roughness)
      .def_readwrite("V",&labelz::V)
      .def_readwrite("dstar",&labelz::dstar)
      .def_readwrite("threshold_incision",&labelz::threshold_incision)
      .def_readwrite("threshold_entrainment",&labelz::threshold_entrainment)
      .def_readwrite("kappa_base",&labelz::kappa_base)
      .def_readwrite("kappa_r_mod",&labelz::kappa_r_mod)
      .def_readwrite("kappa_s_mod",&labelz::kappa_s_mod)
      .def_readwrite("critical_slope",&labelz::critical_slope)
      .def_readwrite("sensitivity_tool_effect",&labelz::sensitivity_tool_effect)
      // .def("get_MF_stack_full", &NodeGraphV2::get_MF_stack_full)
      ;



}
