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
// include "nodegraph.hpp"

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
        Testing flow routing system for my post-doc

        .. currentmodule:: CHONK_cpp

        .. autosummary::
           :toctree: _generate

    )pbdoc";

// (double ttimestep, double tstart_time, std::vector<std::string> tordered_flux_methods, std::string tmove_method)
    py::class_<ModelRunner>(m, "ModelRunner",py::dynamic_attr())
      .def(py::init<>())
      .def(py::init([](double ttimestep, double tstart_time, std::vector<std::string> tordered_flux_methods, std::string tmove_method){return std::unique_ptr<ModelRunner>(new ModelRunner( ttimestep, tstart_time, tordered_flux_methods, tmove_method)); }))
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
      .def("set_lake_switch", &ModelRunner::set_lake_switch)
      .def("get_broken_nodes", &ModelRunner::get_broken_nodes)

    ;
    m.def("pop_elevation_to_SS_SF_SPIL", pop_elevation_to_SS_SF_SPIL);

    py::class_<NodeGraphV2>(m, "NodeGraph",py::dynamic_attr())
      .def(py::init<>())
      .def(py::init([](xt::pytensor<int,1>& D8stack, xt::pytensor<int,1>& D8rec, xt::pytensor<int,1>& Prec, xt::pytensor<double,1>& D8Length, xt::pytensor<int,2>& Mrec,xt::pytensor<double,2>& Mlength, xt::pytensor<double,1>& elevation,xt::pytensor<bool,1>& active_nodes, 
        double dx, double dy, int nrows, int ncols){return std::unique_ptr<NodeGraphV2>(new NodeGraphV2( D8stack,  D8rec, Prec,  D8Length, Mrec, Mlength, elevation,active_nodes,  dx,  dy, nrows, ncols)); }))
      .def("get_MF_stack_full", &NodeGraphV2::get_MF_stack_full)
      ;



}
