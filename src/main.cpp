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
// include "cppintail.hpp"

#include "cppintail.hpp"
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
      .def("get_other_attribute", &ModelRunner::get_other_attribute)
      .def("get_all_nodes_in_depression", &ModelRunner::get_all_nodes_in_depression)
      .def("update_timestep", &ModelRunner::update_timestep)
      .def("DEBUG_get_preacc", &ModelRunner::DEBUG_get_preacc)
    ;


    // py::class_<cppintail>(m, "cppintail",py::dynamic_attr())
    //   .def(py::init<>())
    //   .def(py::init([](float tXMIN, float tXMAX, float tYMIN, float tYMAX, float tXRES, float tYRES, int tNROWS, int tNCOLS, float tNODATAVALUE){return std::unique_ptr<cppintail>(new cppintail(tXMIN, tXMAX, tYMIN, tYMAX, tXRES, tYRES, tNROWS, tNCOLS, tNODATAVALUE)); }))
    //   .def("compute_neighbors", &cppintail::compute_neighbors)
    //   .def("find_nodes_with_no_donors", &cppintail::find_nodes_with_no_donors)
    //   .def("get_flowdir", &cppintail::get_flowdir)
      
    // ;
}
