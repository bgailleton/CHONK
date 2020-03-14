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

#include "cppintail.hpp"


// Common includes
#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <map>

namespace py = pybind11;




PYBIND11_MODULE(pyntail, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Testing flow routing system for my post-doc

        .. currentmodule:: pyntail

        .. autosummary::
           :toctree: _generate

    )pbdoc";


    py::class_<cppintail>(m, "cppintail",py::dynamic_attr())
      .def(py::init<>())
      .def(py::init([](float tXMIN, float tXMAX, float tYMIN, float tYMAX, float tXRES, float tYRES, int tNROWS, int tNCOLS, float tNODATAVALUE){return std::unique_ptr<cppintail>(new cppintail(tXMIN, tXMAX, tYMIN, tYMAX, tXRES, tYRES, tNROWS, tNCOLS, tNODATAVALUE)); }))
      .def("compute_neighbors", &cppintail::compute_neighbors)
      .def("find_nodes_with_no_donors", &cppintail::find_nodes_with_no_donors)
      .def("get_flowdir", &cppintail::get_flowdir)
      
    ;
}
