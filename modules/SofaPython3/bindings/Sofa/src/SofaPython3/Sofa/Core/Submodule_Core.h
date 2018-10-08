#ifndef SOFAPYTHON3_SOFA_CORE_SUBMODULE_H
#define SOFAPYTHON3_SOFA_CORE_SUBMODULE_H

#include <pybind11/pybind11.h>

namespace sofapython3
{
namespace py { using namespace pybind11; }

py::module addSubmoduleCore(py::module& m) ;

} /// namespace sofapython3

#endif /// SOFAPYTHON3_SOFA_CORE_SUBMODULE_H


