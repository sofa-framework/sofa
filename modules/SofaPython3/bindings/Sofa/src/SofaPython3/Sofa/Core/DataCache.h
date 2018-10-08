#ifndef SOFAPYTHON3_SOFA_CORE_DATACACHE_H
#define SOFAPYTHON3_SOFA_CORE_DATACACHE_H

#include <pybind11/numpy.h>

////////////////////////// FORWARD DECLARATION ///////////////////////////
namespace sofa {
    namespace core {
        namespace objectmodel {
            class BaseData;
        }
    }
}


/////////////////////////////// DECLARATION //////////////////////////////
namespace sofapython3
{
/// Makes an alias for the pybind11 namespace to increase readability.
namespace py { using namespace pybind11; }

using sofa::core::objectmodel::BaseData;

bool hasArrayFor(BaseData* d);
py::array resetArrayFor(BaseData* d);
py::array getPythonArrayFor(BaseData* d);
void trimCache();

} /// sofapython3


#endif /// SOFAPYTHON3_SOFA_CORE_DATACACHE_H


