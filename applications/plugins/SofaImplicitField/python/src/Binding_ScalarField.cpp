/******************************************************************************
*                           SofaImplicitField plugin                          *
*                  (c) 2024 CNRS, University of Lille, INRIA                  *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <pybind11/pybind11.h>

#include <SofaPython3/PythonFactory.h>
#include <SofaPython3/PythonEnvironment.h>
#include <SofaPython3/Sofa/Core/Binding_Base.h>
#include <SofaImplicitField/components/geometry/ScalarField.h>

#include "Binding_ScalarField.h"

/// Makes an alias for the pybind11 namespace to increase readability.
namespace py { using namespace pybind11; }

namespace sofaimplicitfield {
using namespace sofapython3;
using sofa::component::geometry::ScalarField;
using sofa::core::objectmodel::BaseObject;
using sofa::type::Vec3d;

class ScalarField_Trampoline : public ScalarField {
public:
    SOFA_CLASS(ScalarField_Trampoline, ScalarField);

    double getValue(Vec3d& pos, int& domain) override{
        SOFA_UNUSED(domain);
        PythonEnvironment::gil acquire;

        PYBIND11_OVERLOAD_PURE(double, ScalarField, getValue, pos.x(), pos.y(), pos.z());
    }
};

void moduleAddScalarField(py::module &m) {
    py::class_<ScalarField, BaseObject, ScalarField_Trampoline,
               py_shared_ptr<ScalarField>> f(m, "ScalarField", py::dynamic_attr(), "");

    f.def(py::init([](py::args &args, py::kwargs &kwargs) {
        auto ff = sofa::core::sptr<ScalarField_Trampoline> (new ScalarField_Trampoline());

        ff->f_listening.setValue(true);

        if (args.size() == 1) ff->setName(py::cast<std::string>(args[0]));

        py::object cc = py::cast(ff);
        for (auto kv : kwargs) {
            std::string key = py::cast<std::string>(kv.first);
            py::object value = py::reinterpret_borrow<py::object>(kv.second);
            if (key == "name") {
                if (args.size() != 0) {
                    throw py::type_error("The name is set twice as a "
                                        "named argument='" + py::cast<std::string>(value) + "' and as a"
                                                                                            "positional argument='" +
                                        py::cast<std::string>(args[0]) + "'.");
                }
            }
        }
        return ff;
    }));

    m.def("getValue", &ScalarField_Trampoline::getValue);
}

}
