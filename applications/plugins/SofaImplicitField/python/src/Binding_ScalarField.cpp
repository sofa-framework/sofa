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
using sofa::type::Vec3;
using sofa::type::Mat3x3;

class ScalarField_Trampoline : public ScalarField {
public:
    SOFA_CLASS(ScalarField_Trampoline, ScalarField);

    // Override this function so that it returns the actual python class name instead of
    // "ScalarField_Trampoline" which correspond to this utility class.
    std::string getClassName() const override
    {
        PythonEnvironment::gil acquire;

        // Get the actual class name from python.
        return py::str(py::cast(this).get_type().attr("__name__"));
    }

    double getValue(Vec3& pos, int& domain) override
    {
        SOFA_UNUSED(domain);
        PythonEnvironment::gil acquire;

        PYBIND11_OVERLOAD_PURE(double, ScalarField, getValue, pos);
    }

    Vec3 getGradient(Vec3& pos, int& domain) override
    {
        SOFA_UNUSED(domain);
        PythonEnvironment::gil acquire;

        PYBIND11_OVERLOAD(Vec3, ScalarField, getGradient, pos);
    }

    void getHessian(Vec3 &pos, Mat3x3& h) override
    {
        /// The implementation is a bit more complex compared to getGradient. This is because we change de signature between the c++ API and the python one.
        PythonEnvironment::gil acquire;

        // Search if there is a python override,
        pybind11::function override = pybind11::get_override(static_cast<const ScalarField*>(this),"getHessian");
        if(!override){
            return ScalarField::getHessian(pos, h);
        }
        // as there is one override, we call it, passing the "pos" argument and storing the return of the
        // value in the "o" variable.
        auto o = override(pos);

        // then we check that the function correctly returned a Mat3x3 object and copy its value
        // in case there is no Mat3x3 returned values, rise an error
        if(py::isinstance<Mat3x3>(o))
            h = py::cast<Mat3x3>(o);
        else
            throw py::type_error("The function getHessian must return a Mat3x3");
        return;
    }
};

void moduleAddScalarField(py::module &m) {
    py::class_<ScalarField, ScalarField_Trampoline, BaseObject,
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
                ff->setName(py::cast<std::string>(value));
            }
        }
        return ff;
    }));

    f.def("getValue", [](ScalarField* self, Vec3 pos){
        int domain=-1;
        // This shouldn't be self->ScalarField::getValue because it is a pure function
        // so there is not ScalarField::getValue emitted.
        return self->getValue(pos, domain);
    });

    f.def("getGradient", [](ScalarField* self, Vec3 pos){
        int domain=-1;
        return self->ScalarField::getGradient(pos, domain);
    });

    f.def("getHessian", [](ScalarField* self, Vec3 pos){
        Mat3x3 result;
        self->getHessian(pos, result);
        return result;
    });
}

}
