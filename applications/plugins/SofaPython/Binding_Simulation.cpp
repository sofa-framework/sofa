/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "PythonFactory.h"
#include "Binding_Simulation.h"
#include "Binding_Base.h"
#include "PythonToSofa.inl"
#include <sofa/simulation/Simulation.h>

static inline sofa::simulation::Simulation* get_simulation(PyObject* obj) {
    return sofa::py::unwrap<sofa::simulation::Simulation>(obj);
}

static PyObject * Simulation_init(PyObject *self, PyObject * args) {
    sofa::simulation::Simulation* simulation  = get_simulation(self);

    PyObject * obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    sofa::simulation::Node* root = sofa::py::unwrap<sofa::simulation::Node>(obj);
    simulation->init(root);
    Py_RETURN_NONE;
}

static PyObject * Simulation_exportXML(PyObject *self, PyObject * args) {
    sofa::simulation::Simulation* simulation  = get_simulation(self);

    PyObject * obj;
    char *filename;
    if (!PyArg_ParseTuple(args, "Os", &obj, &filename)) {
        return NULL;
    }
    sofa::simulation::Node* root = sofa::py::unwrap<sofa::simulation::Node>(obj);
    simulation->exportXML(root, filename);
    Py_RETURN_NONE;
}

static PyObject * Simulation_exportGraph(PyObject *self, PyObject * args) {
    sofa::simulation::Simulation* simulation  = get_simulation(self);

    PyObject * obj;
    char *filename;
    if (!PyArg_ParseTuple(args, "Os", &obj, &filename)) {
        return NULL;
    }
    sofa::simulation::Node* root = sofa::py::unwrap<sofa::simulation::Node>(obj);
    simulation->exportGraph(root, filename);
    Py_RETURN_NONE;
}

SP_CLASS_METHODS_BEGIN(Simulation)
SP_CLASS_METHOD(Simulation, init)
SP_CLASS_METHOD(Simulation, exportXML)
SP_CLASS_METHOD(Simulation, exportGraph)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(Simulation,sofa::simulation::Simulation,Base)



