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

static constexpr const char* init_DOC =
R"DOC(
Initialize the simulation.

:param root: The simulation's root node
:type root: sofa::simulation::Node
)DOC";
static PyObject * Simulation_init(PyObject *self, PyObject * args) {
    sofa::simulation::Simulation* simulation  = get_simulation(self);

    PyObject * obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return nullptr;
    }
    sofa::simulation::Node* root = sofa::py::unwrap<sofa::simulation::Node>(obj);
    simulation->init(root);
    Py_RETURN_NONE;
}

static constexpr const char* unload_DOC =
R"DOC(
Unload the root from the current simulation.

:param root: The simulation's root node
:type root: sofa::simulation::Node
)DOC";
static PyObject * Simulation_unload(PyObject *self, PyObject * args) {
    sofa::simulation::Simulation* simulation  = get_simulation(self);
    PyObject * obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return nullptr;
    }

    sofa::simulation::Node* node = sofa::py::unwrap<sofa::simulation::Node>(obj);
    simulation->unload(node);
    Py_RETURN_NONE;
}

static constexpr const char* exportXML_DOC =
R"DOC(
Export the node's graph as an XML file

:param node: The node to be exported
:param filename: The filename of the resulting XML file
:type node: sofa::simulation::Node
:type filename: str
)DOC";
static PyObject * Simulation_exportXML(PyObject *self, PyObject * args) {
    sofa::simulation::Simulation* simulation  = get_simulation(self);

    PyObject * obj;
    char *filename;
    if (!PyArg_ParseTuple(args, "Os", &obj, &filename)) {
        return nullptr;
    }
    sofa::simulation::Node* root = sofa::py::unwrap<sofa::simulation::Node>(obj);
    simulation->exportXML(root, filename);
    Py_RETURN_NONE;
}

static constexpr const char* exportGraph_DOC =
        R"DOC(
Export the node's graph as a text file

:param node: The node to be exported
:param filename: The filename of the resulting text file
:type node: sofa::simulation::Node
:type filename: str
)DOC";
static PyObject * Simulation_exportGraph(PyObject *self, PyObject * args) {
    sofa::simulation::Simulation* simulation  = get_simulation(self);

    PyObject * obj;
    char *filename;
    if (!PyArg_ParseTuple(args, "Os", &obj, &filename)) {
        return nullptr;
    }
    sofa::simulation::Node* root = sofa::py::unwrap<sofa::simulation::Node>(obj);
    simulation->exportGraph(root, filename);
    Py_RETURN_NONE;
}

SP_CLASS_METHODS_BEGIN(Simulation)
SP_CLASS_METHOD_DOC(Simulation, init, init_DOC)
SP_CLASS_METHOD_DOC(Simulation, unload, unload_DOC)
SP_CLASS_METHOD_DOC(Simulation, exportXML, exportXML_DOC)
SP_CLASS_METHOD_DOC(Simulation, exportGraph, exportGraph_DOC)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(Simulation,sofa::simulation::Simulation,Base)



