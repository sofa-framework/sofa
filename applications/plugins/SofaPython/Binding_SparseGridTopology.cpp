/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "Binding_SparseGridTopology.h"
#include "Binding_MeshTopology.h"
#include "Binding_RegularGridTopology.h"
#include <sofa/defaulttype/Vec.h>
#include "PythonToSofa.inl"
#include "Binding_Vector.h"

using sofa::component::topology::SparseGridTopology ;
using sofa::component::topology::RegularGridTopology ;
using sofa::defaulttype::Vector3;

/// getting a GridTopology* from a PyObject*
static inline SparseGridTopology* get_SparseGridTopology(PyObject* obj) {
    return sofa::py::unwrap<SparseGridTopology>(obj);
}

static PyObject * SparseGridTopology_getRegularGrid(PyObject *self, PyObject * /*args*/)
{
    SparseGridTopology* obj = get_SparseGridTopology( self );
    if (!obj->_regularGrid.get())
        Py_RETURN_NONE;

    return SP_BUILD_PYPTR(RegularGridTopology,RegularGridTopology,obj->_regularGrid.get(),true);
}

static PyObject * SparseGridTopology_getRegularGridCubeIndex(PyObject *self, PyObject * args)
{
    SparseGridTopology* obj = get_SparseGridTopology( self );
    int sparse_grid_cube_index;
    if (!PyArg_ParseTuple(args, "i",&sparse_grid_cube_index))
    {
        return PyInt_FromLong(-1);
    }

    if (sparse_grid_cube_index < 0)
        return PyInt_FromLong(-1);

    if ((size_t) sparse_grid_cube_index >= obj->getNbHexahedra()) {
        std::string msg = "Trying to access the cube index '" + std::to_string(sparse_grid_cube_index) +
                "' but the topology only has '" + std::to_string(obj->getNbHexahedra()) + "' cubes.";
        SP_MESSAGE_ERROR(msg);
        return PyInt_FromLong(-1);
    }

    return PyInt_FromLong(obj->_indicesOfCubeinRegularGrid[sparse_grid_cube_index]);
}

static PyObject * SparseGridTopology_getRegularGridNodeIndex(PyObject *self, PyObject * args)
{
    SparseGridTopology* obj = get_SparseGridTopology( self );
    int sparse_grid_node_index;
    if (!PyArg_ParseTuple(args, "i",&sparse_grid_node_index))
    {
        return PyInt_FromLong(-1);
    }

    if (sparse_grid_node_index < 0)
        return PyInt_FromLong(-1);

    if (sparse_grid_node_index >= obj->getNbPoints()) {
        std::string msg = "Trying to access the node index '" + std::to_string(sparse_grid_node_index) +
                          "' but the topology only has '" + std::to_string(obj->getNbPoints()) + "' nodes.";
        SP_MESSAGE_ERROR(msg);
        return PyInt_FromLong(-1);
    }

    const auto pos = obj->getPointPos(sparse_grid_node_index);



    return PyInt_FromLong(obj->_regularGrid->findPoint(pos));
}

static PyObject * SparseGridTopology_getPointPos(PyObject *self, PyObject * args)
{
    SparseGridTopology* obj = get_SparseGridTopology( self );
    int sparse_grid_node_index;
    if (!PyArg_ParseTuple(args, "i",&sparse_grid_node_index))
    {
        return PyInt_FromLong(-1);
    }

    if (sparse_grid_node_index < 0)
        return PyInt_FromLong(-1);

    if (sparse_grid_node_index >= obj->getNbPoints()) {
        std::string msg = "Trying to access the node index '" + std::to_string(sparse_grid_node_index) +
                          "' but the topology only has '" + std::to_string(obj->getNbPoints()) + "' nodes.";
        SP_MESSAGE_ERROR(msg);
        return PyInt_FromLong(-1);
    }

    const auto pos = obj->getPointPos(sparse_grid_node_index);
    auto *vec = new Vector3(pos);

    return SP_BUILD_PYPTR(Vector3,Vector3,vec,true);
}

static PyObject * SparseGridTopology_getBoundaryCells(PyObject *self, PyObject * args)
{
    SparseGridTopology* obj = get_SparseGridTopology( self );
    std::list<unsigned int> indices;
    for (unsigned int hexa_id = 0; hexa_id < obj->getNbHexahedra(); ++hexa_id) {
        if (obj->getType((int)hexa_id) == SparseGridTopology::BOUNDARY)
            indices.push_back(hexa_id);
    }

    PyObject * list = PyList_New(indices.size());
    unsigned int i = 0;
    for (auto index : indices) {
        PyObject *pyindex = PyInt_FromLong(index);
        PyList_SetItem(list, i, pyindex);
        ++i;
    }
    return list;
}

static PyObject * SparseGridTopology_getBoundaryNodes(PyObject *self, PyObject * args)
{
    SparseGridTopology* obj = get_SparseGridTopology( self );
    std::list<unsigned int> indices;
    for (unsigned int node_id = 0; node_id < obj->getNbPoints(); ++node_id) {
        auto nb_of_connected_hexa = obj->getHexahedraAroundVertex(node_id).size();
        if (nb_of_connected_hexa < 7)
            indices.push_back(node_id);
    }

    PyObject * list = PyList_New(indices.size());
    unsigned int i = 0;
    for (auto index : indices) {
        PyObject *pyindex = PyInt_FromLong(index);
        PyList_SetItem(list, i, pyindex);
        ++i;
    }

    return list;
}

SP_CLASS_METHODS_BEGIN(SparseGridTopology)
                SP_CLASS_METHOD(SparseGridTopology,getRegularGrid)
                SP_CLASS_METHOD(SparseGridTopology,getRegularGridCubeIndex)
                SP_CLASS_METHOD(SparseGridTopology,getRegularGridNodeIndex)
                SP_CLASS_METHOD(SparseGridTopology,getPointPos)
                SP_CLASS_METHOD(SparseGridTopology,getBoundaryCells)
                SP_CLASS_METHOD(SparseGridTopology,getBoundaryNodes)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(SparseGridTopology,SparseGridTopology,MeshTopology)