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

static constexpr const char* getRegularGrid_DOC =
R"DOC(
Get the underneath regular grid object of this sparse grid.

:rtype: RegularGridTopology
)DOC";
static PyObject * SparseGridTopology_getRegularGrid(PyObject *self, PyObject * /*args*/)
{
    SparseGridTopology* obj = get_SparseGridTopology( self );
    if (!obj->_regularGrid.get())
        Py_RETURN_NONE;

    return SP_BUILD_PYPTR(RegularGridTopology,RegularGridTopology,obj->_regularGrid.get(),false);
}

static constexpr const char* getRegularGridCubeIndex_DOC =
R"DOC(
Get the index of a given cube in the underneath regular grid.

:param index: The index of the cube in the sparse grid.

:return: The index of the cube in the regular grid.
:rtype: int
)DOC";
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
        std::string msg = "SparseGridTopology.getRegularGridCubeIndex Trying to access the cube index '" + std::to_string(sparse_grid_cube_index) +
                "' but the topology only has '" + std::to_string(obj->getNbHexahedra()) + "' cubes.";
        SP_MESSAGE_ERROR(msg);
        return PyInt_FromLong(-1);
    }

    return PyInt_FromLong(obj->_indicesOfCubeinRegularGrid[sparse_grid_cube_index]);
}

static constexpr const char* getRegularGridNodeIndex_DOC =
R"DOC(
Get the index of a given node in the underneath regular grid.

:param index: The index of the node in the sparse grid.

:return: The index of the node in the regular grid.
:rtype: int
)DOC";
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
        std::string msg = "SparseGridTopology.getRegularGridNodeIndex Trying to access the node index '" + std::to_string(sparse_grid_node_index) +
                          "' but the topology only has '" + std::to_string(obj->getNbPoints()) + "' nodes.";
        SP_MESSAGE_ERROR(msg);
        return PyInt_FromLong(-1);
    }

    const auto pos = obj->getPointPos(sparse_grid_node_index);
    return PyInt_FromLong(obj->_regularGrid->findPoint(pos));
}

static constexpr const char* getNodePosition_DOC =
R"DOC(
Get the position of node of the grid from its index.

:param index: The index of the node.
:type index: int

:return: A vector3 containing the position of the given node index.
:rtype: Vector3
)DOC";
static PyObject * SparseGridTopology_getNodePosition(PyObject *self, PyObject * args)
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
        std::string msg = "SparseGridTopology.getNodePosition Trying to access the node index '" + std::to_string(sparse_grid_node_index) +
                          "' but the topology only has '" + std::to_string(obj->getNbPoints()) + "' nodes.";
        SP_MESSAGE_ERROR(msg);
        return PyInt_FromLong(-1);
    }

    const auto pos = obj->getPointPos(sparse_grid_node_index);
    auto *vec = new Vector3(pos);

    return SP_BUILD_PYPTR(Vector3,Vector3,vec,true);
}

static constexpr const char* getBoundaryCells_DOC =
R"DOC(
Get the indices of the cells in this grid that lie on the boundary (i.e. intersect the surface mesh).
:rtype: list
)DOC";
static PyObject * SparseGridTopology_getBoundaryCells(PyObject *self, PyObject * args)
{
    SOFA_UNUSED(args);
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

static constexpr const char* getBoundaryNodes_DOC =
R"DOC(
Get the indices of the nodes that forms the boundary of the grid (i.e. have less than 7 connected cells).
:rtype: list
)DOC";
static PyObject * SparseGridTopology_getBoundaryNodes(PyObject *self, PyObject * args)
{
    SOFA_UNUSED(args);
    SparseGridTopology* obj = get_SparseGridTopology( self );
    std::list<unsigned int> indices;
    for (unsigned int node_id = 0; node_id < (unsigned int) obj->getNbPoints(); ++node_id) {
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

static constexpr const char* findCube_DOC =
R"DOC(
Find the cube (grid cell) that contains the given position, and its local barycentric coordinates.

:param position: The position as [x,y,z].

:return: (cube_id, u, v, w) Where cube_id is the index of the cube containing the position, or -1 if no cells enclose the given coordinates.
u, v and w are the barycentric coordinates of the queried position inside the cube, or 0 if the position lies outside the sparse grid.
:rtype: tuple
)DOC";
static PyObject * SparseGridTopology_findCube(PyObject *self, PyObject * args)
{
    using Real = Vector3::value_type;
    SparseGridTopology* obj = get_SparseGridTopology( self );
    PyObject * pylist;
    if (!PyArg_ParseTuple(args, "O",&pylist))
    {
        SP_MESSAGE_ERROR("SparseGridTopology.findCube must have a position ([x, y, z]) as argument.");
        Py_RETURN_NONE;
    }

    if (not PyList_Check(pylist) or PyList_Size(pylist) != 3) {
        SP_MESSAGE_ERROR("SparseGridTopology.findCube must have a position ([x, y, z]) as argument.");
        Py_RETURN_NONE;
    }

    Vector3 pos;
    for (size_t i = 0; i < 3; ++i) {
        auto a = PyList_GetItem(pylist,i);
        if (PyInt_Check(a))
            pos[i] = (Real) PyInt_AsLong(a);
        else
            pos[i] = (Real) PyFloat_AsDouble(a);
    }

    Real fx=0., fy=0., fz=0.;
    Real dx=obj->_regularGrid->getDx()[0];
    Real dy=obj->_regularGrid->getDy()[1];
    Real dz=obj->_regularGrid->getDz()[2];

    int cube_id = obj->findCube(pos, fx, fy, fz);

    if (cube_id < 0) {
        // It may be a node
        const int node_id = obj->_regularGrid->findPoint(pos);
        if (node_id > -1) {
            // It is a node, find the first hexa connected to this node that is either inside or on the boundary
            const auto&  hexas = obj->_regularGrid->getHexahedraAroundVertex(node_id);
            for (const auto & hexa_id : hexas) {
                if (obj->_indicesOfRegularCubeInSparseGrid[hexa_id] > -1) {
                    Vector3 p = pos-obj->_regularGrid->d_p0.getValue();

                    SReal x = p[0]/dx;
                    SReal y = p[1]/dy;
                    SReal z = p[2]/dz;

                    int ix = int(x+1000000)-1000000; // Do not round toward 0...
                    int iy = int(y+1000000)-1000000;
                    int iz = int(z+1000000)-1000000;

                    fx = x-ix;
                    fy = y-iy;
                    fz = z-iz;
                    cube_id = obj->_indicesOfRegularCubeInSparseGrid[hexa_id];
                    break;
                }
            }
        }
    }

    PyObject* py_cube_id = PyInt_FromLong(cube_id);
    PyObject* py_fx = PyFloat_FromDouble(fx);
    PyObject* py_fy = PyFloat_FromDouble(fy);
    PyObject* py_fz = PyFloat_FromDouble(fz);

    PyObject* ret = PyTuple_Pack(4, py_cube_id, py_fx, py_fy, py_fz);

    return ret;
}

static constexpr const char* getNodeIndicesOfCube_DOC =
R"DOC(
Get the indices of the nodes of a given cube (grid cell).

:param index: The index of the cube.

:return: A list containing the indices of the cube nodes
:rtype: list
)DOC";
static PyObject * SparseGridTopology_getNodeIndicesOfCube(PyObject *self, PyObject * args)
{
    SparseGridTopology* obj = get_SparseGridTopology( self );
    int sparse_grid_cube_index;
    if (!PyArg_ParseTuple(args, "i",&sparse_grid_cube_index))
    {
        Py_RETURN_NONE;
    }

    if (sparse_grid_cube_index < 0)
        Py_RETURN_NONE;

    if ((size_t) sparse_grid_cube_index >= obj->getNbHexahedra()) {
        std::string msg = "SparseGridTopology.getNodeIndicesOfCube Trying to access the cube index '" + std::to_string(sparse_grid_cube_index) +
                          "' but the topology only has '" + std::to_string(obj->getNbHexahedra()) + "' cubes.";
        SP_MESSAGE_ERROR(msg);
        Py_RETURN_NONE;
    }

    const auto hexa = obj->getHexahedron((sofa::core::topology::Topology::HexaID) sparse_grid_cube_index);

    PyObject* py_indices = PyList_New(hexa.size());
    for (size_t i = 0; i < hexa.size(); ++i) {
        PyObject* py_node_index = PyInt_FromLong(hexa[i]);
        PyList_SetItem(py_indices, i, py_node_index);
    }

    return py_indices;
}

SP_CLASS_METHODS_BEGIN(SparseGridTopology)
                SP_CLASS_METHOD_DOC(SparseGridTopology, getRegularGrid, getRegularGrid_DOC)
                SP_CLASS_METHOD_DOC(SparseGridTopology, getRegularGridCubeIndex, getRegularGridCubeIndex_DOC)
                SP_CLASS_METHOD_DOC(SparseGridTopology, getRegularGridNodeIndex, getRegularGridNodeIndex_DOC)
                SP_CLASS_METHOD_DOC(SparseGridTopology, getNodePosition, getNodePosition_DOC)
                SP_CLASS_METHOD_DOC(SparseGridTopology, getBoundaryCells, getBoundaryCells_DOC)
                SP_CLASS_METHOD_DOC(SparseGridTopology, getBoundaryNodes, getBoundaryNodes_DOC)
                SP_CLASS_METHOD_DOC(SparseGridTopology, findCube, findCube_DOC)
                SP_CLASS_METHOD_DOC(SparseGridTopology, getNodeIndicesOfCube, getNodeIndicesOfCube_DOC)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(SparseGridTopology,SparseGridTopology,MeshTopology)