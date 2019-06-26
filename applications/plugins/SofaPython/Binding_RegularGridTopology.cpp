/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "Binding_RegularGridTopology.h"
#include "Binding_GridTopology.h"
#include "Binding_Vector.h"
#include "PythonToSofa.inl"

using namespace sofa::component::topology;
using namespace sofa::core::objectmodel;
using sofa::defaulttype::Vector3;

/// getting a RegularGridTopology* from a PyObject*
static inline RegularGridTopology* get_RegularGridTopology(PyObject* obj) {
    return sofa::py::unwrap<RegularGridTopology>(obj);
}

static constexpr const char* setPos_DOC =
R"DOC(
Set the position of the grid from its bounding box.

:param bbox: The bounding box of the regular grid (xmin, xmax, ymin, ymax, zmin, zmax).
:type path: tuple

:return: A list of strings populated with the supported GUI types
:rtype: list
)DOC";
static PyObject * RegularGridTopology_setPos(PyObject *self, PyObject * args)
{
    RegularGridTopology* obj = get_RegularGridTopology( self );
    double xmin,xmax,ymin,ymax,zmin,zmax;
    if (!PyArg_ParseTuple(args, "dddddd",&xmin,&xmax,&ymin,&ymax,&zmin,&zmax))
    {
        return nullptr;
    }
    obj->setPos(xmin,xmax,ymin,ymax,zmin,zmax);
    Py_RETURN_NONE;
}

static constexpr const char* getNodePosition_DOC =
R"DOC(
Get the position of node of the grid from its index.

:param index: The index of the node.
:type index: int

:return: A vector3 containing the position of the given node index.
:rtype: Vector3
)DOC";
static PyObject * RegularGridTopology_getNodePosition(PyObject *self, PyObject * args)
{
    RegularGridTopology *obj = get_RegularGridTopology(self);
    int index;
    if (!PyArg_ParseTuple(args, "i", &index)) {
        Py_RETURN_NONE;
    }

    const auto p = obj->getPoint(index);

    auto *vec = new Vector3(p);
    return SP_BUILD_PYPTR(Vector3, Vector3, vec, true);
}


SP_CLASS_METHODS_BEGIN(RegularGridTopology)
SP_CLASS_METHOD_DOC(RegularGridTopology,setPos, setPos_DOC)
SP_CLASS_METHOD_DOC(RegularGridTopology,getNodePosition, getNodePosition_DOC)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(RegularGridTopology,RegularGridTopology,GridTopology)



