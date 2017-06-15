/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include "Binding_SubsetMultiMapping.h"
#include "Binding_BaseMapping.h"

#include <sofa/core/BaseState.h>



extern "C" PyObject * SubsetMultiMapping3_to_3_addPoint(PyObject *self, PyObject * args)
{
    SubsetMultiMapping3_to_3* obj=down_cast<SubsetMultiMapping3_to_3>(((PySPtr<sofa::core::objectmodel::Base>*)self)->object->toBaseMapping());
    PyObject* pyState;
    int index;
    if (!PyArg_ParseTuple(args, "Oi",&pyState,&index))
    {
        PyErr_BadArgument();
        return NULL;
    }
    sofa::core::BaseState* state=((PySPtr<sofa::core::objectmodel::Base>*)pyState)->object->toBaseState();

    obj->addPoint(state,index);
    Py_RETURN_NONE;
}

SP_CLASS_METHODS_BEGIN(SubsetMultiMapping3_to_3)
SP_CLASS_METHOD(SubsetMultiMapping3_to_3,addPoint)
SP_CLASS_METHODS_END

