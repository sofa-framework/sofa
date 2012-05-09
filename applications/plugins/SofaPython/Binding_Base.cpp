/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseData.h>
using namespace sofa::core::objectmodel;

#include "Binding_Base.h"
#include "Binding_BaseData.h"

extern "C" PyObject * Base_findData(PyObject *self, PyObject * args)
{
    Base* obj=dynamic_cast<Base*>(((PySPtr<Base>*)self)->object.get());
    char *dataName;
    if (!PyArg_ParseTuple(args, "s",&dataName))
        return 0;
    BaseData * data = obj->findData(dataName);
    if (!data)
    {
        PyErr_BadArgument();
        return 0;
    }
    return SP_BUILD_PYPTR(BaseData,data,false);
}


SP_CLASS_METHODS_BEGIN(Base)
SP_CLASS_METHOD(Base,findData)
SP_CLASS_METHODS_END


SP_CLASS_DATA_ATTRIBUTE(Base,name)

SP_CLASS_ATTRS_BEGIN(Base)
SP_CLASS_ATTR(Base,name)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_SPTR_ATTR(Base)
