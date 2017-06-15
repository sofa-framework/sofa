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

#include "Binding_MeshLoader.h"
#include "Binding_BaseLoader.h"

using namespace sofa::core::loader;
using namespace sofa::core;
using namespace sofa::core::objectmodel;

extern "C" PyObject * MeshLoader_reinit(PyObject *self, PyObject * /*args*/)
{
    MeshLoader* obj=down_cast<MeshLoader>(((PySPtr<Base>*)self)->object->toBaseLoader());
    obj->reinit();
    Py_RETURN_NONE;
}




SP_CLASS_METHODS_BEGIN(MeshLoader)
SP_CLASS_METHOD(MeshLoader,reinit)
/*
SP_CLASS_METHOD(MeshLoader,applyTranslation)
SP_CLASS_METHOD(MeshLoader,applyRotation)
SP_CLASS_METHOD(MeshLoader,applyScale)
SP_CLASS_METHOD(MeshLoader,setTranslation)
SP_CLASS_METHOD(MeshLoader,setRotation)
SP_CLASS_METHOD(MeshLoader,setScale)
SP_CLASS_METHOD(MeshLoader,getTranslation)
SP_CLASS_METHOD(MeshLoader,getRotation)
SP_CLASS_METHOD(MeshLoader,getScale)
*/
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(MeshLoader,MeshLoader,BaseLoader)


