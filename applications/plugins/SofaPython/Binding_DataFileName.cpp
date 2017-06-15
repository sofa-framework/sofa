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


#include "Binding_DataFileName.h"
#include "Binding_Data.h"


using namespace sofa::core::objectmodel;



SP_CLASS_ATTR_GET(DataFileName, fullPath)(PyObject *self, void*)
{
    DataFileName* dataFilename = down_cast<DataFileName>( ((PyPtr<BaseData>*)self)->object );
    return PyString_FromString(dataFilename->getFullPath().c_str());
}


SP_CLASS_ATTR_SET(DataFileName, fullPath)(PyObject */*self*/, PyObject * /*args*/, void*)
{
    SP_MESSAGE_ERROR("fullPath attribute is read only")
        PyErr_BadArgument();
    return -1;
}


SP_CLASS_ATTR_GET(DataFileName, relativePath)(PyObject *self, void*)
{
    DataFileName* dataFilename = down_cast<DataFileName>( ((PyPtr<BaseData>*)self)->object );
    return PyString_FromString(dataFilename->getRelativePath().c_str());
}


SP_CLASS_ATTR_SET(DataFileName, relativePath)(PyObject */*self*/, PyObject * /*args*/, void*)
{
    SP_MESSAGE_ERROR("relativePath attribute is read only")
        PyErr_BadArgument();
    return -1;
}




SP_CLASS_ATTRS_BEGIN(DataFileName)
SP_CLASS_ATTR(DataFileName,fullPath)
SP_CLASS_ATTR(DataFileName,relativePath)
SP_CLASS_ATTRS_END



SP_CLASS_METHODS_BEGIN(DataFileName)
SP_CLASS_METHODS_END



SP_CLASS_TYPE_PTR_ATTR(DataFileName,DataFileName,Data)

