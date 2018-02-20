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
#ifndef BINDING_BASE_H
#define BINDING_BASE_H

#include "PythonMacros.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/Factory.inl>
SP_DECLARE_CLASS_TYPE(Base)

template<typename DataType>
class DataCreator : public sofa::helper::BaseCreator<sofa::core::objectmodel::BaseData>
{
public:
    virtual sofa::core::objectmodel::BaseData* createInstance(sofa::helper::NoArgument) override { return new sofa::core::objectmodel::Data<DataType>(); }
    virtual const std::type_info& type() override { return typeid(sofa::core::objectmodel::BaseData);}
};

//sofa::core::objectmodel::BaseData * helper_addNewData(PyObject *args, sofa::core::objectmodel::Base* obj);
sofa::core::objectmodel::BaseData * helper_addNewData(PyObject *args, PyObject * kw, sofa::core::objectmodel::Base * obj);

#endif // BINDING_BASE_H
