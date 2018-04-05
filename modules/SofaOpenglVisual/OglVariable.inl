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
#ifndef SOFA_COMPONENT_VISUALMODEL_OGLVARIABLE_INL
#define SOFA_COMPONENT_VISUALMODEL_OGLVARIABLE_INL

#include <SofaOpenglVisual/OglVariable.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

//
//template<class DataTypes>
//OglVariable<DataTypes>::OglVariable()
//: value(initData(&value, DataTypes(), "value", "Set Uniform Value"))
//{
//    addAlias(&value, "values"); // some variable types hold multiple values, so we authorize both names for this attribute
//}
//
//template<class DataTypes>
//OglVariable<DataTypes>::~OglVariable()
//{
//}
//
//template<class DataTypes>
//void OglVariable<DataTypes>::init()
//{
//    OglShaderElement::init();
//}
//
//template<class DataTypes>
//void OglVariable<DataTypes>::initVisual()
//{
//    core::visual::VisualModel::initVisual();
//}
//
//template<class DataTypes>
//void OglVariable<DataTypes>::reinit()
//{
//    init();
//    initVisual();
//}

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_VISUALMODEL_OGLVARIABLE_H
