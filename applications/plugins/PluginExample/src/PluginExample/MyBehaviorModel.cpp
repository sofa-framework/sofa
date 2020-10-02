/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <PluginExample/MyBehaviorModel.h>

#include <sofa/core/ObjectFactory.h>


namespace sofa::component::behaviormodel
{

MyBehaviorModel::MyBehaviorModel()
    : d_customUnsignedData(initData(&d_customUnsignedData, unsigned(1),"Custom Unsigned Data","Example of unsigned data with custom widget"))
    , d_regularUnsignedData(initData(&d_regularUnsignedData, unsigned(1),"Unsigned Data","Example of unsigned data with standard widget"))
{
    d_customUnsignedData.setWidget("widget_myData");
}


MyBehaviorModel::~MyBehaviorModel()
{
}

void MyBehaviorModel::init()
{
}

void MyBehaviorModel::reinit()
{
}

void MyBehaviorModel::updatePosition(double dt)
{
    SOFA_UNUSED(dt);
}

int MyBehaviorModelClass = core::RegisterObject("Dummy component with a custom widget.").add< MyBehaviorModel >();


} // namespace sofa::component::behaviormodel

