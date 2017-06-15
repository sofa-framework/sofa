/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "StructDataWidget.h"
#include <sofa/helper/Factory.inl>
#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::helper::Creator;
using sofa::helper::fixed_array;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(StructDataWidget);

Creator<DataWidgetFactory, SimpleDataWidget< RigidCoord<2,float> > > DWClass_RigidCoord2f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< RigidCoord<2,double> > > DWClass_RigidCoord2d("default",true);
//Creator<DataWidgetFactory, SimpleDataWidget< RigidDeriv<2,float> > > DWClass_RigidDeriv2f("default",true);
//Creator<DataWidgetFactory, SimpleDataWidget< RigidDeriv<2,double> > > DWClass_RigidDeriv2d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< RigidMass<2,float> > > DWClass_RigidMass2f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< RigidMass<2,double> > > DWClass_RigidMass2d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< RigidCoord<3,float> > > DWClass_RigidCoord3f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< RigidCoord<3,double> > > DWClass_RigidCoord3d("default",true);
//Creator<DataWidgetFactory, SimpleDataWidget< RigidDeriv<3,float> > > DWClass_RigidDeriv3f("default",true);
//Creator<DataWidgetFactory, SimpleDataWidget< RigidDeriv<3,double> > > DWClass_RigidDeriv3d("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< RigidMass<3,float> > > DWClass_RigidMass3f("default",true);
Creator<DataWidgetFactory, SimpleDataWidget< RigidMass<3,double> > > DWClass_RigidMass3d("default",true);

//Creator<DataWidgetFactory, SimpleDataWidget< sofa::core::loader::Material > > DWClass_MeshMaterial("default",true);

} // namespace qt

} // namespace gui

} // namespace sofa
