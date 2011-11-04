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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * FlowVisualModel.cpp
 *
 *  Created on: 18 févr. 2009
 *      Author: froy
 */

#include "FlowVisualModel.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(FlowVisualModel)

// Register in the Factory
int FlowVisualModelClass = core::RegisterObject("FlowVisualModel")
#ifndef SOFA_FLOAT
        .add< FlowVisualModel<Vec3dTypes> >()
//.add< FlowVisualModel<Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
//.add< FlowVisualModel<Vec3fTypes> >()
//.add< FlowVisualModel<Vec2fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class FlowVisualModel<defaulttype::Vec3dTypes>;
//template class FlowVisualModel<defaulttype::Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class FlowVisualModel<defaulttype::Vec3fTypes>;
//template class FlowVisualModel<defaulttype::Vec2fTypes>;
#endif


}

}

}
