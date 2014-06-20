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
#include "ManifoldTriangleSetTopologyAlgorithms.h"
#include "ManifoldTriangleSetTopologyAlgorithms.inl"

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
template class ManifoldTriangleSetTopologyAlgorithms<Vec3dTypes>;
template class ManifoldTriangleSetTopologyAlgorithms<Vec2dTypes>;
template class ManifoldTriangleSetTopologyAlgorithms<Vec1dTypes>;
//      template class ManifoldTriangleSetTopologyAlgorithms<Rigid3dTypes>;
//      template class ManifoldTriangleSetTopologyAlgorithms<Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class ManifoldTriangleSetTopologyAlgorithms<Vec3fTypes>;
template class ManifoldTriangleSetTopologyAlgorithms<Vec2fTypes>;
template class ManifoldTriangleSetTopologyAlgorithms<Vec1fTypes>;
//      template class ManifoldTriangleSetTopologyAlgorithms<Rigid3fTypes>;
//      template class ManifoldTriangleSetTopologyAlgorithms<Rigid2fTypes>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

