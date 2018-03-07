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
SOFA_DECL_CLASS(ManifoldTriangleSetTopologyAlgorithms)
int ManifoldTriangleSetTopologyAlgorithmsClass = core::RegisterObject("ManifoldTriangle set topology algorithms")
#ifdef SOFA_FLOAT
        .add< ManifoldTriangleSetTopologyAlgorithms<Vec3fTypes> >(true) // default template
#else
        .add< ManifoldTriangleSetTopologyAlgorithms<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< ManifoldTriangleSetTopologyAlgorithms<Vec3fTypes> >() // default template
#endif
#endif

#ifndef SOFA_FLOAT
        .add< ManifoldTriangleSetTopologyAlgorithms<Vec2dTypes> >()
        .add< ManifoldTriangleSetTopologyAlgorithms<Vec1dTypes> >()
        //	.add< ManifoldTriangleSetTopologyAlgorithms<Rigid3dTypes> >()
        //	.add< ManifoldTriangleSetTopologyAlgorithms<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ManifoldTriangleSetTopologyAlgorithms<Vec2fTypes> >()
        .add< ManifoldTriangleSetTopologyAlgorithms<Vec1fTypes> >()
        //	.add< ManifoldTriangleSetTopologyAlgorithms<Rigid3fTypes> >()
        //	.add< ManifoldTriangleSetTopologyAlgorithms<Rigid2fTypes> >()
#endif
        ;
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

