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
#include "ManifoldEdgeSetTopologyAlgorithms.inl"

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
SOFA_DECL_CLASS(ManifoldEdgeSetTopologyAlgorithms)
int ManifoldEdgeSetTopologyAlgorithmsClass = core::RegisterObject("ManifoldEdge set topology algorithms")
#ifdef SOFA_FLOAT
        .add< ManifoldEdgeSetTopologyAlgorithms<Vec3fTypes> >(true) // default template
#else
        .add< ManifoldEdgeSetTopologyAlgorithms<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< ManifoldEdgeSetTopologyAlgorithms<Vec3fTypes> >() // default template
#endif
#endif
#ifndef SOFA_FLOAT
        .add< ManifoldEdgeSetTopologyAlgorithms<Vec2dTypes> >()
        .add< ManifoldEdgeSetTopologyAlgorithms<Vec1dTypes> >()
        .add< ManifoldEdgeSetTopologyAlgorithms<Rigid3dTypes> >()
        .add< ManifoldEdgeSetTopologyAlgorithms<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ManifoldEdgeSetTopologyAlgorithms<Vec2fTypes> >()
        .add< ManifoldEdgeSetTopologyAlgorithms<Vec1fTypes> >()
        .add< ManifoldEdgeSetTopologyAlgorithms<Rigid3fTypes> >()
        .add< ManifoldEdgeSetTopologyAlgorithms<Rigid2fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class ManifoldEdgeSetTopologyAlgorithms<Vec3dTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Vec2dTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Vec1dTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Rigid3dTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class ManifoldEdgeSetTopologyAlgorithms<Vec3fTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Vec2fTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Vec1fTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Rigid3fTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Rigid2fTypes>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

