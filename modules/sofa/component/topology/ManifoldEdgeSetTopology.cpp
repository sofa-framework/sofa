/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/topology/ManifoldEdgeSetTopology.h>
#include <sofa/component/topology/ManifoldEdgeSetTopology.inl>

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

SOFA_DECL_CLASS(ManifoldEdgeSetTopology)

// factory related stuff

int ManifoldEdgeSetTopologyClass = core::RegisterObject("Manofold Edge set topology")
#ifndef SOFA_FLOAT
        .add< ManifoldEdgeSetTopology<Vec3dTypes> >()
        .add< ManifoldEdgeSetTopology<Vec2dTypes> >()
        .add< ManifoldEdgeSetTopology<Vec1dTypes> >()
        .add< ManifoldEdgeSetTopology<Rigid3dTypes> >()
        .add< ManifoldEdgeSetTopology<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ManifoldEdgeSetTopology<Vec3fTypes> >()
        .add< ManifoldEdgeSetTopology<Vec2fTypes> >()
        .add< ManifoldEdgeSetTopology<Vec1fTypes> >()
        .add< ManifoldEdgeSetTopology<Rigid3fTypes> >()
        .add< ManifoldEdgeSetTopology<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class ManifoldEdgeSetTopology<Vec3dTypes>;
template class ManifoldEdgeSetTopology<Vec2dTypes>;
template class ManifoldEdgeSetTopology<Vec1dTypes>;
template class ManifoldEdgeSetTopology<Rigid3dTypes>;
template class ManifoldEdgeSetTopology<Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class ManifoldEdgeSetTopology<Vec3fTypes>;
template class ManifoldEdgeSetTopology<Vec2fTypes>;
template class ManifoldEdgeSetTopology<Vec1fTypes>;
template class ManifoldEdgeSetTopology<Rigid3fTypes>;
template class ManifoldEdgeSetTopology<Rigid2fTypes>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

