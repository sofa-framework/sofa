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
#include <sofa/component/topology/EdgeSetTopology.h>
#include <sofa/component/topology/EdgeSetTopology.inl>

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

SOFA_DECL_CLASS(EdgeSetTopology)

// factory related stuff

int EdgeSetTopologyClass = core::RegisterObject("Edge set topology")
#ifndef SOFA_FLOAT
        .add< EdgeSetTopology<Vec3dTypes> >()
        .add< EdgeSetTopology<Vec2dTypes> >()
        .add< EdgeSetTopology<Vec1dTypes> >()
        .add< EdgeSetTopology<Rigid3dTypes> >()
        .add< EdgeSetTopology<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< EdgeSetTopology<Vec3fTypes> >()
        .add< EdgeSetTopology<Vec2fTypes> >()
        .add< EdgeSetTopology<Vec1fTypes> >()
        .add< EdgeSetTopology<Rigid3fTypes> >()
        .add< EdgeSetTopology<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class EdgeSetTopology<Vec3dTypes>;
template class EdgeSetTopology<Vec2dTypes>;
template class EdgeSetTopology<Vec1dTypes>;
template class EdgeSetTopology<Rigid3dTypes>;
template class EdgeSetTopology<Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class EdgeSetTopology<Vec3fTypes>;
template class EdgeSetTopology<Vec2fTypes>;
template class EdgeSetTopology<Vec1fTypes>;
template class EdgeSetTopology<Rigid3fTypes>;
template class EdgeSetTopology<Rigid2fTypes>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

