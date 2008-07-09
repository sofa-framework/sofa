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
#include <sofa/component/topology/Edge2QuadTopologicalMapping.inl>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/QuadSetTopology.h>
#include <sofa/component/topology/EdgeSetTopology.h>

#include <sofa/core/componentmodel/topology/TopologicalMapping.h>
#include <sofa/core/componentmodel/topology/BaseTopology.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Vec3Types.h>

#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using namespace core;
using namespace core::componentmodel::topology;
using namespace sofa::core::componentmodel::behavior;

using namespace sofa::component::topology;

SOFA_DECL_CLASS(Edge2QuadTopologicalMapping)

// Register in the Factory
int Edge2QuadTopologicalMappingClass = core::RegisterObject("Special case of mapping where EdgeSetTopology is converted to QuadSetTopology")
#ifndef SOFA_FLOAT
        .add< Edge2QuadTopologicalMapping< EdgeSetTopology<Rigid3dTypes>, QuadSetTopology<Vec3dTypes> > >()
        .add< Edge2QuadTopologicalMapping< EdgeSetTopology<Rigid2dTypes>, QuadSetTopology<Vec2dTypes> > >()
#endif
#ifndef SOFA_DOUBLE
        .add< Edge2QuadTopologicalMapping< EdgeSetTopology<Rigid3fTypes>, QuadSetTopology<Vec3fTypes> > >()
        .add< Edge2QuadTopologicalMapping< EdgeSetTopology<Rigid2fTypes>, QuadSetTopology<Vec2fTypes> > >()
#endif
        ;
#ifndef SOFA_FLOAT
template class Edge2QuadTopologicalMapping< EdgeSetTopology<Rigid3dTypes>, QuadSetTopology<Vec3dTypes> >;
template class Edge2QuadTopologicalMapping< EdgeSetTopology<Rigid2dTypes>, QuadSetTopology<Vec2dTypes> >;
#endif
#ifndef SOFA_DOUBLE
template class Edge2QuadTopologicalMapping< EdgeSetTopology<Rigid3fTypes>, QuadSetTopology<Vec3fTypes> >;
template class Edge2QuadTopologicalMapping< EdgeSetTopology<Rigid2fTypes>, QuadSetTopology<Vec2fTypes> >;
#endif

;

} // namespace topology

} // namespace component

} // namespace sofa

