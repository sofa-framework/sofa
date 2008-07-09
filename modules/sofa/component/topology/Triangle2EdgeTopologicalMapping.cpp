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
#include <sofa/component/topology/Triangle2EdgeTopologicalMapping.inl>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/EdgeSetTopology.h>
#include <sofa/component/topology/TriangleSetTopology.h>

#include <sofa/core/componentmodel/topology/TopologicalMapping.h>
#include <sofa/core/componentmodel/topology/BaseTopology.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Vec3Types.h>


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

SOFA_DECL_CLASS(Triangle2EdgeTopologicalMapping)

// Register in the Factory
int Triangle2EdgeTopologicalMappingClass = core::RegisterObject("Special case of mapping where TriangleSetTopology is converted to EdgeSetTopology")
#ifndef SOFA_FLOAT
        .add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec3dTypes>, EdgeSetTopology<Vec3dTypes> > >()
        .add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec2dTypes>, EdgeSetTopology<Vec2dTypes> > >()
        .add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec1dTypes>, EdgeSetTopology<Vec1dTypes> > >()

//.add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec3dTypes>, ManifoldEdgeSetTopology<Vec3dTypes> > >()
//.add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec2dTypes>, ManifoldEdgeSetTopology<Vec2dTypes> > >()
//.add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec1dTypes>, ManifoldEdgeSetTopology<Vec1dTypes> > >()
#endif
#ifndef SOFA_DOUBLE
        .add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec3fTypes>, EdgeSetTopology<Vec3fTypes> > >()
        .add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec2fTypes>, EdgeSetTopology<Vec2fTypes> > >()
        .add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec1fTypes>, EdgeSetTopology<Vec1fTypes> > >()

//.add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec3fTypes>, ManifoldEdgeSetTopology<Vec3fTypes> > >()
//.add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec2fTypes>, ManifoldEdgeSetTopology<Vec2fTypes> > >()
//.add< Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec1fTypes>, ManifoldEdgeSetTopology<Vec1fTypes> > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec3dTypes>, EdgeSetTopology<Vec3dTypes> >;
template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec2dTypes>, EdgeSetTopology<Vec2dTypes> >;
template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec1dTypes>, EdgeSetTopology<Vec1dTypes> >;

//template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec3dTypes>, ManifoldEdgeSetTopology<Vec3dTypes> >;
//template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec2dTypes>, ManifoldEdgeSetTopology<Vec2dTypes> >;
//template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec1dTypes>, ManifoldEdgeSetTopology<Vec1dTypes> >;
#endif
#ifndef SOFA_DOUBLE
template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec3fTypes>, EdgeSetTopology<Vec3fTypes> >;
template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec2fTypes>, EdgeSetTopology<Vec2fTypes> >;
template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec1fTypes>, EdgeSetTopology<Vec1fTypes> >;

//template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec3fTypes>, ManifoldEdgeSetTopology<Vec3fTypes> >;
//template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec2fTypes>, ManifoldEdgeSetTopology<Vec2fTypes> >;
//template class Triangle2EdgeTopologicalMapping< TriangleSetTopology<Vec1fTypes>, ManifoldEdgeSetTopology<Vec1fTypes> >;
#endif

;

} // namespace topology

} // namespace component

} // namespace sofa

