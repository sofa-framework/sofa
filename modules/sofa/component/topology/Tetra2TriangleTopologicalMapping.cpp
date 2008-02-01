/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/topology/Tetra2TriangleTopologicalMapping.inl>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/TriangleSetTopology.h>
#include <sofa/component/topology/TetrahedronSetTopology.h>

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

SOFA_DECL_CLASS(Tetra2TriangleTopologicalMapping)

// Register in the Factory
int Tetra2TriangleTopologicalMappingClass = core::RegisterObject("Special case of mapping where TetrahedronSetTopology is converted to TriangleSetTopology")
        .add< Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<Vec3dTypes>, TriangleSetTopology<Vec3dTypes> > >()
        .add< Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<Vec3fTypes>, TriangleSetTopology<Vec3fTypes> > >()
        .add< Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<Vec2dTypes>, TriangleSetTopology<Vec2dTypes> > >()
        .add< Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<Vec1dTypes>, TriangleSetTopology<Vec1dTypes> > >()
        .add< Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<Vec1fTypes>, TriangleSetTopology<Vec1fTypes> > >();

template class Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<Vec3dTypes>, TriangleSetTopology<Vec3dTypes> >;
template class Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<Vec3fTypes>, TriangleSetTopology<Vec3fTypes> >;
template class Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<Vec2dTypes>, TriangleSetTopology<Vec2dTypes> >;
template class Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<Vec1dTypes>, TriangleSetTopology<Vec1dTypes> >;
template class Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<Vec1fTypes>, TriangleSetTopology<Vec1fTypes> >;

;

} // namespace topology

} // namespace component

} // namespace sofa

