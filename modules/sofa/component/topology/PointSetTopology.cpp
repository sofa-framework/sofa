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
#include <sofa/component/topology/PointSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(PointSetTopology)


PointSetTopologyContainer::PointSetTopologyContainer(core::componentmodel::topology::BaseTopology *top)
    : core::componentmodel::topology::TopologyContainer(top)
{
}


unsigned int PointSetTopologyContainer::getNumberOfVertices() const
{

    return m_basicTopology->getDOFNumber();
}


bool PointSetTopologyContainer::checkTopology() const
{
    //std::cout << "*** CHECK PointSetTopologyContainer ***" << std::endl;
    return true;
}

int PointSetTopologyClass = core::RegisterObject("Topology consisting of a set of points")

#ifndef SOFA_FLOAT
        .add< PointSetTopology<Vec3dTypes> >()
        .add< PointSetTopology<Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< PointSetTopology<Vec3fTypes> >()
        .add< PointSetTopology<Vec2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class PointSetTopologyModifier<Vec3dTypes>;
template class PointSetTopologyModifier<Vec2dTypes>;
template class PointSetTopology<Vec3dTypes>;
template class PointSetTopology<Vec2dTypes>;
template class PointSetGeometryAlgorithms<Vec3dTypes>;
template class PointSetGeometryAlgorithms<Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class PointSetTopologyModifier<Vec3fTypes>;
template class PointSetTopologyModifier<Vec2fTypes>;
template class PointSetTopology<Vec3fTypes>;
template class PointSetTopology<Vec2fTypes>;
template class PointSetGeometryAlgorithms<Vec3fTypes>;
template class PointSetGeometryAlgorithms<Vec2fTypes>;
#endif



} // namespace topology

} // namespace component

} // namespace sofa

