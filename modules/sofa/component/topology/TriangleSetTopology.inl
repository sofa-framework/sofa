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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGY_INL

#include <sofa/component/topology/TriangleSetTopology.h>

#include <sofa/component/topology/TriangleSetTopologyAlgorithms.inl>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.inl>
#include <sofa/component/topology/TriangleSetTopologyModifier.inl>

namespace sofa
{
namespace component
{
namespace topology
{

template<class DataTypes>
TriangleSetTopology<DataTypes>::TriangleSetTopology(MechanicalObject<DataTypes> *obj)
    : EdgeSetTopology<DataTypes>( obj)
{}

template<class DataTypes>
void TriangleSetTopology<DataTypes>::createComponents()
{
    this->m_topologyContainer = new TriangleSetTopologyContainer(this);
    this->m_topologyModifier= new TriangleSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new TriangleSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new TriangleSetGeometryAlgorithms<DataTypes>(this);
}

template<class DataTypes>
void TriangleSetTopology<DataTypes>::init()
{
    EdgeSetTopology<DataTypes>::init();
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TriangleSetTOPOLOGY_INL
