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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_INL

#include <sofa/component/topology/QuadSetTopology.h>

#include <sofa/component/topology/QuadSetTopologyAlgorithms.inl>
#include <sofa/component/topology/QuadSetGeometryAlgorithms.inl>
#include <sofa/component/topology/QuadSetTopologyModifier.inl>

namespace sofa
{
namespace component
{
namespace topology
{

template<class DataTypes>
QuadSetTopology<DataTypes>::QuadSetTopology(MechanicalObject<DataTypes> *obj)
    : EdgeSetTopology<DataTypes>( obj)
{
}

template<class DataTypes>
void QuadSetTopology<DataTypes>::createComponents()
{
    this->m_topologyContainer = new QuadSetTopologyContainer(this);
    this->m_topologyModifier= new QuadSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new QuadSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new QuadSetGeometryAlgorithms<DataTypes>(this);
}

template<class DataTypes>
void QuadSetTopology<DataTypes>::init()
{
    EdgeSetTopology<DataTypes>::init();
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_QUADSETTOPOLOGY_INL
