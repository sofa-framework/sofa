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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGY_INL

#include <sofa/component/topology/HexahedronSetTopology.h>

#include <sofa/component/topology/HexahedronSetTopologyAlgorithms.inl>
#include <sofa/component/topology/HexahedronSetGeometryAlgorithms.inl>
#include <sofa/component/topology/HexahedronSetTopologyModifier.inl>

namespace sofa
{
namespace component
{
namespace topology
{

template<class DataTypes>
HexahedronSetTopology<DataTypes>::HexahedronSetTopology(MechanicalObject<DataTypes> *obj)
    : QuadSetTopology<DataTypes>( obj)
{
}

template<class DataTypes>
void HexahedronSetTopology<DataTypes>::createComponents()
{
    this->m_topologyContainer = new HexahedronSetTopologyContainer(this);
    this->m_topologyModifier= new HexahedronSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new HexahedronSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new HexahedronSetGeometryAlgorithms<DataTypes>(this);
}

template<class DataTypes>
void HexahedronSetTopology<DataTypes>::init()
{
    QuadSetTopology<DataTypes>::init();
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_HexahedronSetTOPOLOGY_INL
