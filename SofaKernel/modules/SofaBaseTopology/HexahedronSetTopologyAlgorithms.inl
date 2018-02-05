/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYALGORITHMS_INL

#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetTopologyModifier.h>
#include <SofaBaseTopology/HexahedronSetTopologyAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>
#include <algorithm>
#include <functional>

namespace sofa
{
namespace component
{
namespace topology
{

template<class DataTypes>
void HexahedronSetTopologyAlgorithms< DataTypes >::init()
{
    QuadSetTopologyAlgorithms< DataTypes >::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    this->getContext()->get(m_geometryAlgorithms);
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_HexahedronSetTOPOLOGY_INL
