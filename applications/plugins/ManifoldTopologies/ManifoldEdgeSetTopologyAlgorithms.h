/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGYALGORITHMS_H

#include <SofaBaseTopology/EdgeSetTopologyAlgorithms.h>

namespace sofa
{

namespace component
{

namespace topology
{
class ManifoldEdgeSetTopologyContainer;

class ManifoldEdgeSetTopologyModifier;

template < class DataTypes >
class ManifoldEdgeSetGeometryAlgorithms;

/**
* A class that performs topology algorithms on an ManifoldEdgeSet.
*/
template < class DataTypes >
class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetTopologyAlgorithms : public EdgeSetTopologyAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ManifoldEdgeSetTopologyAlgorithms,DataTypes),SOFA_TEMPLATE(EdgeSetTopologyAlgorithms,DataTypes));

    ManifoldEdgeSetTopologyAlgorithms()
        : EdgeSetTopologyAlgorithms<DataTypes>()
    {}

    virtual ~ManifoldEdgeSetTopologyAlgorithms() {}

    virtual void init();

private:
    ManifoldEdgeSetTopologyContainer*					m_container;
    ManifoldEdgeSetTopologyModifier*					m_modifier;
    ManifoldEdgeSetGeometryAlgorithms< DataTypes >*		m_geometryAlgorithms;
};
} // namespace topology

} // namespace component

} // namespace sofa

#endif
