/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_MANIFOLD_TOPOLOGY_EDGESETTOPOLOGYALGORITHMS_H
#define SOFA_MANIFOLD_TOPOLOGY_EDGESETTOPOLOGYALGORITHMS_H

#include <ManifoldTopologies/config.h>
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

    virtual void init() override;

private:
    ManifoldEdgeSetTopologyContainer*					m_container;
    ManifoldEdgeSetTopologyModifier*					m_modifier;
    ManifoldEdgeSetGeometryAlgorithms< DataTypes >*		m_geometryAlgorithms;
};

#if !defined(SOFA_MANIFOLD_TOPOLOGY_EDGESETTOPOLOGYALGORITHMS_CPP)
extern template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetTopologyAlgorithms<sofa::defaulttype::Vec3Types>;
extern template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetTopologyAlgorithms<sofa::defaulttype::Vec2Types>;
extern template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetTopologyAlgorithms<sofa::defaulttype::Vec1Types>;
extern template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetTopologyAlgorithms<sofa::defaulttype::Rigid3Types>;
extern template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetTopologyAlgorithms<sofa::defaulttype::Rigid2Types>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_MANIFOLD_TOPOLOGY_EDGESETTOPOLOGYALGORITHMS_H
