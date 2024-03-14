/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/solidmechanics/spring/QuadBendingSprings.h>
#include <sofa/component/solidmechanics/spring/StiffSpringForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <iostream>

namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
QuadBendingSprings<DataTypes>::QuadBendingSprings()
    : StiffSpringForceField<DataTypes>()
    , localRange( initData(&localRange, type::Vec<2,int>(-1,-1), "localRange", "optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)" ) )
    , l_topology(initLink("topology", "link to the topology container"))
{
}


template<class DataTypes>
QuadBendingSprings<DataTypes>::~QuadBendingSprings()
{}

template<class DataTypes>
void QuadBendingSprings<DataTypes>::addSpring( unsigned a, unsigned b, std::set<IndexPair>& springSet )
{
    if (localRange.getValue()[0] >= 0)
    {
        if ((int)a < localRange.getValue()[0] && (int)b < localRange.getValue()[0]) return;
    }
    if (localRange.getValue()[1] >= 0)
    {
        if ((int)a > localRange.getValue()[1] && (int)b > localRange.getValue()[1]) return;
    }
    const IndexPair ab(a<b?a:b, a<b?b:a);
    if (springSet.find(ab) != springSet.end()) return;
    springSet.insert(ab);
    const VecCoord& x =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    Real s = (Real)this->ks.getValue();
    Real d = (Real)this->kd.getValue();
    Real l = (x[a]-x[b]).norm();
    this->SpringForceField<DataTypes>::addSpring(a,b, s, d, l );
}

template<class DataTypes>
void QuadBendingSprings<DataTypes>::registerEdge( IndexPair ab, IndexPair cd, std::map<IndexPair, IndexPair>& edgeMap, std::set<IndexPair>& springSet)
{
    if (ab.first > ab.second)
    {
        ab = std::make_pair(ab.second,ab.first);
        cd = std::make_pair(cd.second,cd.first);
    }
    if (edgeMap.find(ab) == edgeMap.end())
    {
        edgeMap[ab] = cd;
    }
    else
    {
        // create a spring between the opposite
        const IndexPair ef = edgeMap[ab];
        this->addSpring(cd.first, ef.first, springSet);
        this->addSpring(cd.second, ef.second, springSet);
    }
}

template<class DataTypes>
void QuadBendingSprings<DataTypes>::init()
{
    // Set the bending springs

    std::map< IndexPair, IndexPair > edgeMap;
    std::set< IndexPair > springSet;

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    sofa::core::topology::BaseMeshTopology* _topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = _topology->getQuads();
    for( unsigned i= 0; i<quads.size(); ++i )
    {
        const sofa::core::topology::BaseMeshTopology::Quad& face = quads[i];
        {
            registerEdge( std::make_pair(face[0], face[1]), std::make_pair(face[3], face[2]), edgeMap, springSet );
            registerEdge( std::make_pair(face[1], face[2]), std::make_pair(face[0], face[3]), edgeMap, springSet );
            registerEdge( std::make_pair(face[2], face[3]), std::make_pair(face[1], face[0]), edgeMap, springSet );
            registerEdge( std::make_pair(face[3], face[0]), std::make_pair(face[2], face[1]), edgeMap, springSet );
        }
    }

    // init the parent class
    StiffSpringForceField<DataTypes>::init();
}


} // namespace sofa::component::solidmechanics::spring
