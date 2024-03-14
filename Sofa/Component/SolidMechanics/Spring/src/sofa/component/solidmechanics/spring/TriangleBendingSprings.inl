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

#include <sofa/component/solidmechanics/spring/TriangleBendingSprings.h>
#include <sofa/component/solidmechanics/spring/StiffSpringForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <iostream>

namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
TriangleBendingSprings<DataTypes>::TriangleBendingSprings()
    : l_topology(initLink("topology", "link to the topology container"))
{
    
}


template<class DataTypes>
TriangleBendingSprings<DataTypes>::~TriangleBendingSprings()
{}

template<class DataTypes>
void TriangleBendingSprings<DataTypes>::addSpring( unsigned a, unsigned b )
{
    const VecCoord& x =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    Real s = (Real)this->ks.getValue();
    Real d = (Real)this->kd.getValue();
    Real l = (x[a]-x[b]).norm();
    this->SpringForceField<DataTypes>::addSpring(a,b, s, d, l );
}

template<class DataTypes>
void TriangleBendingSprings<DataTypes>::registerTriangle( unsigned a, unsigned b, unsigned c, std::map<IndexPair, unsigned>& edgeMap)
{
    using namespace std;
    {
        const IndexPair edge(a<b ? a : b,a<b ? b : a);
        const unsigned opposite = c;
        if( edgeMap.find( edge ) == edgeMap.end() )
        {
            edgeMap[edge] = opposite;
        }
        else
        {
            // create a spring between the opposite
            this->addSpring(opposite,edgeMap[edge]);
        }
    }

    {
        const IndexPair edge(b<c ? b : c,b<c ? c : b);
        const unsigned opposite = a;
        if( edgeMap.find( edge ) == edgeMap.end() )
        {
            edgeMap[edge] = opposite;
        }
        else
        {
            // create a spring between the opposite
            this->addSpring(opposite,edgeMap[edge]);
        }
    }

    {
        const IndexPair edge(c<a ? c : a,c<a ? a : c);
        const unsigned  opposite = b;
        if( edgeMap.find( edge ) == edgeMap.end() )
        {
            edgeMap[edge] = opposite;
        }
        else
        {
            // create a spring between the opposite
            this->addSpring(opposite,edgeMap[edge]);
        }
    }

}



template<class DataTypes>
void TriangleBendingSprings<DataTypes>::init()
{
    this->mstate1 = this->mstate2 = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>( this->getContext()->getMechanicalState() );
    StiffSpringForceField<DataTypes>::clear();

    // Set the bending springs

    std::map< IndexPair, unsigned > edgeMap;

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    sofa::core::topology::BaseMeshTopology* topology = l_topology.get();
    if (topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = topology->getTriangles();
    for( unsigned i= 0; i<triangles.size(); ++i )
    {
        const sofa::core::topology::BaseMeshTopology::Triangle& face = triangles[i];
        {
            registerTriangle( face[0], face[1], face[2], edgeMap );
        }

    }

    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = topology->getQuads();
    for( unsigned i= 0; i<quads.size(); ++i )
    {
        const sofa::core::topology::BaseMeshTopology::Quad& face = quads[i];
        {
            registerTriangle( face[0], face[1], face[2], edgeMap );
            registerTriangle( face[0], face[2], face[3], edgeMap );
        }

    }

    // init the parent class
    StiffSpringForceField<DataTypes>::init();

}


} // namespace sofa::component::solidmechanics::spring
