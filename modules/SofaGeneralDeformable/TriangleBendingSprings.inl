/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
//
// C++ Implementation: TriangleBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_TRIANGLEBENDINGSPRINGS_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_TRIANGLEBENDINGSPRINGS_INL

#include <SofaGeneralDeformable/TriangleBendingSprings.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
TriangleBendingSprings<DataTypes>::TriangleBendingSprings()
{
    //serr<<"TriangleBendingSprings<DataTypes>::TriangleBendingSprings"<<sendl;
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
    //sout<<"=================================TriangleBendingSprings<DataTypes>::addSpring "<<a<<", "<<b<<sendl;
}

template<class DataTypes>
void TriangleBendingSprings<DataTypes>::registerTriangle( unsigned a, unsigned b, unsigned c, std::map<IndexPair, unsigned>& edgeMap)
{
    //sout<<"=================================TriangleBendingSprings<DataTypes>::registerTriangle "<<a<<", "<<b<<", "<<c<<sendl;
    using namespace std;
    {
        IndexPair edge(a<b ? a : b,a<b ? b : a);
        unsigned opposite = c;
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
        IndexPair edge(b<c ? b : c,b<c ? c : b);
        unsigned opposite = a;
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
        IndexPair edge(c<a ? c : a,c<a ? a : c);
        unsigned  opposite = b;
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
    sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();
    assert( topology );

    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = topology->getTriangles();
    //sout<<"==================================TriangleBendingSprings<DataTypes>::init(), triangles size = "<<triangles.size()<<sendl;
    for( unsigned i= 0; i<triangles.size(); ++i )
    {
        const sofa::core::topology::BaseMeshTopology::Triangle& face = triangles[i];
        {
            registerTriangle( face[0], face[1], face[2], edgeMap );
        }

    }

    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = topology->getQuads();
    //sout<<"==================================TriangleBendingSprings<DataTypes>::init(), quad size = "<<topology->getQuads().size()<<sendl;
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


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_TRIANGLEBENDINGSPRINGS_INL */
