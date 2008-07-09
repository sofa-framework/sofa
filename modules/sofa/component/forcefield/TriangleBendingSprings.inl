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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGLEBENDINGSPRINGS_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGLEBENDINGSPRINGS_INL

#include <sofa/component/forcefield/TriangleBendingSprings.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace core::componentmodel::behavior;

template<class DataTypes>
TriangleBendingSprings<DataTypes>::TriangleBendingSprings()
    : dof(NULL)
{
    //std::cerr<<"TriangleBendingSprings<DataTypes>::TriangleBendingSprings"<<std::endl;
}


template<class DataTypes>
TriangleBendingSprings<DataTypes>::~TriangleBendingSprings()
{}

template<class DataTypes>
void TriangleBendingSprings<DataTypes>::addSpring( unsigned a, unsigned b )
{
    const VecCoord& x = *dof->getX();
    Real s = (Real)this->ks.getValue();
    Real d = (Real)this->kd.getValue();
    Real l = (x[a]-x[b]).norm();
    this->SpringForceField<DataTypes>::addSpring(a,b, s, d, l );
    //std::cout<<"=================================TriangleBendingSprings<DataTypes>::addSpring "<<a<<", "<<b<<std::endl;
}

template<class DataTypes>
void TriangleBendingSprings<DataTypes>::registerTriangle( unsigned a, unsigned b, unsigned c, std::map<IndexPair, unsigned>& edgeMap)
{
    //std::cout<<"=================================TriangleBendingSprings<DataTypes>::registerTriangle "<<a<<", "<<b<<", "<<c<<std::endl;
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
    dof = dynamic_cast<MechanicalObject<DataTypes>*>( this->getContext()->getMechanicalState() );
    assert(dof);
    this->mstate1 = this->mstate2 = dof;
    StiffSpringForceField<DataTypes>::clear();

    // Set the bending springs

    std::map< IndexPair, unsigned > edgeMap;
    sofa::core::componentmodel::topology::BaseMeshTopology* topology = dynamic_cast<sofa::core::componentmodel::topology::BaseMeshTopology*>( this->getContext()->getTopology() );
    assert( topology );

    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = topology->getTriangles();
    //std::cout<<"==================================TriangleBendingSprings<DataTypes>::init(), triangles size = "<<triangles.size()<<std::endl;
    for( unsigned i= 0; i<triangles.size(); ++i )
    {
        const sofa::core::componentmodel::topology::BaseMeshTopology::Triangle& face = triangles[i];
        {
            registerTriangle( face[0], face[1], face[2], edgeMap );
        }

    }

    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = topology->getQuads();
    //std::cout<<"==================================TriangleBendingSprings<DataTypes>::init(), quad size = "<<topology->getQuads().size()<<std::endl;
    for( unsigned i= 0; i<quads.size(); ++i )
    {
        const sofa::core::componentmodel::topology::BaseMeshTopology::Quad& face = quads[i];
        {
            registerTriangle( face[0], face[1], face[2], edgeMap );
            registerTriangle( face[0], face[2], face[3], edgeMap );
        }

    }

    // init the parent class
    StiffSpringForceField<DataTypes>::init();

}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
