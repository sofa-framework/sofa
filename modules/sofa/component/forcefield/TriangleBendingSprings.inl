/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
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
#include <sofa/component/topology/MeshTopology.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

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
    //cerr<<"TriangleBendingSprings<DataTypes>::TriangleBendingSprings"<<endl;
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
    //cout<<"=================================TriangleBendingSprings<DataTypes>::addSpring "<<a<<", "<<b<<endl;
}

template<class DataTypes>
void TriangleBendingSprings<DataTypes>::registerTriangle( unsigned a, unsigned b, unsigned c, std::map<IndexPair, unsigned>& edgeMap)
{
    //cout<<"=================================TriangleBendingSprings<DataTypes>::registerTriangle "<<a<<", "<<b<<", "<<c<<endl;
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
    //cout<<"==================================TriangleBendingSprings<DataTypes>::init(), dof size = "<<dof->getX()->size()<<endl;

    // Set the bending springs

    std::map< IndexPair, unsigned > edgeMap;
    topology::MeshTopology* topology = dynamic_cast<topology::MeshTopology*>( this->getContext()->getTopology() );
    assert( topology );

    const topology::MeshTopology::SeqTriangles& triangles = topology->getTriangles();
    //cout<<"==================================TriangleBendingSprings<DataTypes>::init(), triangles size = "<<triangles.size()<<endl;
    for( unsigned i= 0; i<triangles.size(); ++i )
    {
        const topology::MeshTopology::Triangle& face = triangles[i];
        {
            registerTriangle( face[0], face[1], face[2], edgeMap );
        }

    }

    const topology::MeshTopology::SeqQuads& quads = topology->getQuads();
    //cout<<"==================================TriangleBendingSprings<DataTypes>::init(), quad size = "<<topology->getQuads().size()<<endl;
    for( unsigned i= 0; i<quads.size(); ++i )
    {
        const topology::MeshTopology::Quad& face = quads[i];
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
