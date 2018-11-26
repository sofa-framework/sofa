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
#define SOFA_COMPONENT_MASS_MESHMATRIXMASS_CPP
#include <SofaMiscForceField/MeshMatrixMass.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;


#ifndef SOFA_FLOAT

template <>
Vector6 MeshMatrixMass<Vec3dTypes, double>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    const MassVector &vertexMass= d_vertexMassInfo.getValue();
    const MassVector &edgeMass= d_edgeMassInfo.getValue();

    helper::ReadAccessor< DataVecCoord > x = vx;
    helper::ReadAccessor< DataVecDeriv > v = vv;

    Vector6 momentum;
    for( unsigned int i=0 ; i<v.size() ; i++ )
    {
        Deriv linearMomentum = v[i] * vertexMass[i];
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];
        Deriv angularMomentum = cross( x[i], linearMomentum );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    for( int i=0 ; i<_topology->getNbEdges() ; ++i )
    {
        unsigned v0 = _topology->getEdge(i)[0];
        unsigned v1 = _topology->getEdge(i)[1];

        // is it correct to share the edge mass between the 2 vertices?
        double m = edgeMass[i] * 0.5;

        Deriv linearMomentum = v[v0] * m;
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];
        Deriv angularMomentum = cross( x[v0], linearMomentum );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];

        linearMomentum = v[v1] * m;
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];
        angularMomentum = cross( x[v1], linearMomentum );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}

#endif
#ifndef SOFA_DOUBLE

template <>
Vector6 MeshMatrixMass<Vec3fTypes, float>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    const MassVector &vertexMass= d_vertexMassInfo.getValue();
    const MassVector &edgeMass= d_edgeMassInfo.getValue();

    helper::ReadAccessor< DataVecCoord > x = vx;
    helper::ReadAccessor< DataVecDeriv > v = vv;

    Vector6 momentum;
    for( unsigned int i=0 ; i<v.size() ; i++ )
    {
        Deriv linearMomentum = v[i] * vertexMass[i];
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];
        Deriv angularMomentum = cross( x[i], linearMomentum );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    for( int i=0 ; i<_topology->getNbEdges() ; ++i )
    {
        unsigned v0 = _topology->getEdge(i)[0];
        unsigned v1 = _topology->getEdge(i)[1];

        // is it correct to share the edge mass between the 2 vertices?
        float m = edgeMass[i] * 0.5f;

        Deriv linearMomentum = v[v0] * m;
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];
        Deriv angularMomentum = cross( x[v0], linearMomentum );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];

        linearMomentum = v[v1] * m;
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];
        angularMomentum = cross( x[v1], linearMomentum );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}


#endif







// Register in the Factory
int MeshMatrixMassClass = core::RegisterObject("Define a specific mass for each particle")
#ifndef SOFA_FLOAT
        .add< MeshMatrixMass<Vec3dTypes,Vec3dTypes::Real> >()
        .add< MeshMatrixMass<Vec2dTypes,Vec2dTypes::Real> >()
        .add< MeshMatrixMass<Vec1dTypes,Vec1dTypes::Real> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MeshMatrixMass<Vec3fTypes,Vec3fTypes::Real> >()
        .add< MeshMatrixMass<Vec2fTypes,Vec2fTypes::Real> >()
        .add< MeshMatrixMass<Vec1fTypes,Vec1fTypes::Real> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec3dTypes,Vec3dTypes::Real>;
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec2dTypes,Vec2dTypes::Real>;
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec1dTypes,Vec1dTypes::Real>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec3fTypes,Vec3fTypes::Real>;
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec2fTypes,Vec2fTypes::Real>;
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec1fTypes,Vec1fTypes::Real>;
#endif


} // namespace mass

} // namespace component

} // namespace sofa

