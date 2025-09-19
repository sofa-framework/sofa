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

#ifndef SOFA_DEFAULTTYPE_SOLIDTYPES_INL
#define SOFA_DEFAULTTYPE_SOLIDTYPES_INL

#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/helper/logging/Messaging.h>
#include <iostream>


namespace sofa::defaulttype
{

//=================================================================================
template<class R>
SolidTypes<R>::RigidInertia::RigidInertia()
{}

template<class R>
SolidTypes<R>::RigidInertia::RigidInertia( Real m, const Vec& h, const Mat3x3& I ):m(m),h(h),I(I)
{}

template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::RigidInertia::operator * (const SpatialVector& v ) const
{
    return SpatialVector(
            cross(v.lineVec,h)+v.freeVec*m,
            mult(I,v.lineVec) + cross( h, v.freeVec )
            );
}

template<class R>
typename SolidTypes<R>::RigidInertia SolidTypes<R>::RigidInertia::operator * ( const Transform& t ) const
{
    Vec h_mr = h - t.getOriginOfParentInChild() * m;
    Mat3x3 E = t.getRotationMatrix();
    return RigidInertia(
            m, E*h_mr,
            E*(I+crossM(t.getOriginOfParentInChild())*crossM(h)+crossM(h_mr)*crossM(t.getOriginOfParentInChild()))*(E.transposed()) );
}


//===================================================================================

template<class R>
SolidTypes<R>::ArticulatedInertia::ArticulatedInertia()
{}

template<class R>
SolidTypes<R>::ArticulatedInertia::ArticulatedInertia( const Mat3x3& M, const Mat3x3& H, const Mat3x3& I ):M(M),H(H),I(I)
{}

template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::ArticulatedInertia::operator * (const SpatialVector& v ) const
{
    return SpatialVector(
            multTrans(H,v.lineVec) + mult(M,v.freeVec),
            mult(I,v.lineVec) + mult(H,v.freeVec)
            );

}
template<class R>
typename SolidTypes<R>::ArticulatedInertia SolidTypes<R>::ArticulatedInertia::operator * ( Real r ) const
{
    return ArticulatedInertia( M*r, H*r, I*r );

}
template<class R>
typename SolidTypes<R>::ArticulatedInertia& SolidTypes<R>::ArticulatedInertia::operator = (const RigidInertia& Ri )
{
    //                         H(0,0)=0;
    //                         H(0,1)=-Ri.h[2];
    //                         H(0,2)= Ri.h[1];
    //                         H(1,0)= Ri.h[2];
    //                         H(1,1)=0;
    //                         H(1,2)=-Ri.h[0];
    //                         H(2,0)=-Ri.h[1];
    //                         H(2,1)= Ri.h[0];
    //                         H(2,2)=0;
    H = crossM( Ri.h );

    for( int i=0; i<3; i++ )
        for( int j=0; j<3; j++ )
            M(i,j)= i==j ? Ri.m : 0;

    I=Ri.I;
    return *this;
}

template<class R>
typename SolidTypes<R>::ArticulatedInertia& SolidTypes<R>::ArticulatedInertia::operator += (const ArticulatedInertia& Ai )
{
    H += Ai.H;
    M += Ai.M;
    I += Ai.I;
    return *this;
}

template<class R>
typename SolidTypes<R>::ArticulatedInertia SolidTypes<R>::ArticulatedInertia::operator + (const ArticulatedInertia& Ai ) const
{
    return ArticulatedInertia(M+Ai.M, H+Ai.H, I+Ai.I);
}

template<class R>
typename SolidTypes<R>::ArticulatedInertia SolidTypes<R>::ArticulatedInertia::operator - (const ArticulatedInertia& Ai ) const
{
    return ArticulatedInertia(M-Ai.M, H-Ai.H, I-Ai.I);
}

template<class R>
void SolidTypes<R>::ArticulatedInertia::copyTo( Mat66& m ) const
{
    for( int i=0; i<3; i++ )
    {
        for( int j=0; j<3; j++ )
        {
            m(i,j) = H(j,i);
            m(i,j+3) = M(i,j);
            m(i+3,j) = I(i,j);
            m(i+3,j+3) = H(i,j);
        }
    }
}


//===================================================================================


template<class R>
typename SolidTypes<R>::Vec SolidTypes<R>::mult( const typename SolidTypes<R>::Mat3x3& m, const typename SolidTypes<R>::Vec& v )
{
    typename SolidTypes<R>::Vec r;
    for( int i=0; i<3; ++i )
    {
        r[i]=0;
        for( int j=0; j<3; ++j )
            r[i]+=m(i,j) * v[j];
    }
    return r;
}

template<class R>
typename SolidTypes<R>::Vec SolidTypes<R>::multTrans( const typename SolidTypes<R>::Mat3x3& m, const typename SolidTypes<R>::Vec& v )
{
    typename SolidTypes<R>::Vec r;
    for( int i=0; i<3; ++i )
    {
        r[i]=0;
        for( int j=0; j<3; ++j )
            r[i]+=m(j,i) * v[j];
    }
    return r;
}

/// Cross product matrix of a vector
template<class R>
typename SolidTypes<R>::Mat3x3 SolidTypes<R>::crossM( const typename SolidTypes<R>::Vec& v )
{
    typename SolidTypes<R>::Mat3x3 m;
    m(0,0)=0;
    m(0,1)=-v[2];
    m(0,2)= v[1];
    m(1,0)= v[2];
    m(1,1)=0;
    m(1,2)=-v[0];
    m(2,0)=-v[1];
    m(2,1)= v[0];
    m(2,2)=0;
    return m;
}


template<class R>
typename SolidTypes<R>::ArticulatedInertia  SolidTypes<R>::dyad( const SpatialVector& u, const SpatialVector& v )
{
    return ArticulatedInertia(dyad(u.lineVec, v.lineVec), dyad(u.freeVec, v.lineVec),  dyad(u.freeVec, v.freeVec));
}

template<class R>
typename SolidTypes<R>::Mat3x3 SolidTypes<R>::dyad( const Vec& u, const Vec& v )
{
    Mat3x3 m;
    for( int i=0; i<3; i++ )
        for( int j=0; j<3; j++ )
            m(i,j) = u[i]*v[j];
    return m;
}

}

#endif
