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
#include <sofa/type/Transform.h>

namespace sofa::type
{

template<class TReal>
Transform<TReal>::Transform()
    : orientation_() // default constructor set to identity
    , origin_() // default constructor set to {0, 0, 0}
{

}

/// Define using Featherstone's conventions
template<class TReal>
Transform<TReal>::Transform( const Rot& q, const Vec& o )
    : orientation_(q)
    , origin_(o)
{}

/// Define using standard conventions
template<class TReal>
Transform<TReal>::Transform( const Vec& t, const Rot& q )
    : orientation_(q)
    , origin_(-(q.inverseRotate(t)))
{}

/// Define given the origin of the child wrt the parent and the orientation of the child wrt the parent (i.e. standard way)
template<class TReal>
void Transform<TReal>::set
( const Vec& t, const Rot& q )
{
    orientation_ =q, origin_ = -(q.inverseRotate(t));
}

/// Define given the origin of the child wrt the parent and the orientation of the child wrt the parent (i.e. standard way)
template<class TReal>
Transform<TReal> Transform<TReal>::identity()
{
    return Transform( Rot::identity(), Vec(0,0,0) );
}

/// Define as a given SpatialVector integrated during one second. The spatial vector is given in parent coordinates.
template<class TReal>
Transform<TReal>::Transform( const SpatialVector<TReal>& v )
{
    orientation_ = Rot::createFromRotationVector( v.lineVec );
    origin_ = - orientation_.inverseRotate( v.freeVec );
}

template<class TReal>
const typename Transform<TReal>::Vec& Transform<TReal>::getOriginOfParentInChild() const
{
    return origin_;
}

template<class TReal>
typename Transform<TReal>::Vec Transform<TReal>::getOrigin() const
{
    return -orientation_.rotate(origin_);
}

template<class TReal>
void Transform<TReal>::setOrigin( const Vec& op )
{
    origin_ = -orientation_.inverseRotate(op);
}

template<class TReal>
const typename Transform<TReal>::Rot& Transform<TReal>::getOrientation() const
{
    return orientation_;
}

template<class TReal>
void Transform<TReal>::setOrientation( const Rot& q )
{
    orientation_=q;
}
template<class TReal>
SpatialVector<TReal> Transform<TReal>::DTrans()
{

    return SpatialVector(orientation_.quatToRotationVector(), this->getOrigin());
    // Use of quatToRotationVector instead of toEulerVector:
    // this is done to keep the old behavior (before the
    // correction of the toEulerVector  function). If the
    // purpose was to obtain the Eulerian vector and not the
    // rotation vector please use the following line instead
    //return SpatialVector(orientation_.toEulerVector(), this->getOrigin());
}

template<class TReal>
typename Transform<TReal>::Vec Transform<TReal>::projectVector( const Vec& v ) const
{
    return orientation_.rotate( v );
}

template<class TReal>
auto Transform<TReal>::projectPoint(const Vec& p) const -> Transform<TReal>::Vec
{
    return orientation_.rotate( p - origin_ );
}

template<class TReal>
auto Transform<TReal>::backProjectVector(const Vec& v) const -> Transform<TReal>::Vec
{
    return orientation_.inverseRotate( v );
}

template<class TReal>
auto Transform<TReal>::backProjectPoint(const Vec& p) const -> Transform<TReal>::Vec
{
    return orientation_.inverseRotate( p ) + origin_;
}

template<class TReal>
auto Transform<TReal>::getRotationMatrix() const -> Transform<TReal>::Mat3x3
{
    Mat3x3 m;
    m(0,0) = (1.0f - 2.0f * (orientation_[1] * orientation_[1] + orientation_[2] * orientation_[2]));
    m(0,1) = (2.0f * (orientation_[0] * orientation_[1] - orientation_[2] * orientation_[3]));
    m(0,2) = (2.0f * (orientation_[2] * orientation_[0] + orientation_[1] * orientation_[3]));

    m(1,0) = (2.0f * (orientation_[0] * orientation_[1] + orientation_[2] * orientation_[3]));
    m(1,1) = (1.0f - 2.0f * (orientation_[2] * orientation_[2] + orientation_[0] * orientation_[0]));
    m(1,2) = (2.0f * (orientation_[1] * orientation_[2] - orientation_[0] * orientation_[3]));

    m(2,0) = (2.0f * (orientation_[2] * orientation_[0] - orientation_[1] * orientation_[3]));
    m(2,1) = (2.0f * (orientation_[1] * orientation_[2] + orientation_[0] * orientation_[3]));
    m(2,2) = (1.0f - 2.0f * (orientation_[1] * orientation_[1] + orientation_[0] * orientation_[0]));
    return m;
}

template<class TReal>
auto Transform<TReal>::getAdjointMatrix() const -> Transform<TReal>::Mat6x6
{
    /// TODO
    Mat6x6 Adj;
    Mat3x3 Rot;
    Rot = this->getRotationMatrix();
    // correspond au produit vectoriel v^origin
    Mat3x3 Origin;
    Origin(0,0)=(Real)0.0;         Origin(0,1)=origin_[2];    Origin(0,2)=-origin_[1];
    Origin(1,0)=-origin_[2];       Origin(1,1)=(Real)0.0;     Origin(1,2)=origin_[0];
    Origin(2,0)=origin_[1];        Origin(2,1)=-origin_[0];   Origin(2,2)=(Real)0.0;

    Mat3x3 R_Origin = Rot*Origin;

    for (int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            Adj(i,j)     = Rot(i,j);
            Adj(i+3,j+3) = Rot(i,j);
            Adj(i,j+3) =   R_Origin(i,j);
            Adj(i+3,j) = 0.0;
        }
    }


    return Adj;
}

template<class TReal>
void Transform<TReal>::clear()
{
    orientation_.clear();
    origin_=Vec(0,0,0);
}

template<class TReal>
Transform<TReal> Transform<TReal>::operator * (const Transform& f2) const
{
    return Transform(  orientation_ * f2.getOrientation(), f2.getOriginOfParentInChild() + f2.getOrientation().inverseRotate(origin_)) ;
}

template<class TReal>
Transform<TReal>& Transform<TReal>::operator *= (const Transform& f2)
{
    orientation_ *= f2.getOrientation();
    origin_ = f2.getOriginOfParentInChild() + f2.getOrientation().inverseRotate(origin_);
    return (*this);
}


template<class TReal>
SpatialVector<TReal> Transform<TReal>::CreateSpatialVector()
{
    return SpatialVector(this->getOrientation().quatToRotationVector(),  this->getOrigin() );
}

template<class TReal>
SpatialVector<TReal> Transform<TReal>::operator * (const SpatialVector<TReal>& sv ) const
{
    return SpatialVector(
            orientation_.rotate(sv.lineVec),
            orientation_.rotate( cross(sv.lineVec, origin_ ) + sv.freeVec)
            );
}

template<class TReal>
SpatialVector<TReal> Transform<TReal>::operator / (const SpatialVector<TReal>& sv ) const
{
    return inversed()*sv;
}




template<class TReal>
Transform<TReal> Transform<TReal>::inversed() const
{
    return Transform( orientation_.inverse(), -(orientation_.rotate(origin_)) );
}

template<class TReal>
void Transform<TReal>::writeOpenGlMatrix( double *m ) const
{
    orientation_.writeOpenGlMatrix(m);
    Vec t = getOrigin();
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
}

template<class TReal>
void Transform<TReal>::printInternal( std::ostream& out ) const
{
    out<<"internal t= "<<origin_<<std::endl;
    out<<"E= "<<orientation_<<std::endl;
}


template<class TReal>
Transform<TReal>& Transform<TReal>::operator += (const SpatialVector<TReal>& v)
{
    *this *= Transform(v);
    return *this;
}

template<class TReal>
Transform<TReal>& Transform<TReal>::operator +=(const Transform& a)
{
    origin_ += a.getOriginOfParentInChild();

    // previously commented out:
    orientation_ += a.getOrientation();
    orientation_.normalize();

    return *this;
}
}
