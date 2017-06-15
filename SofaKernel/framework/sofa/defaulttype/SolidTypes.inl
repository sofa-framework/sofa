/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <iostream>

namespace sofa
{

namespace defaulttype
{

template<class R>
SolidTypes<R>::SpatialVector::SpatialVector()
{}



template<class R>
SolidTypes<R>::SpatialVector::SpatialVector( const Vec& l, const Vec& f ):lineVec(l),freeVec(f)
{}

/*
template<class R>
SolidTypes<R>::SpatialVector::SpatialVector(const SolidTypes<R>::Transform &DTrans)
{
    freeVec = DTrans.getOrigin();
    lineVec = DTrans.getOrientation().toEulerVector(); // Consider to use quatToRotationVector instead of toEulerVector to have the rotation vector
}
*/

template<class R>
void SolidTypes<R>::SpatialVector::clear()
{
    lineVec = freeVec = Vec(0,0,0);
}

template<class R>
typename SolidTypes<R>::SpatialVector& SolidTypes<R>::SpatialVector::operator += (const SpatialVector& v)
{
    lineVec += v.lineVec;
    freeVec += v.freeVec;
    return *this;
}
/*
template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::SpatialVector::operator * ( Real a ) const
{
        return SpatialVector( lineVec *a, freeVec * a);
}

template<class R>
typename SolidTypes<R>::SpatialVector& SolidTypes<R>::SpatialVector::operator *= ( Real a )
{
   lineVec *=a; freeVec *= a;
        return *this;
}
*/
template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::SpatialVector::operator + ( const SpatialVector& v ) const
{
    return SpatialVector(lineVec+v.lineVec,freeVec+v.freeVec);
}

template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::SpatialVector::operator - ( const SpatialVector& v ) const
{
    return SpatialVector(lineVec-v.lineVec,freeVec-v.freeVec);
}

template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::SpatialVector::operator - ( ) const
{
    return SpatialVector(-lineVec,-freeVec);
}

/// Spatial dot product (cross terms)
template<class R>
typename SolidTypes<R>::Real SolidTypes<R>::SpatialVector::operator * ( const SpatialVector& v ) const
{
    //msg_info()<<" SolidTypes<R>::SpatialVector: "<<*this<<" * "<<v<<" = "<< lineVec * v.freeVec + freeVec * v.lineVec<<std::endl;
    return lineVec * v.freeVec + freeVec * v.lineVec;
}

/// Spatial cross product
template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::SpatialVector::cross( const SpatialVector& v ) const
{
    return SpatialVector(
            defaulttype::cross(lineVec,v.lineVec),
            defaulttype::cross(freeVec,v.lineVec) + defaulttype::cross(lineVec,v.freeVec)
            );
}

template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::SpatialVector::operator * (const Mat66& m) const
{
    SpatialVector result;
    for( int i=0; i<3; i++ )
    {
        result.lineVec[i]=0;
        result.freeVec[i]=0;
        for( int j=0; j<3; j++ )
        {
            result.lineVec[i] += lineVec[j]*m[i][j] + freeVec[j]*m[i][j+3];
            result.freeVec[i] += lineVec[j]*m[i+3][j] + freeVec[j]*m[i+3][j+3];
        }
    }
    return result;
}
//======================================================================================================

template<class R>
SolidTypes<R>::Transform::Transform()
{
    *this = this->identity();
}

/// Define using Featherstone's conventions
template<class R>
SolidTypes<R>::Transform::Transform( const Rot& q, const Vec& o ):orientation_(q),origin_(o)
{}

/// Define using standard conventions
template<class R>
SolidTypes<R>::Transform::Transform( const Vec& t, const Rot& q )
    :orientation_(q),origin_(-(q.inverseRotate(t)))
{}

/// Define given the origin of the child wrt the parent and the orientation of the child wrt the parent (i.e. standard way)
template<class R>
void SolidTypes<R>::Transform::set
( const Vec& t, const Rot& q )
{
    orientation_ =q, origin_ = -(q.inverseRotate(t));
}

/// Define given the origin of the child wrt the parent and the orientation of the child wrt the parent (i.e. standard way)
// template<class R>
//         typename SolidTypes<R>::Transform  SolidTypes<R>::Transform::inParent( const Vec& t, const Rot& q )
// {
//     Transform f;
//     f.setInParent( t, q );
//     return f;
// }

template<class R>
typename SolidTypes<R>::Transform SolidTypes<R>::Transform::identity()
{
    return Transform( Rot::identity(), Vec(0,0,0) );
}

/// Define as a given SpatialVector integrated during one second. The spatial vector is given in parent coordinates.
template<class R>
SolidTypes<R>::Transform::Transform( const SpatialVector& v )
{
    //origin_ = v.freeVec;
    orientation_ = Rot::createFromRotationVector( v.lineVec );
    origin_ = - orientation_.inverseRotate( v.freeVec );
    //msg_info()<<"SolidTypes<R>::Transform::Transform( const SpatialVector& v ), v = "<<v<<", this = "<<*this<<std::endl;
}

template<class R>
const typename SolidTypes<R>::Vec& SolidTypes<R>::Transform::getOriginOfParentInChild() const
{
    return origin_;
}

template<class R>
typename SolidTypes<R>::Vec SolidTypes<R>::Transform::getOrigin() const
{
    return -orientation_.rotate(origin_);
}

template<class R>
void SolidTypes<R>::Transform::setOrigin( const Vec& op )
{
    origin_ = -orientation_.inverseRotate(op);
}

template<class R>
const typename SolidTypes<R>::Rot& SolidTypes<R>::Transform::getOrientation() const
{
    return orientation_;
}

template<class R>
void SolidTypes<R>::Transform::setOrientation( const Rot& q )
{
    orientation_=q;
}
template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::Transform::DTrans()
{

    return SpatialVector(orientation_.quatToRotationVector(), this->getOrigin());
    // Use of quatToRotationVector instead of toEulerVector:
    // this is done to keep the old behavior (before the
    // correction of the toEulerVector  function). If the
    // purpose was to obtain the Eulerian vector and not the
    // rotation vector please use the following line instead
    //return SpatialVector(orientation_.toEulerVector(), this->getOrigin());
}

template<class R>
typename SolidTypes<R>::Vec SolidTypes<R>::Transform::projectVector( const Vec& v ) const
{
    return orientation_.rotate( v );
}

template<class R>
typename SolidTypes<R>::Vec SolidTypes<R>::Transform::projectPoint( const Vec& p ) const
{
    return orientation_.rotate( p - origin_ );
}

template<class R>
typename SolidTypes<R>::Vec SolidTypes<R>::Transform::backProjectVector( const Vec& v ) const
{
    return orientation_.inverseRotate( v );
}

template<class R>
typename SolidTypes<R>::Vec SolidTypes<R>::Transform::backProjectPoint( const Vec& p ) const
{
    return orientation_.inverseRotate( p ) + origin_;
}

template<class R>
typename SolidTypes<R>::Mat3x3 SolidTypes<R>::Transform::getRotationMatrix() const
{
    Mat3x3 m;
    m[0][0] = (1.0f - 2.0f * (orientation_[1] * orientation_[1] + orientation_[2] * orientation_[2]));
    m[0][1] = (2.0f * (orientation_[0] * orientation_[1] - orientation_[2] * orientation_[3]));
    m[0][2] = (2.0f * (orientation_[2] * orientation_[0] + orientation_[1] * orientation_[3]));

    m[1][0] = (2.0f * (orientation_[0] * orientation_[1] + orientation_[2] * orientation_[3]));
    m[1][1] = (1.0f - 2.0f * (orientation_[2] * orientation_[2] + orientation_[0] * orientation_[0]));
    m[1][2] = (2.0f * (orientation_[1] * orientation_[2] - orientation_[0] * orientation_[3]));

    m[2][0] = (2.0f * (orientation_[2] * orientation_[0] - orientation_[1] * orientation_[3]));
    m[2][1] = (2.0f * (orientation_[1] * orientation_[2] + orientation_[0] * orientation_[3]));
    m[2][2] = (1.0f - 2.0f * (orientation_[1] * orientation_[1] + orientation_[0] * orientation_[0]));
    return m;
}

template<class R>
typename SolidTypes<R>::Mat6x6 SolidTypes<R>::Transform::getAdjointMatrix() const
{
    /// TODO
    Mat6x6 Adj;
    Mat3x3 Rot;
    Rot = this->getRotationMatrix();
    // correspond au produit vectoriel v^origin
    Mat3x3 Origin;
    Origin[0][0]=(Real)0.0;         Origin[0][1]=origin_[2];    Origin[0][2]=-origin_[1];
    Origin[1][0]=-origin_[2];       Origin[1][1]=(Real)0.0;     Origin[1][2]=origin_[0];
    Origin[2][0]=origin_[1];        Origin[2][1]=-origin_[0];   Origin[2][2]=(Real)0.0;

    Mat3x3 R_Origin = Rot*Origin;

    for (int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            Adj[i][j]     = Rot[i][j];
            Adj[i+3][j+3] = Rot[i][j];
            Adj[i][j+3] =   R_Origin[i][j];
            Adj[i+3][j] = 0.0;
        }
    }


    return Adj;
}

template<class R>
void SolidTypes<R>::Transform::clear()
{
    orientation_.clear();
    origin_=Vec(0,0,0);
}

template<class R>
typename SolidTypes<R>::Transform SolidTypes<R>::Transform::operator * (const Transform& f2) const
{
    //msg_info()<<"SolidTypes<R>::Transform::operator *, orientation = "<<orientation_<<", f2.orientation = "<<f2.getOrientation()<<", product = "<<orientation_ * f2.getOrientation()<<std::endl;
    return Transform(  orientation_ * f2.getOrientation(), f2.getOriginOfParentInChild() + f2.getOrientation().inverseRotate(origin_)) ;
}

template<class R>
typename SolidTypes<R>::Transform& SolidTypes<R>::Transform::operator *= (const Transform& f2)
{
    orientation_ *= f2.getOrientation();
    origin_ = f2.getOriginOfParentInChild() + f2.getOrientation().inverseRotate(origin_);
    return (*this);
}


template<class R>
typename SolidTypes<R>::SpatialVector  SolidTypes<R>::Transform::CreateSpatialVector()
{
    return SpatialVector(this->getOrientation().quatToRotationVector(),  this->getOrigin() );
}

template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::Transform::operator * (const SpatialVector& sv ) const
{
    /*
       return SpatialVector(
                  orientation_.rotate(sv.lineVec),
                  orientation_.rotate(sv.freeVec - cross( origin_, sv.lineVec) )
              );*/

    //std::cout<<"sv.lineVec"<<sv.lineVec<<" orientation_.rotate(sv.lineVec)"<<orientation_.rotate(sv.lineVec)<<std::endl;

    return SpatialVector(
            orientation_.rotate(sv.lineVec),
            orientation_.rotate( cross(sv.lineVec, origin_ ) + sv.freeVec)
            );
}

template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::Transform::operator / (const SpatialVector& sv ) const
{
    return inversed()*sv;
}




template<class R>
typename SolidTypes<R>::Transform SolidTypes<R>::Transform::inversed() const
{
    //return Transform( orientation_.inverse(), -(orientation_.inverse().rotate(origin_)) );
    return Transform( orientation_.inverse(), -(orientation_.rotate(origin_)) );
}

template<class R>
void SolidTypes<R>::Transform::writeOpenGlMatrix( GLdouble *m ) const
{
    /*    msg_info()<<"SolidTypes<R>::Transform::writeOpenGlMatrix, this = "<<*this<<std::endl;
        msg_info()<<"SolidTypes<R>::Transform::writeOpenGlMatrix, origin_ = "<<origin_<<std::endl;*/
    orientation_.writeOpenGlMatrix(m);
    Vec t = getOrigin();
    /*	msg_info()<<"SolidTypes<R>::Transform::writeOpenGlMatrix, t = "<<t<<std::endl;*/
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
}

//This type should not have its own draw function
//template<class R>
//void SolidTypes<R>::Transform::glDraw() const
//{
//#ifndef SOFA_NO_OPENGL
//    glPushAttrib( GL_COLOR_BUFFER_BIT );
//    glColor3f(1,0,0);
//    glVertex3f( 0,0,0 );
//    glVertex3f( 1,0,0 );
//    glColor3f(0,1,0);
//    glVertex3f( 0,0,0 );
//    glVertex3f( 0,1,0 );
//    glColor3f(0,0,1);
//    glVertex3f( 0,0,0 );
//    glVertex3f( 0,0,1 );
//    glPopAttrib();
//#endif /* SOFA_NO_OPENGL */
//}

template<class R>
void SolidTypes<R>::Transform::printInternal( std::ostream& out ) const
{
    out<<"internal t= "<<origin_<<std::endl;
    out<<"E= "<<orientation_<<std::endl;
}


template<class R>
typename SolidTypes<R>::Transform& SolidTypes<R>::Transform::operator += (const SpatialVector& v)
{
    *this *= Transform(v);
    return *this;
}

// template<class R>
// typename SolidTypes<R>::Transform& SolidTypes<R>::Transform::operator += (const SpatialVector& v)
// {
//     SpatialVector vlocal = (*this)/v;
//     Transform tv(vlocal);
//     *this = Transform(
//                  tv.getOrientation()*getOrientation(),
//     tv.getOrientation().inverseRotate( getOriginOfParentInChild()-getOrientation().rotate(tv.getOrigin())
//                                  )
//              );
//     msg_info()<<"SolidTypes<R>::Transform::operator += SpatialVector, new value = "<<*this<<std::endl;
//     return *this;
// }

template<class R>
typename SolidTypes<R>::Transform& SolidTypes<R>::Transform::operator +=(const Transform& a)
{
    dmsg_warning("SolidTypes::operator+") << "+";
    origin_ += a.getOriginOfParentInChild();

    // previously commented out:
    orientation_ += a.getOrientation();
    orientation_.normalize();

    return *this;
}

/*
template<class R>
      typename SolidTypes<R>::Transform& SolidTypes<R>::Transform::operator*=(Real a)
{
   std::cout << "SolidTypes<R>::Transform::operator *="<<std::endl;
   origin_ *= a;
        //orientation *= a;
   return *this;
}

template<class R>
      typename SolidTypes<R>::Transform SolidTypes<R>::Transform::operator*(Real a) const
{
   Transform r = *this;
   r*=a;
   return r;
}
*/

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
    //                         H[0][0]=0;
    //                         H[0][1]=-Ri.h[2];
    //                         H[0][2]= Ri.h[1];
    //                         H[1][0]= Ri.h[2];
    //                         H[1][1]=0;
    //                         H[1][2]=-Ri.h[0];
    //                         H[2][0]=-Ri.h[1];
    //                         H[2][1]= Ri.h[0];
    //                         H[2][2]=0;
    H = crossM( Ri.h );

    for( int i=0; i<3; i++ )
        for( int j=0; j<3; j++ )
            M[i][j]= i==j ? Ri.m : 0;

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
            m[i][j] = H[j][i];
            m[i][j+3] = M[i][j];
            m[i+3][j] = I[i][j];
            m[i+3][j+3] = H[i][j];
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
            r[i]+=m[i][j] * v[j];
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
            r[i]+=m[j][i] * v[j];
    }
    return r;
}

/// Cross product matrix of a vector
template<class R>
typename SolidTypes<R>::Mat3x3 SolidTypes<R>::crossM( const typename SolidTypes<R>::Vec& v )
{
    typename SolidTypes<R>::Mat3x3 m;
    m[0][0]=0;
    m[0][1]=-v[2];
    m[0][2]= v[1];
    m[1][0]= v[2];
    m[1][1]=0;
    m[1][2]=-v[0];
    m[2][0]=-v[1];
    m[2][1]= v[0];
    m[2][2]=0;
    return m;
}


template<class R>
typename SolidTypes<R>::ArticulatedInertia  SolidTypes<R>::dyad( const SpatialVector& u, const SpatialVector& v )
{
    //return ArticulatedInertia(dyad(u.lineVec, v.freeVec), dyad(u.freeVec, v.freeVec),  dyad(u.freeVec, v.lineVec));
    return ArticulatedInertia(dyad(u.lineVec, v.lineVec), dyad(u.freeVec, v.lineVec),  dyad(u.freeVec, v.freeVec));
}

template<class R>
typename SolidTypes<R>::Mat3x3 SolidTypes<R>::dyad( const Vec& u, const Vec& v )
{
    Mat3x3 m;
    for( int i=0; i<3; i++ )
        for( int j=0; j<3; j++ )
            m[i][j] = u[i]*v[j];
    return m;
}


}// defaulttype

}// sofa

#endif
