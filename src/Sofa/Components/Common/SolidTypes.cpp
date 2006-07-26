//
// C++ Implementation: SolidTypes
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "SolidTypes.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Components
{

namespace Common
{

template<class R>
SolidTypes<R>::SpatialVector::SpatialVector()
{}

template<class R>
SolidTypes<R>::SpatialVector::SpatialVector( const Vec& l, const Vec& f ):lineVec(l),freeVec(f)
{}

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
    return lineVec * v.freeVec + freeVec * v.lineVec;
}

/// Spatial cross product
template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::SpatialVector::cross( const SpatialVector& v ) const
{
    return SpatialVector(
            Common::cross(lineVec,v.lineVec),
            Common::cross(freeVec,v.lineVec) + Common::cross(lineVec,v.freeVec)
            );
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

// template<class R>
// SolidTypes<R>::Transform::Transform( const Vec& o, const Rot& q ):orientation_(q),origin_(o)
// {}

/// Define given the origin of the child wrt the parent and the orientation of the child wrt the parent (i.e. standard way)
template<class R>
void SolidTypes<R>::Transform::setTranslationRotation( const Vec& t, const Rot& q )
{
    orientation_ =q, origin_ = -(q.inverseRotate(t));
}

/// Define given the origin of the child wrt the parent and the orientation of the child wrt the parent (i.e. standard way)
template<class R>
typename SolidTypes<R>::Transform  SolidTypes<R>::Transform::set( const Vec& t, const Rot& q )
{
    Transform f;
    f.setTranslationRotation( t, q );
    return f;
}

template<class R>
typename SolidTypes<R>::Transform SolidTypes<R>::Transform::identity()
{
    return Transform( Rot::identity(), Vec(0,0,0) );
}

/// Define as a given SpatialVector integrated during one second
template<class R>
SolidTypes<R>::Transform::Transform( const SpatialVector& v )
{
    origin_ = v.freeVec;
    orientation_ = Rot::createFromRotationVector( v.lineVec );
    //cerr<<"SolidTypes<R>::Transform::Transform( const SpatialVector& v ), v = "<<v<<", this = "<<*this<<endl;
}

template<class R>
const typename SolidTypes<R>::Vec& SolidTypes<R>::Transform::getOriginInChild() const
{
    return origin_;
}

template<class R>
typename SolidTypes<R>::Vec SolidTypes<R>::Transform::getOriginInParent() const
{
    return -orientation_.rotate(origin_);
}

template<class R>
void SolidTypes<R>::Transform::setOriginInParent( const Vec& op )
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
typename SolidTypes<R>::Mat SolidTypes<R>::Transform::getRotationMatrix() const
{
    Mat m;
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
void SolidTypes<R>::Transform::clear()
{
    orientation_.clear();
    origin_=Vec(0,0,0);
}

template<class R>
typename SolidTypes<R>::Transform SolidTypes<R>::Transform::operator * (const Transform& f2) const
{
    //cerr<<"SolidTypes<R>::Transform::operator *, orientation = "<<orientation_<<", f2.orientation = "<<f2.getOrientation()<<", product = "<<orientation_ * f2.getOrientation()<<endl;
    return Transform(  orientation_ * f2.getOrientation(), f2.getOriginInChild() + f2.getOrientation().inverseRotate(origin_)) ;
}

template<class R>
typename SolidTypes<R>::Transform& SolidTypes<R>::Transform::operator *= (const Transform& f2)
{
    orientation_ *= f2.getOrientation();
    origin_ = f2.getOriginInChild() + f2.getOrientation().inverseRotate(origin_);
    return (*this);
}

template<class R>
typename SolidTypes<R>::SpatialVector SolidTypes<R>::Transform::operator * (const SpatialVector& sv ) const
{
    return SpatialVector(
            orientation_.rotate(sv.lineVec),
            orientation_.rotate(sv.freeVec - cross( origin_, sv.lineVec) )
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
void SolidTypes<R>::Transform::writeOpenGlMatrix( Real *m ) const
{
    /*    cerr<<"SolidTypes<R>::Transform::writeOpenGlMatrix, this = "<<*this<<endl;
        cerr<<"SolidTypes<R>::Transform::writeOpenGlMatrix, origin_ = "<<origin_<<endl;*/
    orientation_.writeOpenGlMatrix(m);
    Vec t = -projectVector(origin_);
    /*	cerr<<"SolidTypes<R>::Transform::writeOpenGlMatrix, t = "<<t<<endl;*/
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
}

template<class R>
void SolidTypes<R>::Transform::glDraw() const
{
    glPushAttrib( GL_COLOR_BUFFER_BIT );
    glColor3f(1,0,0);
    glVertex3f( 0,0,0 );
    glVertex3f( 1,0,0 );
    glColor3f(0,1,0);
    glVertex3f( 0,0,0 );
    glVertex3f( 0,1,0 );
    glColor3f(0,0,1);
    glVertex3f( 0,0,0 );
    glVertex3f( 0,0,1 );
    glPopAttrib();
}

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

template<class R>
typename SolidTypes<R>::Transform& SolidTypes<R>::Transform::operator +=(const Transform& a)
{
    std::cout << "SolidTypes<R>::Transform::operator +="<<std::endl;
    origin_ += a.getOriginInChild();
    //orientation += a.getOrientation();
    //orientation.normalize();
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
SolidTypes<R>::RigidInertia::RigidInertia( Real m, const Vec& h, const Mat& I ):m(m),h(h),I(I)
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
    Vec h_mr = h - t.getOriginInChild() * m;
    Mat E = t.getRotationMatrix();
    return RigidInertia(
            m, E*h_mr,
            E*(I+crossM(t.getOriginInChild())*crossM(h)+crossM(h_mr)*crossM(t.getOriginInChild()))*(E.transposed()) );
}


//===================================================================================

template<class R>
SolidTypes<R>::ArticulatedInertia::ArticulatedInertia()
{}

template<class R>
SolidTypes<R>::ArticulatedInertia::ArticulatedInertia( const Mat& M, const Mat& H, const Mat& I ):M(M),H(H),I(I)
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





//===================================================================================


template<class R>
typename SolidTypes<R>::Vec SolidTypes<R>::mult( const typename SolidTypes<R>::Mat& m, const typename SolidTypes<R>::Vec& v )
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
typename SolidTypes<R>::Vec SolidTypes<R>::multTrans( const typename SolidTypes<R>::Mat& m, const typename SolidTypes<R>::Vec& v )
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
typename SolidTypes<R>::Mat SolidTypes<R>::crossM( const typename SolidTypes<R>::Vec& v )
{
    typename SolidTypes<R>::Mat m;
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
typename SolidTypes<R>::Mat SolidTypes<R>::dyad( const Vec& u, const Vec& v )
{
    Mat m;
    for( int i=0; i<3; i++ )
        for( int j=0; j<3; j++ )
            m[i][j] = u[i]*v[j];
    return m;
}


}//Common

}//Components

}//Sofa

