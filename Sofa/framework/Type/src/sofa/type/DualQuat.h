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

#include <sofa/type/config.h>

#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Quat.h>
#include <iostream>

namespace // anonymous
{
    template<class T>
    inline T rsqrt(const T& a)
    {
        return sqrt(a);
    }

} // anonymous namespace

// used to "delete" the constructor with RigidCoord
namespace sofa::defaulttype
{
    template<unsigned int, typename real2>
    class RigidCoord;
}

namespace sofa::type
{

template<class real>
class SOFA_TYPE_API DualQuatCoord3
{
    typedef real value_type;
    typedef type::Vec<3,real> Pos;
    typedef type::Vec<3,real> Vec3;
    typedef type::Vec<4,real> Quat;

    static constexpr Size total_size = 8;
    static constexpr Size spatial_dimensions = 3;

    Quat dual;
    Quat orientation;

public:

    DualQuatCoord3 (const Quat &Dual, const Quat &orient)
        : dual(Dual), orientation(orient) {}

    template<typename real2>
    DualQuatCoord3(const DualQuatCoord3<real2>& c)
        : dual(c.getDual()), orientation(c.getOrientation()) {}

    DualQuatCoord3(const Pos& p, const sofa::type::Quat<real>& q)
    {
        for (unsigned int i = 0; i < 4; i++)
            orientation[i] = q[i];
        setTranslation(p);
    }

    DualQuatCoord3 () { clear(); }
    void clear() { dual[0]=dual[1]=dual[2]=dual[3]=orientation[0]=orientation[1]=orientation[2]=orientation[3]=(real)0.; }

    ~DualQuatCoord3() {}

    static int max_size()  { return 8; }
    real* ptr() { return dual.ptr(); }
    const real* ptr() const { return dual.ptr(); }
    static unsigned int size() {return 8;}

    static DualQuatCoord3<real> identity()
    {
        DualQuatCoord3<real> c;
        c.getOrientation()[3]=(real)1.0;
        return c;
    }

    void setTranslation(const Vec3& p)
    {
        dual[0] =  (real)0.5* ( p[0]*orientation[3] + p[1]*orientation[2] - p[2]*orientation[1] );
        dual[1] =  (real)0.5* (-p[0]*orientation[2] + p[1]*orientation[3] + p[2]*orientation[0] );
        dual[2] =  (real)0.5* ( p[0]*orientation[1] - p[1]*orientation[0] + p[2]*orientation[3] );
        dual[3] = -(real)0.5* ( p[0]*orientation[0] + p[1]*orientation[1] + p[2]*orientation[2] );
    }

    Vec3 getTranslation()
    {
        Vec3 t;
        t[0] =  (real)2. * ( -dual[3]*orientation[0] + dual[0]*orientation[3] - dual[1]*orientation[2] + dual[2]*orientation[1] );
        t[1] =  (real)2. * ( -dual[3]*orientation[1] + dual[0]*orientation[2] + dual[1]*orientation[3] - dual[2]*orientation[0] );
        t[2] =  (real)2. * ( -dual[3]*orientation[2] - dual[0]*orientation[1] + dual[1]*orientation[0] + dual[2]*orientation[3] );
        return t;
    }

    Quat& getDual () { return dual; }
    Quat& getOrientation () { return orientation; }
    const Quat& getDual () const { return dual; }
    const Quat& getOrientation () const { return orientation; }




    //
    real norm2() const;
    real norm() const    { return rsqrt(norm2()); }
    void normalize();
    void invert() ;

    template<typename real2>
    void toMatrix( type::Mat<3,4,real2>& m) const;
    template<typename  real2>
    void toRotationMatrix( type::Mat<3,3,real2>& m) const
    {
        m(0,0) = (real2) (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]));
        m(0,1) = (real2) (2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3]));
        m(0,2) = (real2) (2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3]));

        m(1,0) = (real2) (2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3]));
        m(1,1) = (real2) (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]));
        m(1,2) = (real2) (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]));

        m(2,0) = (real2) (2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3]));
        m(2,1) = (real2) (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]));
        m(2,2) = (real2) (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]));
    }


    // IO
    // write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const DualQuatCoord3<real>& v )
    {
        out<<v.dual<<" "<<v.orientation;
        return out;
    }
    // read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, DualQuatCoord3<real>& v )
    {
        in>>v.dual>>v.orientation;
        return in;
    }

    // transformations
    Vec3 rotate(const Vec3& v) const;
    Vec3 inverseRotate(const Vec3& v) const;
    DualQuatCoord3<real> inverse( ) ;
    // compute the product with another frame on the right
    DualQuatCoord3<real> multRight( const DualQuatCoord3<real>& c ) const ;
    // Apply a transformation with respect to itself
    DualQuatCoord3<real> multLeft( const DualQuatCoord3<real>& c );
    // Project a point from the child frame to the parent frame: P = R(q)p + t(q)
    Vec3 pointToParent( const Vec3& p )  ;
    // Project a point from the parent frame to the child frame
    Vec3 pointToChild( const Vec3& v ) ;
    // compute the projection of a vector from the parent frame to the child
    Vec3 vectorToChild( const Vec3& v ) ;

    // operators
    template<typename real2>
    void operator =(const DualQuatCoord3<real2>& c) { dual = c.getDual(); orientation = c.getOrientation(); }

    void operator =(const Vec3& p)     { setTranslation(p); }

    void operator +=(const DualQuatCoord3<real>& a)         { dual += a.getDual(); orientation += a.getOrientation(); }

    template<typename real2>
    void operator*=(real2 a)   { orientation *= a; dual *= a; }

    template<typename real2>
    void operator/=(real2 a) { orientation /= a; dual /= a; }

    template<typename real2>
    DualQuatCoord3<real> operator*(real2 a) const { DualQuatCoord3 r = *this; r*=a; return r; }

    real operator*(const DualQuatCoord3<real>& a) const
    {
        return dual[0]*a.dual[0]+dual[1]*a.dual[1]+dual[2]*a.dual[2]+dual[3]*a.dual[3]
                +orientation[0]*a.orientation[0]+orientation[1]*a.orientation[1]
                +orientation[2]*a.orientation[2]+orientation[3]*a.orientation[3];
    }

    DualQuatCoord3<real> operator + (const type::Vec<6,real>& a)
    {
        DualQuatCoord3 r;

        r.orientation[0] = orientation[0] + (real)0.5* (a[3] * orientation[3] + a[4] * orientation[2] - a[5] * orientation[1]);
        r.orientation[1] = orientation[1] +(real)0.5* (a[4] * orientation[3] + a[5] * orientation[0] - a[3] * orientation[2]);
        r.orientation[2] = orientation[2] +(real)0.5* (a[5] * orientation[3] + a[3] * orientation[1] - a[4] * orientation[0]);
        r.orientation[3] = orientation[3] +(real)0.5* (-(a[3] * orientation[0] + a[4] * orientation[1] + a[5] * orientation[2]));

        r.setTranslation(getTranslation()+Vec3(a[0],a[1],a[2]));
        r.orientation.normalize();
        return r;
    }

    // Access to i-th element.
    real& operator[](int i)
    {
        if (i<4)
            return this->dual(i);
        else
            return this->orientation[i-4];
    }

    // Const access to i-th element.
    const real& operator[](int i) const
    {
        if (i<4)
            return this->dual(i);
        else
            return this->orientation[i-4];
    }

    // Jacobian functions
    // get velocity/quaternion change mapping : dq = J(q) v
    void velocity_getJ( type::Mat<4,3,real>& J0, type::Mat<4,3,real>& JE);
    // get quaternion change: dq = J(q) v
    DualQuatCoord3<real> velocity_applyJ( const type::Vec<6,real>& a );
    // get velocity : v = JT(q) dq
    type::Vec<6,real> velocity_applyJT( const DualQuatCoord3<real>& dq );
    // get jacobian of the normalization : dqn = J(q) dq
    void normalize_getJ( type::Mat<4,4,real>& J0, type::Mat<4,4,real>& JE) ;
    // get normalized quaternion change: dqn = J(q) dq
    DualQuatCoord3<real> normalize_applyJ( const DualQuatCoord3<real>& dq ) ;
    // get unnormalized quaternion change: dq = JT(q) dqn
    DualQuatCoord3<real> normalize_applyJT( const DualQuatCoord3<real>& dqn ) ;
    // get Jacobian change: dJ = H(p) dq
    void  normalize_getdJ( type::Mat<4,4,real>& dJ0, type::Mat<4,4,real>& dJE, const DualQuatCoord3<real>& dq ) ;
    // get jacobian of the product with another frame f on the right : d(q*f) = J(q) f
    void multRight_getJ( type::Mat<4,4,real>& J0, type::Mat<4,4,real>& JE)  ;
    // get jacobian of the product with another frame f on the left : d(f*q) = J(q) f
    void multLeft_getJ( type::Mat<4,4,real>& J0, type::Mat<4,4,real>& JE)  ;
    // get jacobian of the transformation : dP = J(p,q) dq
    void pointToParent_getJ( type::Mat<3,4,real>& J0, type::Mat<3,4,real>& JE,const Vec3& p) ;
    // get transformed position change: dP = J(p,q) dq
    Vec3 pointToParent_applyJ( const DualQuatCoord3<real>& dq ,const Vec3& p) ;
    // get quaternion change: dq = JT(p,q) dP
    DualQuatCoord3<real> pointToParent_applyJT( const Vec3& dP ,const Vec3& p) ;
    // get rigid transformation change: d(R,t) = H(q) dq
    type::Mat<3,4,real> rigid_applyH( const DualQuatCoord3<real>& dq ) ;
    // get rotation change: dR = H(q) dq
    type::Mat<3,3,real> rotation_applyH( const DualQuatCoord3<real>& dq ) ;
    // get quaternion change: dq = H^T(q) d(R,t)
    DualQuatCoord3<real> rigid_applyHT( const type::Mat<3,4,real>& dR ) ;
    // get quaternion change: dq = H^T(q) dR
    DualQuatCoord3<real> rotation_applyHT( const type::Mat<3,3,real>& dR ) ;
    // get Jacobian change: dJ = H(p) dq
    type::Mat<3,8,real> pointToParent_applyH( const DualQuatCoord3<real>& dq ,const Vec3& p) ;
    // get quaternion change: dq = H^T(p) dJ
    DualQuatCoord3<real> pointToParent_applyHT( const type::Mat<3,8,real>& dJ ,const Vec3& p) ;

};

#if !defined(SOFA_TYPE_DUALQUAT_CPP)
extern template class SOFA_TYPE_API DualQuatCoord3<double>;
extern template class SOFA_TYPE_API DualQuatCoord3<float>;
#endif

} // namespace sofa::type
