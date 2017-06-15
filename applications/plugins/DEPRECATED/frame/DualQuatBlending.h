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
#ifndef FRAME_DualQuatBlending_H
#define FRAME_DualQuatBlending_H

//#include <sofa/defaulttype/Vec.h>
//#include <sofa/defaulttype/Mat.h>
//#include <sofa/helper/vector.h>
//#include <sofa/helper/rmath.h>
//#include <iostream>

namespace sofa
{

namespace defaulttype
{


/** Template class used to implement the linear blending. The default class does nothing, all the implementation is in the specializations in file DualQuatBlending.inl
    */
template<class In, class Out, class Material, int nbRef, int type>
class DualQuatBlending {};

//using std::endl;
//using sofa::helper::vector;

//template<int N, typename real>
//class DualQuatCoord;

//template<typename real>
//class DualQuatCoord<3,real>
//{
//public:
//    typedef real value_type;
//    typedef real Real;
//    typedef Vec<3,Real> Pos;
//    typedef Vec<3,Real> Vec3;
//    typedef Vec<4,Real> Quat;

//protected:
//    Quat orientation;
//    Quat dual;
//public:
//    DualQuatCoord (const Quat &Dual, const Quat &orient)
//        : dual(Dual), orientation(orient) {}

//    template<typename real2>
//    DualQuatCoord(const DualQuatCoord<3,real2>& c)
//        : dual(c.getDual()), orientation(c.getOrientation()) {}

//    template<typename real2>
//    DualQuatCoord(const RigidCoord<3,real2>& c)
//    {
//        for(unsigned int i=0;i<4;i++) orientation[i] =  c.getOrientation()[i];
//        setTranslation(c.getCenter());
//    }


//    DualQuatCoord () { clear(); }
//    void clear() { dual[0]=dual[1]=dual[2]=dual[3]=orientation[0]=orientation[1]=orientation[2]=orientation[3]=(Real)0.; }



//    template<typename real2>
//    void operator =(const DualQuatCoord<3,real2>& c)
//    {
//        dual = c.getDual();
//        orientation = c.getOrientation();
//    }

//    template<typename real2>
//    void operator =(const RigidCoord<3,real2>& c)
//    {
//        for(unsigned int i=0;i<4;i++) orientation[i] =  c.getOrientation()[i];
//        setTranslation(c.getCenter());
//    }

//    void operator =(const Vec3& p)
//    {
//        setTranslation(p);
//    }

//    void setTranslation(const Vec3& p)
//    {
//        dual[0] =  (real)0.5* ( p[0]*orientation[3] + p[1]*orientation[2] - p[2]*orientation[1] );
//        dual[1] =  (real)0.5* (-p[0]*orientation[2] + p[1]*orientation[3] + p[2]*orientation[0] );
//        dual[2] =  (real)0.5* ( p[0]*orientation[1] - p[1]*orientation[0] + p[2]*orientation[3] );
//        dual[3] = -(real)0.5* ( p[0]*orientation[0] + p[1]*orientation[1] + p[2]*orientation[2] );
//    }

//    Vec3 getTranslation()
//    {
//        Vec3 t;
//        t[0] =  (real)2. * ( -dual[3]*orientation[0] + dual[0]*orientation[3] - dual[1]*orientation[2] + dual[2]*orientation[1] );
//        t[1] =  (real)2. * ( -dual[3]*orientation[1] + dual[0]*orientation[2] + dual[1]*orientation[3] - dual[2]*orientation[0] );
//        t[2] =  (real)2. * ( -dual[3]*orientation[2] - dual[0]*orientation[1] + dual[1]*orientation[0] + dual[2]*orientation[3] );
//        return t;
//    }

//    DualQuatCoord<3,real> operator + (const Vec<6,real>& a) const
//    {
//        DualQuatCoord r;

//        r.orientation[0] = orientation[0] + (real)0.5* (getVOrientation(a)[0] * orientation[3] + getVOrientation(a)[1] * orientation[2] - getVOrientation(a)[2] * orientation[1]);
//        r.orientation[1] = orientation[1] +(real)0.5* (getVOrientation(a)[1] * orientation[3] + getVOrientation(a)[2] * orientation[0] - getVOrientation(a)[0] * orientation[2]);
//        r.orientation[2] = orientation[2] +(real)0.5* (getVOrientation(a)[2] * orientation[3] + getVOrientation(a)[0] * orientation[1] - getVOrientation(a)[1] * orientation[0]);
//        r.orientation[3] = orientation[3] +(real)0.5* (-(getVOrientation(a)[0] * orientation[0] + getVOrientation(a)[1] * orientation[1] + getVOrientation(a)[2] * orientation[2]));

//        r.setTranslation(getTranslation()+getVCenter(a));
//        r.orientation.normalize();
//        return r;
//    }


//    // get velocity/quaternion change mapping : dq = J(q) v
//    void velocity_getJ( Mat<4,3,real>& J0, Mat<4,3,real>& JE)
//    {
//        // multiplication by orientation quaternion
//        J0[0][0] = orientation[3];  	J0[0][1] = orientation[2];  	J0[0][2] =-orientation[1];
//        J0[1][0] =-orientation[2];  	J0[1][1] = orientation[3];  	J0[1][2] = orientation[0];
//        J0[2][0] = orientation[1];  	J0[2][1] =-orientation[0];  	J0[2][2] = orientation[3];
//        J0[3][0] =-orientation[0];  	J0[3][1] =-orientation[1];  	J0[3][2] =-orientation[2];
//        J0*=(real)0.5;

//        Vec3 t=getTranslation();
//        JE[0][0] = dual[3]+orientation[1]*t[1]+orientation[2]*t[2];  	JE[0][1] = dual[2]-orientation[1]*t[0]-orientation[3]*t[2];  	JE[0][2] =-dual[1]-orientation[2]*t[0]+orientation[3]*t[1];
//        JE[1][0] =-dual[2]-orientation[0]*t[1]+orientation[3]*t[2];  	JE[1][1] = dual[3]+orientation[0]*t[0]+orientation[2]*t[2];  	JE[1][2] = dual[0]-orientation[3]*t[0]-orientation[2]*t[1];
//        JE[2][0] = dual[1]-orientation[3]*t[1]-orientation[0]*t[2];  	JE[2][1] =-dual[0]+orientation[3]*t[0]-orientation[1]*t[2];  	JE[2][2] = dual[3]+orientation[0]*t[0]+orientation[1]*t[1];
//        JE[3][0] =-dual[0]+orientation[2]*t[1]-orientation[1]*t[2];  	JE[3][1] =-dual[1]-orientation[2]*t[0]+orientation[0]*t[2];  	JE[3][2] =-dual[2]+orientation[1]*t[0]-orientation[0]*t[1];
//        JE*=(real)0.5;
//    }

//    // get quaternion change: dqn = J(q) dq
//    DualQuatCoord<3,real> velocity_applyJ( const Vec<6,real>& a )
//    {
//        DualQuatCoord r ;

//        r.orientation[0] = (real)0.5* (getVOrientation(a)[0] * orientation[3] + getVOrientation(a)[1] * orientation[2] - getVOrientation(a)[2] * orientation[1]);
//        r.orientation[1] = (real)0.5* (getVOrientation(a)[1] * orientation[3] + getVOrientation(a)[2] * orientation[0] - getVOrientation(a)[0] * orientation[2]);
//        r.orientation[2] = (real)0.5* (getVOrientation(a)[2] * orientation[3] + getVOrientation(a)[0] * orientation[1] - getVOrientation(a)[1] * orientation[0]);
//        r.orientation[3] = (real)0.5* (-(getVOrientation(a)[0] * orientation[0] + getVOrientation(a)[1] * orientation[1] + getVOrientation(a)[2] * orientation[2]));

//        r.setTranslation(getTranslation()+getVCenter(a));
//        r.dual-=dual;

//        //Mat<4,3,real> J0,JE;
//        //velocity_getJ(J0,JE);
//        //r.orientation=J0*getVOrientation(v);
//        //r.dual=JE*getVOrientation(a)+J0*getVCenter(v);
//        return r;
//    }

//    // get velocity : dq = JT(q) dqn
//    Vec<6,real> velocity_applyJT( const DualQuatCoord<3,real>& dq )
//    {
//        Mat<4,3,real> J0,JE;
//        velocity_getJ(J0,JE);
//        Vec<6,real> v;
//        getVOrientation(v)=J0.transposed()*dq.orientation+JE.transposed()*dq.dual;
//        getVCenter(v)=J0.transposed()*dq.dual;
//        return v;
//    }


//    void operator +=(const DualQuatCoord<3,real>& a)
//    {
//        dual += a.getDual();
//        orientation += a.getOrientation();
//    }

//    template<typename real2>
//    void operator*=(real2 a)
//    {
//        orientation *= a;
//        dual *= a;
//    }

//    template<typename real2>
//    void operator/=(real2 a)
//    {
//        orientation /= a;
//        dual /= a;
//    }

//    template<typename real2>
//    DualQuatCoord<3,real> operator*(real2 a) const
//    {
//        DualQuatCoord r = *this;
//        r*=a;
//        return r;
//    }


//    /// dot product, mostly used to compute residuals as sqrt(x*x)
//    Real operator*(const DualQuatCoord<3,real>& a) const
//    {
//        return dual[0]*a.dual[0]+dual[1]*a.dual[1]+dual[2]*a.dual[2]+dual[3]*a.dual[3]
//                +orientation[0]*a.orientation[0]+orientation[1]*a.orientation[1]
//                +orientation[2]*a.orientation[2]+orientation[3]*a.orientation[3];
//    }

//    /// Squared norm
//    real norm2() const
//    {
//        real r = (real)0.;
//        for (int i=0;i<4;i++) r += orientation[i]*orientation[i] + dual[i]*dual[i];
//        return r;
//    }

//    /// Euclidean norm
//    real norm() const
//    {
//        return helper::rsqrt(norm2());
//    }


//    void normalize()
//    {
//        real Q0 = (real) sqrt ( orientation[0]*orientation[0] + orientation[1]*orientation[1] + orientation[2]*orientation[2] + orientation[3]*orientation[3] );
//        if( Q0 == 0) return;
//        real Q0QE = (real) ( orientation[0]*dual[0] + orientation[1]*dual[1] + orientation[2]*dual[2] + orientation[3]*dual[3] );
//        real Q0inv=(real)1./Q0;
//        orientation *= Q0inv;
//        dual -= orientation*Q0QE*Q0inv;
//        dual *= Q0inv;
//    }

//    // get jacobian of the normalization : dqn = J(q) dq
//    void normalize_getJ( Mat<4,4,real>& J0, Mat<4,4,real>& JE)
//    {
//        J0.fill(0); JE.fill(0);

//        unsigned int i,j;
//        real Q0 = (real) sqrt ( orientation[0]*orientation[0] + orientation[1]*orientation[1] + orientation[2]*orientation[2] + orientation[3]*orientation[3] );
//        if(Q0==0) return;
//        real Q0QE = (real) ( orientation[0]*dual[0] + orientation[1]*dual[1] + orientation[2]*dual[2] + orientation[3]*dual[3] );
//        real Q0inv=(real)1./Q0;
//        DualQuatCoord<3,real> qn;
//        qn.orientation = orientation*Q0inv;
//        qn.dual = dual-qn.orientation*Q0QE*Q0inv;
//        qn.dual *= Q0inv;
//        for(i=0;i<4;i++) J0[i][i]=(real)1.-qn.orientation[i]*qn.orientation[i];
//        for(i=0;i<4;i++) for(j=0;j<i;j++) J0[i][j]=J0[j][i]=-qn.orientation[j]*qn.orientation[i];
//        for(i=0;i<4;i++) for(j=0;j<=i;j++) JE[i][j]=JE[j][i]=-qn.dual[j]*qn.orientation[i]-qn.dual[i]*qn.orientation[j];
//        J0 *= Q0inv;
//        JE -= J0*Q0QE*Q0inv;
//        JE *= Q0inv;
//    }

//    // get normalized quaternion change: dqn = J(q) dq
//    DualQuatCoord<3,real> normalize_applyJ( const DualQuatCoord<3,real>& dq )
//    {
//        Mat<4,4,real> J0,JE;
//        normalize_getJ(J0,JE);
//        DualQuatCoord r;
//        r.orientation=J0*dq.orientation;
//        r.dual=JE*dq.orientation+J0*dq.dual;
//        return r;
//    }

//    // get unnormalized quaternion change: dq = JT(q) dqn
//    DualQuatCoord<3,real> normalize_applyJT( const DualQuatCoord<3,real>& dqn )
//    {
//        Mat<4,4,real> J0,JE;
//        normalize_getJ(J0,JE);
//        DualQuatCoord r;
//        r.orientation=J0*dqn.orientation+JE*dqn.dual;
//        r.dual=J0*dqn.dual;
//        return r;
//    }

//    // get Jacobian change: dJ = H(p) dq
//    void  normalize_getdJ( Mat<4,4,real>& dJ0, Mat<4,4,real>& dJE, const DualQuatCoord<3,real>& dq )
//    {
//        dJ0.fill(0); dJE.fill(0);

//        unsigned int i,j;
//        real Q0 = (real) sqrt ( orientation[0]*orientation[0] + orientation[1]*orientation[1] + orientation[2]*orientation[2] + orientation[3]*orientation[3] );
//        if(Q0==0) return;
//        real Q0QE = (real) ( orientation[0]*dual[0] + orientation[1]*dual[1] + orientation[2]*dual[2] + orientation[3]*dual[3] );
//        real Q0inv=(real)1./Q0;
//        real Q0inv2=Q0inv*Q0inv;
//        real Q0QE2=Q0QE*Q0inv2;
//        DualQuatCoord<3,real> qn;
//        qn.orientation = orientation*Q0inv;
//        qn.dual = dual-qn.orientation*Q0QE*Q0inv;
//        qn.dual *= Q0inv;

//        real q0dqe = (real) ( qn.orientation[0]*dq.dual[0] + qn.orientation[1]*dq.dual[1] + qn.orientation[2]*dq.dual[2] + qn.orientation[3]*dq.dual[3]);

//        if(dq.orientation[0] || dq.orientation[1] || dq.orientation[2] || dq.orientation[3]) {
//            real q0dq0 = (real) ( qn.orientation[0]*dq.orientation[0] + qn.orientation[1]*dq.orientation[1] + qn.orientation[2]*dq.orientation[2] + qn.orientation[3]*dq.orientation[3]);
//            real qedq0 = (real) ( qn.dual[0]*dq.orientation[0] + qn.dual[1]*dq.orientation[1] + qn.dual[2]*dq.orientation[2] + qn.dual[3]*dq.orientation[3]);
//            for(i=0;i<4;i++) {
//                for(j=0;j<=i;j++) {
//                    dJ0[i][j]=dJ0[j][i]=-qn.orientation[j]*dq.orientation[i]-qn.orientation[i]*dq.orientation[j]+(real)3.*q0dq0*qn.orientation[i]*qn.orientation[j];
//                    dJE[i][j]=dJE[j][i]=-qn.orientation[j]*dq.dual[i]-qn.orientation[i]*dq.dual[j]+(real)3.*q0dqe*qn.orientation[i]*qn.orientation[j]
//                            -qn.dual[j]*dq.orientation[i]-qn.dual[i]*dq.orientation[j]+(real)3.*qedq0*qn.orientation[i]*qn.orientation[j]
//                            +(real)3.*q0dq0*(qn.dual[j]*qn.orientation[i]+qn.dual[i]*qn.orientation[j]);
//                }
//                dJ0[i][i]-=q0dq0;
//                dJE[i][i]-=q0dqe+qedq0;
//            }
//            dJ0*=Q0inv2; dJE*=Q0inv2;
//            for(i=0;i<4;i++) for(j=0;j<4;j++) dJE[i][j]-=(real)2.*dJ0[i][j]*Q0QE2;
//        }
//        else {
//            for(i=0;i<4;i++) {
//                for(j=0;j<=i;j++) dJE[i][j]=dJE[j][i]=-qn.orientation[j]*dq.dual[i]-qn.orientation[i]*dq.dual[j]+(real)3.*q0dqe*qn.orientation[i]*qn.orientation[j];
//                dJE[i][i]-=q0dqe;
//            }
//            dJE*=Q0inv2;
//        }
//    }

//    Quat& getDual () { return dual; }
//    Quat& getOrientation () { return orientation; }
//    const Quat& getDual () const { return dual; }
//    const Quat& getOrientation () const { return orientation; }

//    static DualQuatCoord<3,real> identity() {
//        DualQuatCoord c;
//        c.getOrientation[3]=(real)1.;
//        return c;
//    }

//    Vec3 rotate(const Vec3& v) const
//    {
//        return Vec3(
//                    (Real)((1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))*v[0] + (2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3])) * v[1] + (2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3])) * v[2]),
//                    (Real)((2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3]))*v[0] + (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]))*v[1] + (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]))*v[2]),
//                    (Real)((2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3]))*v[0] + (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]))*v[1] + (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]))*v[2])
//                    );
//    }
//    Vec3 inverseRotate(const Vec3& v) const
//    {
//        return Vec3(
//                    (Real)((1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))*v[0] + (2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3])) * v[1] + (2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3])) * v[2]),
//                    (Real)((2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3]))*v[0] + (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]))*v[1] + (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]))*v[2]),
//                    (Real)((2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3]))*v[0] + (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]))*v[1] + (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]))*v[2])
//                    );
//    }

//    void invert()
//    {
//        for ( unsigned int j = 0; j < 3; j++ )
//        {
//            orientation[j]=-orientation[j];
//            dual[j]=-dual[j];
//        }
//    }

//    DualQuatCoord<3,real> inverse( )
//    {
//        DualQuatCoord r;
//        for ( unsigned int j = 0; j < 3; j++ )
//        {
//            r.orientation[j]=-orientation[j];
//            r.dual[j]=-dual[j];
//        }
//        r.orientation[3]=orientation[3];
//        r.dual[3]=dual[3];
//        return r;
//    }


//    /// compute the product with another frame on the right
//    DualQuatCoord<3,real> multRight( const DualQuatCoord<3,real>& c ) const
//    {
//        DualQuatCoord r;
//        //r.orientation = orientation * c.getOrientation();
//        //r.dual = orientation * c.getDual() + dual * c.getOrientation();
//        r.orientation[0] = orientation[3]*c.getOrientation()[0] - orientation[2]*c.getOrientation()[1] + orientation[1]*c.getOrientation()[2] + orientation[0]*c.getOrientation()[3];
//        r.orientation[1] = orientation[2]*c.getOrientation()[0] + orientation[3]*c.getOrientation()[1] - orientation[0]*c.getOrientation()[2] + orientation[1]*c.getOrientation()[3];
//        r.orientation[2] =-orientation[1]*c.getOrientation()[0] + orientation[0]*c.getOrientation()[1] + orientation[3]*c.getOrientation()[2] + orientation[2]*c.getOrientation()[3];
//        r.orientation[3] =-orientation[0]*c.getOrientation()[0] - orientation[1]*c.getOrientation()[1] - orientation[2]*c.getOrientation()[2] + orientation[3]*c.getOrientation()[3];
//        r.dual[0] = orientation[3]*c.getDual()[0] - orientation[2]*c.getDual()[1] + orientation[1]*c.getDual()[2] + orientation[0]*c.getDual()[3];
//        r.dual[1] = orientation[2]*c.getDual()[0] + orientation[3]*c.getDual()[1] - orientation[0]*c.getDual()[2] + orientation[1]*c.getDual()[3];
//        r.dual[2] =-orientation[1]*c.getDual()[0] + orientation[0]*c.getDual()[1] + orientation[3]*c.getDual()[2] + orientation[2]*c.getDual()[3];
//        r.dual[3] =-orientation[0]*c.getDual()[0] - orientation[1]*c.getDual()[1] - orientation[2]*c.getDual()[2] + orientation[3]*c.getDual()[3];
//        r.dual[0]+= dual[3]*c.getOrientation()[0] - dual[2]*c.getOrientation()[1] + dual[1]*c.getOrientation()[2] + dual[0]*c.getOrientation()[3];
//        r.dual[1]+= dual[2]*c.getOrientation()[0] + dual[3]*c.getOrientation()[1] - dual[0]*c.getOrientation()[2] + dual[1]*c.getOrientation()[3];
//        r.dual[2]+=-dual[1]*c.getOrientation()[0] + dual[0]*c.getOrientation()[1] + dual[3]*c.getOrientation()[2] + dual[2]*c.getOrientation()[3];
//        r.dual[3]+=-dual[0]*c.getOrientation()[0] - dual[1]*c.getOrientation()[1] - dual[2]*c.getOrientation()[2] + dual[3]*c.getOrientation()[3];
//        return r;
//    }

//    /// get jacobian of the product with another frame f on the right : d(q*f) = J(q) f
//    void multRight_getJ( Mat<4,4,real>& J0,Mat<4,4,real>& JE)
//    {
//        J0[0][0] = orientation[3];	J0[0][1] =-orientation[2];	J0[0][2] = orientation[1];	J0[0][3] = orientation[0];
//        J0[1][0] = orientation[2];	J0[1][1] = orientation[3];	J0[1][2] =-orientation[0];	J0[1][3] = orientation[1];
//        J0[2][0] =-orientation[1];	J0[2][1] = orientation[0];	J0[2][2] = orientation[3];	J0[2][3] = orientation[2];
//        J0[3][0] =-orientation[0];	J0[3][1] =-orientation[1];	J0[3][2] =-orientation[2];	J0[3][3] = orientation[3];
//        JE[0][0] = dual[3];		JE[0][1] =-dual[2];		JE[0][2] = dual[1];		JE[0][3] = dual[0];
//        JE[1][0] = dual[2];		JE[1][1] = dual[3];		JE[1][2] =-dual[0];		JE[1][3] = dual[1];
//        JE[2][0] =-dual[1];		JE[2][1] = dual[0];		JE[2][2] = dual[3];		JE[2][3] = dual[2];
//        JE[3][0] =-dual[0];		JE[3][1] =-dual[1];		JE[3][2] =-dual[2];		JE[3][3] = dual[3];
//    }

//    /// Apply a transformation with respect to itself
//    DualQuatCoord<3,real> multLeft( const DualQuatCoord<3,real>& c )
//    {
//        DualQuatCoord r;
//        r.orientation[0] = orientation[3]*c.getOrientation()[0] + orientation[2]*c.getOrientation()[1] - orientation[1]*c.getOrientation()[2] + orientation[0]*c.getOrientation()[3];
//        r.orientation[1] =-orientation[2]*c.getOrientation()[0] + orientation[3]*c.getOrientation()[1] + orientation[0]*c.getOrientation()[2] + orientation[1]*c.getOrientation()[3];
//        r.orientation[2] = orientation[1]*c.getOrientation()[0] - orientation[0]*c.getOrientation()[1] + orientation[3]*c.getOrientation()[2] + orientation[2]*c.getOrientation()[3];
//        r.orientation[3] =-orientation[0]*c.getOrientation()[0] - orientation[1]*c.getOrientation()[1] - orientation[2]*c.getOrientation()[2] + orientation[3]*c.getOrientation()[3];
//        r.dual[0] = orientation[3]*c.getDual()[0] + orientation[2]*c.getDual()[1] - orientation[1]*c.getDual()[2] + orientation[0]*c.getDual()[3];
//        r.dual[1] =-orientation[2]*c.getDual()[0] + orientation[3]*c.getDual()[1] + orientation[0]*c.getDual()[2] + orientation[1]*c.getDual()[3];
//        r.dual[2] = orientation[1]*c.getDual()[0] - orientation[0]*c.getDual()[1] + orientation[3]*c.getDual()[2] + orientation[2]*c.getDual()[3];
//        r.dual[3] =-orientation[0]*c.getDual()[0] - orientation[1]*c.getDual()[1] - orientation[2]*c.getDual()[2] + orientation[3]*c.getDual()[3];
//        r.dual[0]+= dual[3]*c.getOrientation()[0] + dual[2]*c.getOrientation()[1] - dual[1]*c.getOrientation()[2] + dual[0]*c.getOrientation()[3];
//        r.dual[1]+=-dual[2]*c.getOrientation()[0] + dual[3]*c.getOrientation()[1] + dual[0]*c.getOrientation()[2] + dual[1]*c.getOrientation()[3];
//        r.dual[2]+= dual[1]*c.getOrientation()[0] - dual[0]*c.getOrientation()[1] + dual[3]*c.getOrientation()[2] + dual[2]*c.getOrientation()[3];
//        r.dual[3]+=-dual[0]*c.getOrientation()[0] - dual[1]*c.getOrientation()[1] - dual[2]*c.getOrientation()[2] + dual[3]*c.getOrientation()[3];
//        return r;
//    }
//    /// get jacobian of the product with another frame f on the left : d(f*q) = J(q) f
//    void multLeft_getJ( Mat<4,4,real>& J0,Mat<4,4,real>& JE)
//    {
//        J0[0][0] = orientation[3];	J0[0][1] = orientation[2];	J0[0][2] =-orientation[1];	J0[0][3] = orientation[0];
//        J0[1][0] =-orientation[2];	J0[1][1] = orientation[3];	J0[1][2] = orientation[0];	J0[1][3] = orientation[1];
//        J0[2][0] = orientation[1];	J0[2][1] =-orientation[0];	J0[2][2] = orientation[3];	J0[2][3] = orientation[2];
//        J0[3][0] =-orientation[0];	J0[3][1] =-orientation[1];	J0[3][2] =-orientation[2];	J0[3][3] = orientation[3];
//        JE[0][0] = dual[3];		JE[0][1] = dual[2];		JE[0][2] =-dual[1];		JE[0][3] = dual[0];
//        JE[1][0] =-dual[2];		JE[1][1] = dual[3];		JE[1][2] = dual[0];		JE[1][3] = dual[1];
//        JE[2][0] = dual[1];		JE[2][1] =-dual[0];		JE[2][2] = dual[3];		JE[2][3] = dual[2];
//        JE[3][0] =-dual[0];		JE[3][1] =-dual[1];		JE[3][2] =-dual[2];		JE[3][3] = dual[3];
//    }


//    /// Write to the given matrix
//    template<typename  real2>
//    void toMatrix( Mat<3,4,real2>& m) const
//    {
//        m[0][0] = (real2) (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]));
//        m[0][1] = (real2) (2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3]));
//        m[0][2] = (real2) (2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3]));

//        m[1][0] = (real2) (2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3]));
//        m[1][1] = (real2) (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]));
//        m[1][2] = (real2) (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]));

//        m[2][0] = (real2) (2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3]));
//        m[2][1] = (real2) (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]));
//        m[2][2] = (real2) (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]));

//        Vec3 p=getTranslation();
//        m[0][3] =  (real2) p[0];
//        m[1][3] =  (real2) p[1];
//        m[2][3] =  (real2) p[2];
//    }

//    template<typename  real2>
//    void toRotationMatrix( Mat<3,3,real2>& m) const
//    {
//        m[0][0] = (real2) (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]));
//        m[0][1] = (real2) (2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3]));
//        m[0][2] = (real2) (2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3]));

//        m[1][0] = (real2) (2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3]));
//        m[1][1] = (real2) (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]));
//        m[1][2] = (real2) (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]));

//        m[2][0] = (real2) (2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3]));
//        m[2][1] = (real2) (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]));
//        m[2][2] = (real2) (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]));
//    }


//    /// Project a point from the child frame to the parent frame: P = R(q)p + t(q)
//    Vec3 pointToParent( const Vec3& p )
//    {
//        Vec3 p2;
//        p2[0] = (real) ((1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))*p[0] + (2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3])) * p[1] + (2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3])) * p[2]);
//        p2[1] = (real) ((2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3]))*p[0] + (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]))*p[1] + (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]))*p[2]);
//        p2[2] = (real) ((2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3]))*p[0] + (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]))*p[1] + (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]))*p[2]);
//        p2+=getTranslation();
//        return p2;
//    }

//    // get jacobian of the transformation : dP = J(p,q) dq
//    void pointToParent_getJ( Mat<3,4,real>& J0,Mat<3,4,real>& JE,const Vec3& p)
//    {
//        J0.fill(0); JE.fill(0);
//        J0[0][0] = (real)2.*(- dual[3] + orientation[0]*p[0] + orientation[1]*p[1] + orientation[2]*p[2]);	J0[0][1] =   (real)2.*(dual[2] - orientation[1]*p[0] + orientation[0]*p[1] + orientation[3]*p[2]);	J0[0][2] = (real)2.*(- dual[1] - orientation[2]*p[0] - orientation[3]*p[1] + orientation[0]*p[2]);	J0[0][3] = (real)2.*(  dual[0] + orientation[3]*p[0] - orientation[2]*p[1] + orientation[1]*p[2]);
//        J0[1][0] = -J0[0][1];	J0[1][1] = J0[0][0];	J0[1][2] = J0[0][3];	J0[1][3] = -J0[0][2];
//        J0[2][0] = -J0[0][2];	J0[2][1] = -J0[0][3];	J0[2][2] = J0[0][0];	J0[2][3] = J0[0][1];
//        JE[0][0] = (real)2.* orientation[3];		JE[0][1] = -(real)2.*orientation[2];		JE[0][2] =  (real)2.*orientation[1];		JE[0][3] = -(real)2.*orientation[0];
//        JE[1][0] = -JE[0][1];	JE[1][1] = JE[0][0];	JE[1][2] = JE[0][3];	JE[1][3] = -JE[0][2];
//        JE[2][0] = -JE[0][2];	JE[2][1] = -JE[0][3];	JE[2][2] = JE[0][0];	JE[2][3] = JE[0][1];
//    }

//    // get transformed position change: dP = J(p,q) dq
//    Vec3 pointToParent_applyJ( const DualQuatCoord<3,real>& dq ,const Vec3& p)
//    {
//        Mat<3,4,real> J0,JE;
//        pointToParent_getJ(J0,JE,p);
//        Vec3 r=J0*dq.orientation+JE*dq.dual;
//        return r;
//    }

//    // get quaternion change: dq = JT(p,q) dP
//    DualQuatCoord<3,real> pointToParent_applyJT( const Vec3& dP ,const Vec3& p)
//    {
//        Mat<3,4,real> J0,JE;
//        pointToParent_getJ(J0,JE,p);
//        DualQuatCoord r;
//        r.orientation=J0.transposed()*dP;
//        r.dual=JE.transposed()*dP;
//        return r;
//    }


//    // get rigid transformation change: d(R,t) = H(q) dq
//    Mat<3,4,real> rigid_applyH( const DualQuatCoord<3,real>& dq )
//    {
//        Mat<3,4,real> dR;
//        dR[0][0]=(real)2.*(-2*orientation[1]*dq.orientation[1]-2*orientation[2]*dq.orientation[2]);
//        dR[0][1]=(real)2.*(orientation[1]*dq.orientation[0]+orientation[0]*dq.orientation[1]-orientation[3]*dq.orientation[2]-orientation[2]*dq.orientation[3]);
//        dR[0][2]=(real)2.*(orientation[2]*dq.orientation[0]+orientation[3]*dq.orientation[1]+orientation[0]*dq.orientation[2]+orientation[1]*dq.orientation[3]);
//        dR[0][3]=(real)2.*(-dual[3]*dq.orientation[0]+dual[2]*dq.orientation[1]-dual[1]*dq.orientation[2]+dual[0]*dq.orientation[3]+orientation[3]*dq.dual[0]-orientation[2]*dq.dual[1]+orientation[1]*dq.dual[2]-orientation[0]*dq.dual[3]);
//        dR[1][0]=(real)2.*(orientation[1]*dq.orientation[0]+orientation[0]*dq.orientation[1]+orientation[3]*dq.orientation[2]+orientation[2]*dq.orientation[3]);
//        dR[1][1]=(real)2.*(-2*orientation[0]*dq.orientation[0]-2*orientation[2]*dq.orientation[2]);
//        dR[1][2]=(real)2.*(-orientation[3]*dq.orientation[0]+orientation[2]*dq.orientation[1]+orientation[1]*dq.orientation[2]-orientation[0]*dq.orientation[3]);
//        dR[1][3]=(real)2.*(-dual[2]*dq.orientation[0]-dual[3]*dq.orientation[1]+dual[0]*dq.orientation[2]+dual[1]*dq.orientation[3]+orientation[2]*dq.dual[0]+orientation[3]*dq.dual[1]-orientation[0]*dq.dual[2]-orientation[1]*dq.dual[3]);
//        dR[2][0]=(real)2.*(orientation[2]*dq.orientation[0]-orientation[3]*dq.orientation[1]+orientation[0]*dq.orientation[2]-orientation[1]*dq.orientation[3]);
//        dR[2][1]=(real)2.*(orientation[3]*dq.orientation[0]+orientation[2]*dq.orientation[1]+orientation[1]*dq.orientation[2]+orientation[0]*dq.orientation[3]);
//        dR[2][2]=(real)2.*(-2*orientation[0]*dq.orientation[0]-2*orientation[1]*dq.orientation[1]);
//        dR[2][3]=(real)2.*(dual[1]*dq.orientation[0]-dual[0]*dq.orientation[1]-dual[3]*dq.orientation[2]+dual[2]*dq.orientation[3]-orientation[1]*dq.dual[0]+orientation[0]*dq.dual[1]+orientation[3]*dq.dual[2]-orientation[2]*dq.dual[3]);
//        return dR;
//    }
//    // get rotation change: dR = H(q) dq
//    Mat<3,3,real> rotation_applyH( const DualQuatCoord<3,real>& dq )
//    {
//        Mat<3,3,real> dR;
//        dR[0][0]=(real)2.*(-2*orientation[1]*dq.orientation[1]-2*orientation[2]*dq.orientation[2]);
//        dR[0][1]=(real)2.*(orientation[1]*dq.orientation[0]+orientation[0]*dq.orientation[1]-orientation[3]*dq.orientation[2]-orientation[2]*dq.orientation[3]);
//        dR[0][2]=(real)2.*(orientation[2]*dq.orientation[0]+orientation[3]*dq.orientation[1]+orientation[0]*dq.orientation[2]+orientation[1]*dq.orientation[3]);
//        dR[1][0]=(real)2.*(orientation[1]*dq.orientation[0]+orientation[0]*dq.orientation[1]+orientation[3]*dq.orientation[2]+orientation[2]*dq.orientation[3]);
//        dR[1][1]=(real)2.*(-2*orientation[0]*dq.orientation[0]-2*orientation[2]*dq.orientation[2]);
//        dR[1][2]=(real)2.*(-orientation[3]*dq.orientation[0]+orientation[2]*dq.orientation[1]+orientation[1]*dq.orientation[2]-orientation[0]*dq.orientation[3]);
//        dR[2][0]=(real)2.*(orientation[2]*dq.orientation[0]-orientation[3]*dq.orientation[1]+orientation[0]*dq.orientation[2]-orientation[1]*dq.orientation[3]);
//        dR[2][1]=(real)2.*(orientation[3]*dq.orientation[0]+orientation[2]*dq.orientation[1]+orientation[1]*dq.orientation[2]+orientation[0]*dq.orientation[3]);
//        dR[2][2]=(real)2.*(-2*orientation[0]*dq.orientation[0]-2*orientation[1]*dq.orientation[1]);
//        return dR;
//    }

//    // get quaternion change: dq = H^T(q) d(R,t)
//    DualQuatCoord<3,real> rigid_applyHT( const Mat<3,4,real>& dR )
//    {
//        DualQuatCoord r;
//        r.orientation[0]=(real)2.*(orientation[1]*dR[0][1]+orientation[2]*dR[0][2]-dual[3]*dR[0][3]+orientation[1]*dR[1][0]-2*orientation[0]*dR[1][1]-orientation[3]*dR[1][2]-dual[2]*dR[1][3]+orientation[2]*dR[2][0]+orientation[3]*dR[2][1]-2*orientation[0]*dR[2][2]+dual[1]*dR[2][3]);
//        r.orientation[1]=(real)2.*(-2*orientation[1]*dR[0][0]+orientation[0]*dR[0][1]+orientation[3]*dR[0][2]+dual[2]*dR[0][3]+orientation[0]*dR[1][0]+orientation[2]*dR[1][2]-dual[3]*dR[1][3]-orientation[3]*dR[2][0]+orientation[2]*dR[2][1]-2*orientation[1]*dR[2][2]-dual[0]*dR[2][3]);
//        r.orientation[2]=(real)2.*(-2*orientation[2]*dR[0][0]-orientation[3]*dR[0][1]+orientation[0]*dR[0][2]-dual[1]*dR[0][3]+orientation[3]*dR[1][0]-2*orientation[2]*dR[1][1]+orientation[1]*dR[1][2]+dual[0]*dR[1][3]+orientation[0]*dR[2][0]+orientation[1]*dR[2][1]-dual[3]*dR[2][3]);
//        r.orientation[3]=(real)2.*(-orientation[2]*dR[0][1]+orientation[1]*dR[0][2]+dual[0]*dR[0][3]+orientation[2]*dR[1][0]-orientation[0]*dR[1][2]+dual[1]*dR[1][3]-orientation[1]*dR[2][0]+orientation[0]*dR[2][1]+dual[2]*dR[2][3]);
//        r.dual[0]=(real)2.*(orientation[3]*dR[0][3]+orientation[2]*dR[1][3]-orientation[1]*dR[2][3]);
//        r.dual[1]=(real)2.*(-orientation[2]*dR[0][3]+orientation[3]*dR[1][3]+orientation[0]*dR[2][3]);
//        r.dual[2]=(real)2.*(orientation[1]*dR[0][3]-orientation[0]*dR[1][3]+orientation[3]*dR[2][3]);
//        r.dual[3]=(real)2.*(-orientation[0]*dR[0][3]-orientation[1]*dR[1][3]-orientation[2]*dR[2][3]);
//        return r;
//    }
//    // get quaternion change: dq = H^T(q) dR
//    DualQuatCoord<3,real> rotation_applyHT( const Mat<3,3,real>& dR )
//    {
//        DualQuatCoord r;
//        r.orientation[0]=(real)2.*(orientation[1]*dR[0][1]+orientation[2]*dR[0][2]+orientation[1]*dR[1][0]-2*orientation[0]*dR[1][1]-orientation[3]*dR[1][2]+orientation[2]*dR[2][0]+orientation[3]*dR[2][1]-2*orientation[0]*dR[2][2]);
//        r.orientation[1]=(real)2.*(-2*orientation[1]*dR[0][0]+orientation[0]*dR[0][1]+orientation[3]*dR[0][2]+orientation[0]*dR[1][0]+orientation[2]*dR[1][2]-orientation[3]*dR[2][0]+orientation[2]*dR[2][1]-2*orientation[1]*dR[2][2]);
//        r.orientation[2]=(real)2.*(-2*orientation[2]*dR[0][0]-orientation[3]*dR[0][1]+orientation[0]*dR[0][2]+orientation[3]*dR[1][0]-2*orientation[2]*dR[1][1]+orientation[1]*dR[1][2]+orientation[0]*dR[2][0]+orientation[1]*dR[2][1]);
//        r.orientation[3]=(real)2.*(-orientation[2]*dR[0][1]+orientation[1]*dR[0][2]+orientation[2]*dR[1][0]-orientation[0]*dR[1][2]-orientation[1]*dR[2][0]+orientation[0]*dR[2][1]);
//        r.dual[0]=r.dual[1]=r.dual[2]=r.dual[3]=(real)0.;
//        return r;
//    }

//    // get Jacobian change: dJ = H(p) dq
//    Mat<3,8,real> pointToParent_applyH( const DualQuatCoord<3,real>& dq ,const Vec3& p)
//    {
//        Mat<3,8,real> dJ;
//        dJ.fill(0);
//        dJ[0][0] = (real)2.*(- dq.dual[3] + dq.orientation[0]*p[0] + dq.orientation[1]*p[1] + dq.orientation[2]*p[2]);	dJ[0][1] =   (real)2.*(dq.dual[2] - dq.orientation[1]*p[0] + dq.orientation[0]*p[1] + dq.orientation[3]*p[2]);	dJ[0][2] = (real)2.*(- dq.dual[1] - dq.orientation[2]*p[0] - dq.orientation[3]*p[1] + dq.orientation[0]*p[2]);	dJ[0][3] = (real)2.*(  dq.dual[0] + dq.orientation[3]*p[0] - dq.orientation[2]*p[1] + dq.orientation[1]*p[2]);
//        dJ[1][0] = -dJ[0][1];	dJ[1][1] = dJ[0][0];	dJ[1][2] = dJ[0][3];	dJ[1][3] = -dJ[0][2];
//        dJ[2][0] = -dJ[0][2];	dJ[2][1] = -dJ[0][3];	dJ[2][2] = dJ[0][0];	dJ[2][3] = dJ[0][1];
//        dJ[0][4] =  (real)2.*dq.orientation[3];		dJ[0][5] = -(real)2.*dq.orientation[2];		dJ[0][6] =  (real)2.*dq.orientation[1];		dJ[0][7] = -(real)2.*dq.orientation[0];
//        dJ[1][4] = -dJ[0][5];	dJ[1][5] = dJ[0][4];	dJ[1][6] = dJ[0][7];	dJ[1][7] = -dJ[0][6];
//        dJ[2][4] = -dJ[0][6];	dJ[2][5] = -dJ[0][7];	dJ[2][6] = dJ[0][4];	dJ[2][7] = dJ[0][5];
//        return dJ;
//    }

//    // get quaternion change: dq = H^T(p) dJ
//    DualQuatCoord<3,real> pointToParent_applyHT( const Mat<3,8,real>& dJ ,const Vec3& p)
//    {
//        DualQuatCoord r;
//        r.orientation[0]=(real)2.*(p[0]*dJ[0][0]+p[1]*dJ[0][1]+p[2]*dJ[0][2]-dJ[0][7]-p[1]*dJ[1][0]+p[0]*dJ[1][1]-p[2]*dJ[1][3]-dJ[1][6]-p[2]*dJ[2][0]+p[0]*dJ[2][2]+p[1]*dJ[2][3]+dJ[2][5]);
//        r.orientation[1]=(real)2.*(p[1]*dJ[0][0]-p[0]*dJ[0][1]+p[2]*dJ[0][3]+dJ[0][6]+p[0]*dJ[1][0]+p[1]*dJ[1][1]+p[2]*dJ[1][2]-dJ[1][7]-p[2]*dJ[2][1]+p[1]*dJ[2][2]-p[0]*dJ[2][3]-dJ[2][4]);
//        r.orientation[2]=(real)2.*(p[2]*dJ[0][0]-p[0]*dJ[0][2]-p[1]*dJ[0][3]-dJ[0][5]+p[2]*dJ[1][1]-p[1]*dJ[1][2]+p[0]*dJ[1][3]+dJ[1][4]+p[0]*dJ[2][0]+p[1]*dJ[2][1]+p[2]*dJ[2][2]-dJ[2][7]);
//        r.orientation[3]=(real)2.*(p[2]*dJ[0][1]-p[1]*dJ[0][2]+p[0]*dJ[0][3]+dJ[0][4]-p[2]*dJ[1][0]+p[0]*dJ[1][2]+p[1]*dJ[1][3]+dJ[1][5]+p[1]*dJ[2][0]-p[0]*dJ[2][1]+p[2]*dJ[2][3]+dJ[2][6]);
//        r.dual[0]=(real)2.*(dJ[0][3]+dJ[1][2]-dJ[2][1]);
//        r.dual[1]=(real)2.*(-dJ[0][2]+dJ[1][3]+dJ[2][0]);
//        r.dual[2]=(real)2.*(dJ[0][1]-dJ[1][0]+dJ[2][3]);
//        r.dual[3]=(real)2.*(-dJ[0][0]-dJ[1][1]-dJ[2][2]);
//        return r;
//    }

//    /// Project a point from the parent frame to the child frame
//    Vec3 pointToChild( const Vec3& v ) const
//    {
//        Vec3 p,v2=v-getTranslation();
//        p[0]=(real)((1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))*v2[0] + (2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3])) * v2[1] + (2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3])) * v2[2]);
//        p[1]=(real)((2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3]))*v2[0] + (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]))*v2[1] + (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]))*v2[2]);
//        p[2]=(real)((2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3]))*v2[0] + (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]))*v2[1] + (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]))*v2[2]);
//        return p;
//    }

//    /// compute the projection of a vector from the parent frame to the child
//    Vec3 vectorToChild( const Vec3& v ) const {
//        //return orientation.inverseRotate(v);
//        Vec3 p;
//        p[0]=(real)((1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))*v[0] + (2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3])) * v[1] + (2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3])) * v[2]);
//        p[1]=(real)((2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3]))*v[0] + (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]))*v[1] + (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]))*v[2]);
//        p[2]=(real)((2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3]))*v[0] + (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]))*v[1] + (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]))*v[2]);
//        return p;
//    }

//    /// write to an output stream
//    inline friend std::ostream& operator << ( std::ostream& out, const DualQuatCoord<3,real>& v ){
//        out<<v.dual<<" "<<v.orientation;
//        return out;
//    }
//    /// read from an input stream
//    inline friend std::istream& operator >> ( std::istream& in, DualQuatCoord<3,real>& v ){
//        in>>v.dual>>v.orientation;
//        return in;
//    }
//    static int max_size()
//    {
//        return 3;
//    }

//    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
//    enum { total_size = 8 };
//    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for DualQuats)
//    enum { spatial_dimensions = 3 };

//    real* ptr() { return dual.ptr(); }
//    const real* ptr() const { return dual.ptr(); }

//    static unsigned int size(){return 8;}

//    /// Access to i-th element.
//    real& operator[](int i)
//    {
//        if (i<4)
//            return this->dual(i);
//        else
//            return this->orientation[i-4];
//    }

//    /// Const access to i-th element.
//    const real& operator[](int i) const
//    {
//        if (i<4)
//            return this->dual(i);
//        else
//            return this->orientation[i-4];
//    }
//};





} // namespace defaulttype


} // namespace sofa


#endif
