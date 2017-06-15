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
#ifndef SOFA_HELPER_DUALQUAT_INL
#define SOFA_HELPER_DUALQUAT_INL

#include "DualQuat.h"

namespace sofa
{

namespace helper
{

template<typename real>
real DualQuatCoord3<real>::norm2() const
{
    real r = (real)0.;
    for (int i=0; i<4; i++) r += orientation[i]*orientation[i] + dual[i]*dual[i];
    return r;
}

template<typename real>
void DualQuatCoord3<real>::normalize()
{
    real Q0 = (real) sqrt ( orientation[0]*orientation[0] + orientation[1]*orientation[1] + orientation[2]*orientation[2] + orientation[3]*orientation[3] );
    if( Q0 == 0) return;
    real Q0QE = (real) ( orientation[0]*dual[0] + orientation[1]*dual[1] + orientation[2]*dual[2] + orientation[3]*dual[3] );
    real Q0inv=(real)1./Q0;
    orientation *= Q0inv;
    dual -= orientation*Q0QE*Q0inv;
    dual *= Q0inv;
}



// get velocity/quaternion change mapping : dq = J(q) v
template<typename real>
void DualQuatCoord3<real>::velocity_getJ( sofa::defaulttype::Mat<4,3,real>& J0, sofa::defaulttype::Mat<4,3,real>& JE)
{
    // multiplication by orientation quaternion
    J0[0][0] = orientation[3];  	J0[0][1] = orientation[2];  	J0[0][2] =-orientation[1];
    J0[1][0] =-orientation[2];  	J0[1][1] = orientation[3];  	J0[1][2] = orientation[0];
    J0[2][0] = orientation[1];  	J0[2][1] =-orientation[0];  	J0[2][2] = orientation[3];
    J0[3][0] =-orientation[0];  	J0[3][1] =-orientation[1];  	J0[3][2] =-orientation[2];
    J0*=(real)0.5;

    sofa::defaulttype::Vec<3,real> t=getTranslation();
    JE[0][0] = dual[3]+orientation[1]*t[1]+orientation[2]*t[2];  	JE[0][1] = dual[2]-orientation[1]*t[0]-orientation[3]*t[2];  	JE[0][2] =-dual[1]-orientation[2]*t[0]+orientation[3]*t[1];
    JE[1][0] =-dual[2]-orientation[0]*t[1]+orientation[3]*t[2];  	JE[1][1] = dual[3]+orientation[0]*t[0]+orientation[2]*t[2];  	JE[1][2] = dual[0]-orientation[3]*t[0]-orientation[2]*t[1];
    JE[2][0] = dual[1]-orientation[3]*t[1]-orientation[0]*t[2];  	JE[2][1] =-dual[0]+orientation[3]*t[0]-orientation[1]*t[2];  	JE[2][2] = dual[3]+orientation[0]*t[0]+orientation[1]*t[1];
    JE[3][0] =-dual[0]+orientation[2]*t[1]-orientation[1]*t[2];  	JE[3][1] =-dual[1]-orientation[2]*t[0]+orientation[0]*t[2];  	JE[3][2] =-dual[2]+orientation[1]*t[0]-orientation[0]*t[1];
    JE*=(real)0.5;
}

// get quaternion change: dq = J(q) v
template<typename real>
DualQuatCoord3<real> DualQuatCoord3<real>::velocity_applyJ( const sofa::defaulttype::Vec<6,real>& a )
{
    DualQuatCoord3 r ;

    r.orientation[0] = (real)0.5* (a[3] * orientation[3] + a[4] * orientation[2] - a[5] * orientation[1]);
    r.orientation[1] = (real)0.5* (a[4] * orientation[3] + a[5] * orientation[0] - a[3] * orientation[2]);
    r.orientation[2] = (real)0.5* (a[5] * orientation[3] + a[3] * orientation[1] - a[4] * orientation[0]);
    r.orientation[3] = (real)0.5* (-(a[3] * orientation[0] + a[4] * orientation[1] + a[5] * orientation[2]));

    r.setTranslation(getTranslation()+Vec3(a[0],a[1],a[2]));
    r.dual-=dual;

    //Mat<4,3,real> J0,JE;
    //velocity_getJ(J0,JE);
    //r.orientation=J0*getVOrientation(v);
    //r.dual=JE*getVOrientation(a)+J0*getVCenter(v);
    return r;
}

// get velocity : v = JT(q) dq
template<typename real>
sofa::defaulttype::Vec<6,real> DualQuatCoord3<real>::velocity_applyJT( const DualQuatCoord3<real>& dq )
{
    sofa::defaulttype::Mat<4,3,real> J0,JE;
    velocity_getJ(J0,JE);
    sofa::defaulttype::Vec<3,real> omega=J0.transposed()*dq.orientation+JE.transposed()*dq.dual;
    sofa::defaulttype::Vec<3,real> vel=J0.transposed()*dq.dual;
    sofa::defaulttype::Vec<6,real> v;
    for(unsigned int i=0; i<3; i++) {v[i]=vel[i]; v[i+3]=omega[i];}
    return v;
}

// get jacobian of the normalization : dqn = J(q) dq
template<typename real>
void DualQuatCoord3<real>::normalize_getJ( sofa::defaulttype::Mat<4,4,real>& J0, sofa::defaulttype::Mat<4,4,real>& JE)
{
    J0.fill(0); JE.fill(0);

    unsigned int i,j;
    real Q0 = (real) sqrt ( orientation[0]*orientation[0] + orientation[1]*orientation[1] + orientation[2]*orientation[2] + orientation[3]*orientation[3] );
    if(Q0==0) return;
    real Q0QE = (real) ( orientation[0]*dual[0] + orientation[1]*dual[1] + orientation[2]*dual[2] + orientation[3]*dual[3] );
    real Q0inv=(real)1./Q0;
    DualQuatCoord3<real> qn;
    qn.orientation = orientation*Q0inv;
    qn.dual = dual-qn.orientation*Q0QE*Q0inv;
    qn.dual *= Q0inv;
    for(i=0; i<4; i++) J0[i][i]=(real)1.-qn.orientation[i]*qn.orientation[i];
    for(i=0; i<4; i++) for(j=0; j<i; j++) J0[i][j]=J0[j][i]=-qn.orientation[j]*qn.orientation[i];
    for(i=0; i<4; i++) for(j=0; j<=i; j++) JE[i][j]=JE[j][i]=-qn.dual[j]*qn.orientation[i]-qn.dual[i]*qn.orientation[j];
    J0 *= Q0inv;
    JE -= J0*Q0QE*Q0inv;
    JE *= Q0inv;
}


// get normalized quaternion change: dqn = J(q) dq
template<typename real>
DualQuatCoord3<real> DualQuatCoord3<real>::normalize_applyJ( const DualQuatCoord3<real>& dq )
{
    sofa::defaulttype::Mat<4,4,real> J0,JE;
    normalize_getJ(J0,JE);
    DualQuatCoord3<real> r;
    r.orientation=J0*dq.orientation;
    r.dual=JE*dq.orientation+J0*dq.dual;
    return r;
}


// get unnormalized quaternion change: dq = JT(q) dqn
template<typename real>
DualQuatCoord3<real> DualQuatCoord3<real>::normalize_applyJT( const DualQuatCoord3<real>& dqn )
{
    sofa::defaulttype::Mat<4,4,real> J0,JE;
    normalize_getJ(J0,JE);
    DualQuatCoord3<real> r;
    r.orientation=J0*dqn.orientation+JE*dqn.dual;
    r.dual=J0*dqn.dual;
    return r;
}

// get Jacobian change: dJ = H(p) dq
template<typename real>
void  DualQuatCoord3<real>::normalize_getdJ( sofa::defaulttype::Mat<4,4,real>& dJ0, sofa::defaulttype::Mat<4,4,real>& dJE, const DualQuatCoord3<real>& dq )
{
    dJ0.fill(0); dJE.fill(0);

    unsigned int i,j;
    real Q0 = (real) sqrt ( orientation[0]*orientation[0] + orientation[1]*orientation[1] + orientation[2]*orientation[2] + orientation[3]*orientation[3] );
    if(Q0==0) return;
    real Q0QE = (real) ( orientation[0]*dual[0] + orientation[1]*dual[1] + orientation[2]*dual[2] + orientation[3]*dual[3] );
    real Q0inv=(real)1./Q0;
    real Q0inv2=Q0inv*Q0inv;
    real Q0QE2=Q0QE*Q0inv2;
    DualQuatCoord3<real> qn;
    qn.orientation = orientation*Q0inv;
    qn.dual = dual-qn.orientation*Q0QE*Q0inv;
    qn.dual *= Q0inv;

    real q0dqe = (real) ( qn.orientation[0]*dq.dual[0] + qn.orientation[1]*dq.dual[1] + qn.orientation[2]*dq.dual[2] + qn.orientation[3]*dq.dual[3]);

    if(dq.orientation[0] || dq.orientation[1] || dq.orientation[2] || dq.orientation[3])
    {
        real q0dq0 = (real) ( qn.orientation[0]*dq.orientation[0] + qn.orientation[1]*dq.orientation[1] + qn.orientation[2]*dq.orientation[2] + qn.orientation[3]*dq.orientation[3]);
        real qedq0 = (real) ( qn.dual[0]*dq.orientation[0] + qn.dual[1]*dq.orientation[1] + qn.dual[2]*dq.orientation[2] + qn.dual[3]*dq.orientation[3]);
        for(i=0; i<4; i++)
        {
            for(j=0; j<=i; j++)
            {
                dJ0[i][j]=dJ0[j][i]=-qn.orientation[j]*dq.orientation[i]-qn.orientation[i]*dq.orientation[j]+(real)3.*q0dq0*qn.orientation[i]*qn.orientation[j];
                dJE[i][j]=dJE[j][i]=-qn.orientation[j]*dq.dual[i]-qn.orientation[i]*dq.dual[j]+(real)3.*q0dqe*qn.orientation[i]*qn.orientation[j]
                        -qn.dual[j]*dq.orientation[i]-qn.dual[i]*dq.orientation[j]+(real)3.*qedq0*qn.orientation[i]*qn.orientation[j]
                        +(real)3.*q0dq0*(qn.dual[j]*qn.orientation[i]+qn.dual[i]*qn.orientation[j]);
            }
            dJ0[i][i]-=q0dq0;
            dJE[i][i]-=q0dqe+qedq0;
        }
        dJ0*=Q0inv2; dJE*=Q0inv2;
        for(i=0; i<4; i++) for(j=0; j<4; j++) dJE[i][j]-=(real)2.*dJ0[i][j]*Q0QE2;
    }
    else
    {
        for(i=0; i<4; i++)
        {
            for(j=0; j<=i; j++) dJE[i][j]=dJE[j][i]=-qn.orientation[j]*dq.dual[i]-qn.orientation[i]*dq.dual[j]+(real)3.*q0dqe*qn.orientation[i]*qn.orientation[j];
            dJE[i][i]-=q0dqe;
        }
        dJE*=Q0inv2;
    }
}


template<typename real>
sofa::defaulttype::Vec<3,real> DualQuatCoord3<real>::rotate(const sofa::defaulttype::Vec<3,real>& v) const
{
    return sofa::defaulttype::Vec<3,real>(
            (real)((1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))*v[0] + (2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3])) * v[1] + (2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3])) * v[2]),
            (real)((2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3]))*v[0] + (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]))*v[1] + (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]))*v[2]),
            (real)((2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3]))*v[0] + (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]))*v[1] + (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]))*v[2])
            );
}

template<typename real>
sofa::defaulttype::Vec<3,real> DualQuatCoord3<real>::inverseRotate(const sofa::defaulttype::Vec<3,real>& v) const
{
    return sofa::defaulttype::Vec<3,real>(
            (real)((1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))*v[0] + (2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3])) * v[1] + (2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3])) * v[2]),
            (real)((2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3]))*v[0] + (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]))*v[1] + (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]))*v[2]),
            (real)((2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3]))*v[0] + (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]))*v[1] + (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]))*v[2])
            );
}

template<typename real>
void DualQuatCoord3<real>::invert()
{
    for ( unsigned int j = 0; j < 3; j++ )
    {
        orientation[j]=-orientation[j];
        dual[j]=-dual[j];
    }
}

template<typename real>
DualQuatCoord3<real> DualQuatCoord3<real>::inverse( )
{
    DualQuatCoord3<real> r;
    for ( unsigned int j = 0; j < 3; j++ )
    {
        r.orientation[j]=-orientation[j];
        r.dual[j]=-dual[j];
    }
    r.orientation[3]=orientation[3];
    r.dual[3]=dual[3];
    return r;
}


// compute the product with another frame on the right
template<typename real>
DualQuatCoord3<real> DualQuatCoord3<real>::multRight( const DualQuatCoord3<real>& c ) const
{
    DualQuatCoord3<real> r;
    //r.orientation = orientation * c.getOrientation();
    //r.dual = orientation * c.getDual() + dual * c.getOrientation();
    r.orientation[0] = orientation[3]*c.getOrientation()[0] - orientation[2]*c.getOrientation()[1] + orientation[1]*c.getOrientation()[2] + orientation[0]*c.getOrientation()[3];
    r.orientation[1] = orientation[2]*c.getOrientation()[0] + orientation[3]*c.getOrientation()[1] - orientation[0]*c.getOrientation()[2] + orientation[1]*c.getOrientation()[3];
    r.orientation[2] =-orientation[1]*c.getOrientation()[0] + orientation[0]*c.getOrientation()[1] + orientation[3]*c.getOrientation()[2] + orientation[2]*c.getOrientation()[3];
    r.orientation[3] =-orientation[0]*c.getOrientation()[0] - orientation[1]*c.getOrientation()[1] - orientation[2]*c.getOrientation()[2] + orientation[3]*c.getOrientation()[3];
    r.dual[0] = orientation[3]*c.getDual()[0] - orientation[2]*c.getDual()[1] + orientation[1]*c.getDual()[2] + orientation[0]*c.getDual()[3];
    r.dual[1] = orientation[2]*c.getDual()[0] + orientation[3]*c.getDual()[1] - orientation[0]*c.getDual()[2] + orientation[1]*c.getDual()[3];
    r.dual[2] =-orientation[1]*c.getDual()[0] + orientation[0]*c.getDual()[1] + orientation[3]*c.getDual()[2] + orientation[2]*c.getDual()[3];
    r.dual[3] =-orientation[0]*c.getDual()[0] - orientation[1]*c.getDual()[1] - orientation[2]*c.getDual()[2] + orientation[3]*c.getDual()[3];
    r.dual[0]+= dual[3]*c.getOrientation()[0] - dual[2]*c.getOrientation()[1] + dual[1]*c.getOrientation()[2] + dual[0]*c.getOrientation()[3];
    r.dual[1]+= dual[2]*c.getOrientation()[0] + dual[3]*c.getOrientation()[1] - dual[0]*c.getOrientation()[2] + dual[1]*c.getOrientation()[3];
    r.dual[2]+=-dual[1]*c.getOrientation()[0] + dual[0]*c.getOrientation()[1] + dual[3]*c.getOrientation()[2] + dual[2]*c.getOrientation()[3];
    r.dual[3]+=-dual[0]*c.getOrientation()[0] - dual[1]*c.getOrientation()[1] - dual[2]*c.getOrientation()[2] + dual[3]*c.getOrientation()[3];
    return r;
}

// get jacobian of the product with another frame f on the right : d(q*f) = J(q) f
template<typename real>
void DualQuatCoord3<real>::multRight_getJ( sofa::defaulttype::Mat<4,4,real>& J0,sofa::defaulttype::Mat<4,4,real>& JE)
{
    J0[0][0] = orientation[3];	J0[0][1] =-orientation[2];	J0[0][2] = orientation[1];	J0[0][3] = orientation[0];
    J0[1][0] = orientation[2];	J0[1][1] = orientation[3];	J0[1][2] =-orientation[0];	J0[1][3] = orientation[1];
    J0[2][0] =-orientation[1];	J0[2][1] = orientation[0];	J0[2][2] = orientation[3];	J0[2][3] = orientation[2];
    J0[3][0] =-orientation[0];	J0[3][1] =-orientation[1];	J0[3][2] =-orientation[2];	J0[3][3] = orientation[3];
    JE[0][0] = dual[3];		JE[0][1] =-dual[2];		JE[0][2] = dual[1];		JE[0][3] = dual[0];
    JE[1][0] = dual[2];		JE[1][1] = dual[3];		JE[1][2] =-dual[0];		JE[1][3] = dual[1];
    JE[2][0] =-dual[1];		JE[2][1] = dual[0];		JE[2][2] = dual[3];		JE[2][3] = dual[2];
    JE[3][0] =-dual[0];		JE[3][1] =-dual[1];		JE[3][2] =-dual[2];		JE[3][3] = dual[3];
}

// Apply a transformation with respect to itself
template<typename real>
DualQuatCoord3<real> DualQuatCoord3<real>::multLeft( const DualQuatCoord3<real>& c )
{
    DualQuatCoord3<real> r;
    r.orientation[0] = orientation[3]*c.getOrientation()[0] + orientation[2]*c.getOrientation()[1] - orientation[1]*c.getOrientation()[2] + orientation[0]*c.getOrientation()[3];
    r.orientation[1] =-orientation[2]*c.getOrientation()[0] + orientation[3]*c.getOrientation()[1] + orientation[0]*c.getOrientation()[2] + orientation[1]*c.getOrientation()[3];
    r.orientation[2] = orientation[1]*c.getOrientation()[0] - orientation[0]*c.getOrientation()[1] + orientation[3]*c.getOrientation()[2] + orientation[2]*c.getOrientation()[3];
    r.orientation[3] =-orientation[0]*c.getOrientation()[0] - orientation[1]*c.getOrientation()[1] - orientation[2]*c.getOrientation()[2] + orientation[3]*c.getOrientation()[3];
    r.dual[0] = orientation[3]*c.getDual()[0] + orientation[2]*c.getDual()[1] - orientation[1]*c.getDual()[2] + orientation[0]*c.getDual()[3];
    r.dual[1] =-orientation[2]*c.getDual()[0] + orientation[3]*c.getDual()[1] + orientation[0]*c.getDual()[2] + orientation[1]*c.getDual()[3];
    r.dual[2] = orientation[1]*c.getDual()[0] - orientation[0]*c.getDual()[1] + orientation[3]*c.getDual()[2] + orientation[2]*c.getDual()[3];
    r.dual[3] =-orientation[0]*c.getDual()[0] - orientation[1]*c.getDual()[1] - orientation[2]*c.getDual()[2] + orientation[3]*c.getDual()[3];
    r.dual[0]+= dual[3]*c.getOrientation()[0] + dual[2]*c.getOrientation()[1] - dual[1]*c.getOrientation()[2] + dual[0]*c.getOrientation()[3];
    r.dual[1]+=-dual[2]*c.getOrientation()[0] + dual[3]*c.getOrientation()[1] + dual[0]*c.getOrientation()[2] + dual[1]*c.getOrientation()[3];
    r.dual[2]+= dual[1]*c.getOrientation()[0] - dual[0]*c.getOrientation()[1] + dual[3]*c.getOrientation()[2] + dual[2]*c.getOrientation()[3];
    r.dual[3]+=-dual[0]*c.getOrientation()[0] - dual[1]*c.getOrientation()[1] - dual[2]*c.getOrientation()[2] + dual[3]*c.getOrientation()[3];
    return r;
}

// get jacobian of the product with another frame f on the left : d(f*q) = J(q) f
template<typename real>
void DualQuatCoord3<real>::multLeft_getJ( sofa::defaulttype::Mat<4,4,real>& J0,sofa::defaulttype::Mat<4,4,real>& JE)
{
    J0[0][0] = orientation[3];	J0[0][1] = orientation[2];	J0[0][2] =-orientation[1];	J0[0][3] = orientation[0];
    J0[1][0] =-orientation[2];	J0[1][1] = orientation[3];	J0[1][2] = orientation[0];	J0[1][3] = orientation[1];
    J0[2][0] = orientation[1];	J0[2][1] =-orientation[0];	J0[2][2] = orientation[3];	J0[2][3] = orientation[2];
    J0[3][0] =-orientation[0];	J0[3][1] =-orientation[1];	J0[3][2] =-orientation[2];	J0[3][3] = orientation[3];
    JE[0][0] = dual[3];		JE[0][1] = dual[2];		JE[0][2] =-dual[1];		JE[0][3] = dual[0];
    JE[1][0] =-dual[2];		JE[1][1] = dual[3];		JE[1][2] = dual[0];		JE[1][3] = dual[1];
    JE[2][0] = dual[1];		JE[2][1] =-dual[0];		JE[2][2] = dual[3];		JE[2][3] = dual[2];
    JE[3][0] =-dual[0];		JE[3][1] =-dual[1];		JE[3][2] =-dual[2];		JE[3][3] = dual[3];
}


// Write to the given matrix
template<typename  real>
template<typename real2>
void DualQuatCoord3<real>::toMatrix( sofa::defaulttype::Mat<3,4,real2>& m) const
{
    m[0][0] = (real2) (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]));
    m[0][1] = (real2) (2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3]));
    m[0][2] = (real2) (2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3]));

    m[1][0] = (real2) (2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3]));
    m[1][1] = (real2) (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]));
    m[1][2] = (real2) (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]));

    m[2][0] = (real2) (2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3]));
    m[2][1] = (real2) (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]));
    m[2][2] = (real2) (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]));

    sofa::defaulttype::Vec<3,real> p=getTranslation();
    m[0][3] =  (real2) p[0];
    m[1][3] =  (real2) p[1];
    m[2][3] =  (real2) p[2];
}



// Project a point from the child frame to the parent frame: P = R(q)p + t(q)
template<typename  real>
sofa::defaulttype::Vec<3,real> DualQuatCoord3<real>::pointToParent( const sofa::defaulttype::Vec<3,real>& p )
{
    sofa::defaulttype::Vec<3,real> p2;
    p2[0] = (real) ((1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))*p[0] + (2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3])) * p[1] + (2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3])) * p[2]);
    p2[1] = (real) ((2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3]))*p[0] + (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]))*p[1] + (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]))*p[2]);
    p2[2] = (real) ((2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3]))*p[0] + (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]))*p[1] + (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]))*p[2]);
    p2+=getTranslation();
    return p2;
}

// get jacobian of the transformation : dP = J(p,q) dq
template<typename  real>
void DualQuatCoord3<real>::pointToParent_getJ( sofa::defaulttype::Mat<3,4,real>& J0,sofa::defaulttype::Mat<3,4,real>& JE,const sofa::defaulttype::Vec<3,real>& p)
{
    J0.fill(0); JE.fill(0);
    J0[0][0] = (real)2.*(- dual[3] + orientation[0]*p[0] + orientation[1]*p[1] + orientation[2]*p[2]);	J0[0][1] =   (real)2.*(dual[2] - orientation[1]*p[0] + orientation[0]*p[1] + orientation[3]*p[2]);	J0[0][2] = (real)2.*(- dual[1] - orientation[2]*p[0] - orientation[3]*p[1] + orientation[0]*p[2]);	J0[0][3] = (real)2.*(  dual[0] + orientation[3]*p[0] - orientation[2]*p[1] + orientation[1]*p[2]);
    J0[1][0] = -J0[0][1];	J0[1][1] = J0[0][0];	J0[1][2] = J0[0][3];	J0[1][3] = -J0[0][2];
    J0[2][0] = -J0[0][2];	J0[2][1] = -J0[0][3];	J0[2][2] = J0[0][0];	J0[2][3] = J0[0][1];
    JE[0][0] = (real)2.* orientation[3];		JE[0][1] = -(real)2.*orientation[2];		JE[0][2] =  (real)2.*orientation[1];		JE[0][3] = -(real)2.*orientation[0];
    JE[1][0] = -JE[0][1];	JE[1][1] = JE[0][0];	JE[1][2] = JE[0][3];	JE[1][3] = -JE[0][2];
    JE[2][0] = -JE[0][2];	JE[2][1] = -JE[0][3];	JE[2][2] = JE[0][0];	JE[2][3] = JE[0][1];
}

// get transformed position change: dP = J(p,q) dq
template<typename  real>
sofa::defaulttype::Vec<3,real> DualQuatCoord3<real>::pointToParent_applyJ( const DualQuatCoord3<real>& dq ,const sofa::defaulttype::Vec<3,real>& p)
{
    sofa::defaulttype::Mat<3,4,real> J0,JE;
    pointToParent_getJ(J0,JE,p);
    sofa::defaulttype::Vec<3,real> r=J0*dq.orientation+JE*dq.dual;
    return r;
}

// get quaternion change: dq = JT(p,q) dP
template<typename  real>
DualQuatCoord3<real> DualQuatCoord3<real>::pointToParent_applyJT( const sofa::defaulttype::Vec<3,real>& dP ,const sofa::defaulttype::Vec<3,real>& p)
{
    sofa::defaulttype::Mat<3,4,real> J0,JE;
    pointToParent_getJ(J0,JE,p);
    DualQuatCoord3<real> r;
    r.orientation=J0.transposed()*dP;
    r.dual=JE.transposed()*dP;
    return r;
}


// get rigid transformation change: d(R,t) = H(q) dq
template<typename  real>
sofa::defaulttype::Mat<3,4,real> DualQuatCoord3<real>::rigid_applyH( const DualQuatCoord3<real>& dq )
{
    sofa::defaulttype::Mat<3,4,real> dR;
    dR[0][0]=(real)2.*(-2*orientation[1]*dq.orientation[1]-2*orientation[2]*dq.orientation[2]);
    dR[0][1]=(real)2.*(orientation[1]*dq.orientation[0]+orientation[0]*dq.orientation[1]-orientation[3]*dq.orientation[2]-orientation[2]*dq.orientation[3]);
    dR[0][2]=(real)2.*(orientation[2]*dq.orientation[0]+orientation[3]*dq.orientation[1]+orientation[0]*dq.orientation[2]+orientation[1]*dq.orientation[3]);
    dR[0][3]=(real)2.*(-dual[3]*dq.orientation[0]+dual[2]*dq.orientation[1]-dual[1]*dq.orientation[2]+dual[0]*dq.orientation[3]+orientation[3]*dq.dual[0]-orientation[2]*dq.dual[1]+orientation[1]*dq.dual[2]-orientation[0]*dq.dual[3]);
    dR[1][0]=(real)2.*(orientation[1]*dq.orientation[0]+orientation[0]*dq.orientation[1]+orientation[3]*dq.orientation[2]+orientation[2]*dq.orientation[3]);
    dR[1][1]=(real)2.*(-2*orientation[0]*dq.orientation[0]-2*orientation[2]*dq.orientation[2]);
    dR[1][2]=(real)2.*(-orientation[3]*dq.orientation[0]+orientation[2]*dq.orientation[1]+orientation[1]*dq.orientation[2]-orientation[0]*dq.orientation[3]);
    dR[1][3]=(real)2.*(-dual[2]*dq.orientation[0]-dual[3]*dq.orientation[1]+dual[0]*dq.orientation[2]+dual[1]*dq.orientation[3]+orientation[2]*dq.dual[0]+orientation[3]*dq.dual[1]-orientation[0]*dq.dual[2]-orientation[1]*dq.dual[3]);
    dR[2][0]=(real)2.*(orientation[2]*dq.orientation[0]-orientation[3]*dq.orientation[1]+orientation[0]*dq.orientation[2]-orientation[1]*dq.orientation[3]);
    dR[2][1]=(real)2.*(orientation[3]*dq.orientation[0]+orientation[2]*dq.orientation[1]+orientation[1]*dq.orientation[2]+orientation[0]*dq.orientation[3]);
    dR[2][2]=(real)2.*(-2*orientation[0]*dq.orientation[0]-2*orientation[1]*dq.orientation[1]);
    dR[2][3]=(real)2.*(dual[1]*dq.orientation[0]-dual[0]*dq.orientation[1]-dual[3]*dq.orientation[2]+dual[2]*dq.orientation[3]-orientation[1]*dq.dual[0]+orientation[0]*dq.dual[1]+orientation[3]*dq.dual[2]-orientation[2]*dq.dual[3]);
    return dR;
}
// get rotation change: dR = H(q) dq
template<typename  real>
sofa::defaulttype::Mat<3,3,real> DualQuatCoord3<real>::rotation_applyH( const DualQuatCoord3<real>& dq )
{
    sofa::defaulttype::Mat<3,3,real> dR;
    dR[0][0]=(real)2.*(-2*orientation[1]*dq.orientation[1]-2*orientation[2]*dq.orientation[2]);
    dR[0][1]=(real)2.*(orientation[1]*dq.orientation[0]+orientation[0]*dq.orientation[1]-orientation[3]*dq.orientation[2]-orientation[2]*dq.orientation[3]);
    dR[0][2]=(real)2.*(orientation[2]*dq.orientation[0]+orientation[3]*dq.orientation[1]+orientation[0]*dq.orientation[2]+orientation[1]*dq.orientation[3]);
    dR[1][0]=(real)2.*(orientation[1]*dq.orientation[0]+orientation[0]*dq.orientation[1]+orientation[3]*dq.orientation[2]+orientation[2]*dq.orientation[3]);
    dR[1][1]=(real)2.*(-2*orientation[0]*dq.orientation[0]-2*orientation[2]*dq.orientation[2]);
    dR[1][2]=(real)2.*(-orientation[3]*dq.orientation[0]+orientation[2]*dq.orientation[1]+orientation[1]*dq.orientation[2]-orientation[0]*dq.orientation[3]);
    dR[2][0]=(real)2.*(orientation[2]*dq.orientation[0]-orientation[3]*dq.orientation[1]+orientation[0]*dq.orientation[2]-orientation[1]*dq.orientation[3]);
    dR[2][1]=(real)2.*(orientation[3]*dq.orientation[0]+orientation[2]*dq.orientation[1]+orientation[1]*dq.orientation[2]+orientation[0]*dq.orientation[3]);
    dR[2][2]=(real)2.*(-2*orientation[0]*dq.orientation[0]-2*orientation[1]*dq.orientation[1]);
    return dR;
}

// get quaternion change: dq = H^T(q) d(R,t)
template<typename  real>
DualQuatCoord3<real> DualQuatCoord3<real>::rigid_applyHT( const sofa::defaulttype::Mat<3,4,real>& dR )
{
    DualQuatCoord3<real> r;
    r.orientation[0]=(real)2.*(orientation[1]*dR[0][1]+orientation[2]*dR[0][2]-dual[3]*dR[0][3]+orientation[1]*dR[1][0]-2*orientation[0]*dR[1][1]-orientation[3]*dR[1][2]-dual[2]*dR[1][3]+orientation[2]*dR[2][0]+orientation[3]*dR[2][1]-2*orientation[0]*dR[2][2]+dual[1]*dR[2][3]);
    r.orientation[1]=(real)2.*(-2*orientation[1]*dR[0][0]+orientation[0]*dR[0][1]+orientation[3]*dR[0][2]+dual[2]*dR[0][3]+orientation[0]*dR[1][0]+orientation[2]*dR[1][2]-dual[3]*dR[1][3]-orientation[3]*dR[2][0]+orientation[2]*dR[2][1]-2*orientation[1]*dR[2][2]-dual[0]*dR[2][3]);
    r.orientation[2]=(real)2.*(-2*orientation[2]*dR[0][0]-orientation[3]*dR[0][1]+orientation[0]*dR[0][2]-dual[1]*dR[0][3]+orientation[3]*dR[1][0]-2*orientation[2]*dR[1][1]+orientation[1]*dR[1][2]+dual[0]*dR[1][3]+orientation[0]*dR[2][0]+orientation[1]*dR[2][1]-dual[3]*dR[2][3]);
    r.orientation[3]=(real)2.*(-orientation[2]*dR[0][1]+orientation[1]*dR[0][2]+dual[0]*dR[0][3]+orientation[2]*dR[1][0]-orientation[0]*dR[1][2]+dual[1]*dR[1][3]-orientation[1]*dR[2][0]+orientation[0]*dR[2][1]+dual[2]*dR[2][3]);
    r.dual[0]=(real)2.*(orientation[3]*dR[0][3]+orientation[2]*dR[1][3]-orientation[1]*dR[2][3]);
    r.dual[1]=(real)2.*(-orientation[2]*dR[0][3]+orientation[3]*dR[1][3]+orientation[0]*dR[2][3]);
    r.dual[2]=(real)2.*(orientation[1]*dR[0][3]-orientation[0]*dR[1][3]+orientation[3]*dR[2][3]);
    r.dual[3]=(real)2.*(-orientation[0]*dR[0][3]-orientation[1]*dR[1][3]-orientation[2]*dR[2][3]);
    return r;
}
// get quaternion change: dq = H^T(q) dR
template<typename  real>
DualQuatCoord3<real> DualQuatCoord3<real>::rotation_applyHT( const sofa::defaulttype::Mat<3,3,real>& dR )
{
    DualQuatCoord3<real> r;
    r.orientation[0]=(real)2.*(orientation[1]*dR[0][1]+orientation[2]*dR[0][2]+orientation[1]*dR[1][0]-2*orientation[0]*dR[1][1]-orientation[3]*dR[1][2]+orientation[2]*dR[2][0]+orientation[3]*dR[2][1]-2*orientation[0]*dR[2][2]);
    r.orientation[1]=(real)2.*(-2*orientation[1]*dR[0][0]+orientation[0]*dR[0][1]+orientation[3]*dR[0][2]+orientation[0]*dR[1][0]+orientation[2]*dR[1][2]-orientation[3]*dR[2][0]+orientation[2]*dR[2][1]-2*orientation[1]*dR[2][2]);
    r.orientation[2]=(real)2.*(-2*orientation[2]*dR[0][0]-orientation[3]*dR[0][1]+orientation[0]*dR[0][2]+orientation[3]*dR[1][0]-2*orientation[2]*dR[1][1]+orientation[1]*dR[1][2]+orientation[0]*dR[2][0]+orientation[1]*dR[2][1]);
    r.orientation[3]=(real)2.*(-orientation[2]*dR[0][1]+orientation[1]*dR[0][2]+orientation[2]*dR[1][0]-orientation[0]*dR[1][2]-orientation[1]*dR[2][0]+orientation[0]*dR[2][1]);
    r.dual[0]=r.dual[1]=r.dual[2]=r.dual[3]=(real)0.;
    return r;
}

// get Jacobian change: dJ = H(p) dq
template<typename  real>
sofa::defaulttype::Mat<3,8,real> DualQuatCoord3<real>::pointToParent_applyH( const DualQuatCoord3<real>& dq ,const sofa::defaulttype::Vec<3,real>& p)
{
    sofa::defaulttype::Mat<3,8,real> dJ;
    dJ.fill(0);
    dJ[0][0] = (real)2.*(- dq.dual[3] + dq.orientation[0]*p[0] + dq.orientation[1]*p[1] + dq.orientation[2]*p[2]);	dJ[0][1] =   (real)2.*(dq.dual[2] - dq.orientation[1]*p[0] + dq.orientation[0]*p[1] + dq.orientation[3]*p[2]);	dJ[0][2] = (real)2.*(- dq.dual[1] - dq.orientation[2]*p[0] - dq.orientation[3]*p[1] + dq.orientation[0]*p[2]);	dJ[0][3] = (real)2.*(  dq.dual[0] + dq.orientation[3]*p[0] - dq.orientation[2]*p[1] + dq.orientation[1]*p[2]);
    dJ[1][0] = -dJ[0][1];	dJ[1][1] = dJ[0][0];	dJ[1][2] = dJ[0][3];	dJ[1][3] = -dJ[0][2];
    dJ[2][0] = -dJ[0][2];	dJ[2][1] = -dJ[0][3];	dJ[2][2] = dJ[0][0];	dJ[2][3] = dJ[0][1];
    dJ[0][4] =  (real)2.*dq.orientation[3];		dJ[0][5] = -(real)2.*dq.orientation[2];		dJ[0][6] =  (real)2.*dq.orientation[1];		dJ[0][7] = -(real)2.*dq.orientation[0];
    dJ[1][4] = -dJ[0][5];	dJ[1][5] = dJ[0][4];	dJ[1][6] = dJ[0][7];	dJ[1][7] = -dJ[0][6];
    dJ[2][4] = -dJ[0][6];	dJ[2][5] = -dJ[0][7];	dJ[2][6] = dJ[0][4];	dJ[2][7] = dJ[0][5];
    return dJ;
}

// get quaternion change: dq = H^T(p) dJ
template<typename  real>
DualQuatCoord3<real> DualQuatCoord3<real>::pointToParent_applyHT( const sofa::defaulttype::Mat<3,8,real>& dJ ,const sofa::defaulttype::Vec<3,real>& p)
{
    DualQuatCoord3<real> r;
    r.orientation[0]=(real)2.*(p[0]*dJ[0][0]+p[1]*dJ[0][1]+p[2]*dJ[0][2]-dJ[0][7]-p[1]*dJ[1][0]+p[0]*dJ[1][1]-p[2]*dJ[1][3]-dJ[1][6]-p[2]*dJ[2][0]+p[0]*dJ[2][2]+p[1]*dJ[2][3]+dJ[2][5]);
    r.orientation[1]=(real)2.*(p[1]*dJ[0][0]-p[0]*dJ[0][1]+p[2]*dJ[0][3]+dJ[0][6]+p[0]*dJ[1][0]+p[1]*dJ[1][1]+p[2]*dJ[1][2]-dJ[1][7]-p[2]*dJ[2][1]+p[1]*dJ[2][2]-p[0]*dJ[2][3]-dJ[2][4]);
    r.orientation[2]=(real)2.*(p[2]*dJ[0][0]-p[0]*dJ[0][2]-p[1]*dJ[0][3]-dJ[0][5]+p[2]*dJ[1][1]-p[1]*dJ[1][2]+p[0]*dJ[1][3]+dJ[1][4]+p[0]*dJ[2][0]+p[1]*dJ[2][1]+p[2]*dJ[2][2]-dJ[2][7]);
    r.orientation[3]=(real)2.*(p[2]*dJ[0][1]-p[1]*dJ[0][2]+p[0]*dJ[0][3]+dJ[0][4]-p[2]*dJ[1][0]+p[0]*dJ[1][2]+p[1]*dJ[1][3]+dJ[1][5]+p[1]*dJ[2][0]-p[0]*dJ[2][1]+p[2]*dJ[2][3]+dJ[2][6]);
    r.dual[0]=(real)2.*(dJ[0][3]+dJ[1][2]-dJ[2][1]);
    r.dual[1]=(real)2.*(-dJ[0][2]+dJ[1][3]+dJ[2][0]);
    r.dual[2]=(real)2.*(dJ[0][1]-dJ[1][0]+dJ[2][3]);
    r.dual[3]=(real)2.*(-dJ[0][0]-dJ[1][1]-dJ[2][2]);
    return r;
}


// Project a point from the parent frame to the child frame
template<typename  real>
sofa::defaulttype::Vec<3,real> DualQuatCoord3<real>::pointToChild( const sofa::defaulttype::Vec<3,real>& v )
{
    sofa::defaulttype::Vec<3,real> p,v2=v-this->getTranslation();
    p[0]=(real)((1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))*v2[0] + (2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3])) * v2[1] + (2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3])) * v2[2]);
    p[1]=(real)((2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3]))*v2[0] + (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]))*v2[1] + (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]))*v2[2]);
    p[2]=(real)((2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3]))*v2[0] + (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]))*v2[1] + (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]))*v2[2]);
    return p;
}

// compute the projection of a vector from the parent frame to the child
template<typename  real>
sofa::defaulttype::Vec<3,real> DualQuatCoord3<real>::vectorToChild( const sofa::defaulttype::Vec<3,real>& v )
{
    //return orientation.inverseRotate(v);
    sofa::defaulttype::Vec<3,real> p;
    p[0]=(real)((1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))*v[0] + (2.0f * (orientation[0] * orientation[1] + orientation[2] * orientation[3])) * v[1] + (2.0f * (orientation[2] * orientation[0] - orientation[1] * orientation[3])) * v[2]);
    p[1]=(real)((2.0f * (orientation[0] * orientation[1] - orientation[2] * orientation[3]))*v[0] + (1.0f - 2.0f * (orientation[2] * orientation[2] + orientation[0] * orientation[0]))*v[1] + (2.0f * (orientation[1] * orientation[2] + orientation[0] * orientation[3]))*v[2]);
    p[2]=(real)((2.0f * (orientation[2] * orientation[0] + orientation[1] * orientation[3]))*v[0] + (2.0f * (orientation[1] * orientation[2] - orientation[0] * orientation[3]))*v[1] + (1.0f - 2.0f * (orientation[1] * orientation[1] + orientation[0] * orientation[0]))*v[2]);
    return p;
}





} // namespace helper

} // namespace sofa

#endif
