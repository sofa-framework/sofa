/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#ifndef FLEXIBLE_PolynomialBasis_H
#define FLEXIBLE_PolynomialBasis_H

#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/MatSym.h>
#include <set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace sofa
{
namespace defaulttype
{


template< int _N, typename _Real, int _dim, int _order>
class Basis
{
public:
    typedef _Real Real;
    static const unsigned int N = _N;
    static const unsigned int dim = _dim;
    static const unsigned int order = _order;
    typedef Vec<N,Real> T;
    typedef Vec<dim,T> Gradient;
    typedef MatSym<dim,T> Hessian;
};

/**
    Generic class to implement polynomial decomposition.
    A function f is approximated as: f(p) = F.p~ where :
        - f is a function of type T (vector)
        - p is a coordinate in @param dim dimensions (e.g. [x,y,z] in 3d)
        - p~ is the complete basis of order @param order in @param dim dimensions (e.g. [1,x,y,z,x^2,xy,xz,y^2,yz,z^2] at order 2, in 3d)
        - F are the coefficient values, of type @param T, implemented here
  */

template< int _N, typename _Real, int _dim, int _order>
class PolynomialBasis: public Basis<_N,_Real,_dim,_order>
{
public:
    typedef Basis<_N,_Real,_dim,_order> inherit;
    typedef typename inherit::T T;
    typedef typename inherit::Real Real;
    enum { N = inherit::N };
    static const unsigned int dim = _dim;      ///< number of spatial dimensions
    static const unsigned int order = _order;  ///< polynomial order
    static const unsigned int bdim =  dim==1? order+1 : ( dim==2? (order+1)*(order+2)/2 : dim==3? (order+1)*(order+2)*(order+3)/6 : 0); ///< size of complete basis
    enum { total_size = bdim * N };  // number of entries

    typedef Vec<bdim,T> CoeffVec;
    typedef Vec<total_size,Real> TotalVec;
    typedef typename inherit::Gradient Gradient;
    typedef typename inherit::Hessian Hessian;

    CoeffVec v;

    PolynomialBasis() { v.clear(); }
    PolynomialBasis( const PolynomialBasis& d):v(d.v) {}
    PolynomialBasis( const TotalVec& d) {getVec()=d;}
    void clear() { v.clear(); }

    void fill(Real r) { for(unsigned int i=0; i<bdim; i++) v[i].fill(r); }

    /// seen as a vector
    Real* ptr() { return v[0].ptr(); }
    const Real* ptr() const { return v[0].ptr(); }

    TotalVec& getVec() { return *reinterpret_cast<TotalVec*>(ptr()); }
    const TotalVec& getVec() const  { return *reinterpret_cast<const TotalVec*>(ptr()); }

    /// val of f , order 0
    T& getVal() { return v[0]; }
    const T& getVal() const { return v[0]; }

    /// gradient, order 2
    Gradient& getGradient() { return *reinterpret_cast<Gradient*>(&v[1]); }
    const Gradient& getGradient() const { return *reinterpret_cast<const Gradient*>(&v[1]); }

    /// Hessian, order 3
    Hessian& getHessian() { return *reinterpret_cast<Hessian*>(&v[1+dim]); }
    const Hessian& getHessian() const { return *reinterpret_cast<const Hessian*>(&v[1+dim]); }
};




/** Helper functions for polynomial basis
  * used in high order quadrature methods (elastons)
  **/


template<typename real>
inline void getCompleteBasis(helper::vector<real>& basis, const Vec<3,real>& p,const unsigned int order)
{
    typedef Vec<3,real> Coord;

    unsigned int j,k,dim=(order+1)*(order+2)*(order+3)/6;

    basis.resize(dim);  for (j=0; j<dim; j++) basis[j]=0;

    unsigned int count=0;
    // order 0
    basis[count]=1;
    count++;
    if (count==dim) return;
    // order 1
    for (j=0; j<3; j++)
    {
        basis[count]=p[j];
        count++;
    }
    if (count==dim) return;
    // order 2
    for (j=0; j<3; j++) for (k=j; k<3; k++)
    {
        basis[count]=p[j]*p[k];
        count++;
    }
    if (count==dim) return;
    // order 3
    Coord p2;    for (j=0; j<3; j++) p2[j]=p[j]*p[j];
    basis[count]=p[0]*p[1]*p[2];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++)
    {
        basis[count]=p2[j]*p[k];
        count++;
    }
    if (count==dim) return;
    // order 4
    Coord p3;    for (j=0; j<3; j++) p3[j]=p2[j]*p[j];
    for (j=0; j<3; j++) for (k=j; k<3; k++)
    {
        basis[count]=p2[j]*p2[k];
        count++;
    }
    basis[count]=p2[0]*p[1]*p[2];
    count++;
    basis[count]=p[0]*p2[1]*p[2];
    count++;
    basis[count]=p[0]*p[1]*p2[2];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++) if (j!=k)
    {
        basis[count]=p3[j]*p[k];
        count++;
    }
    if (count==dim) return;

    return; // order>4 not implemented...
}



/** Returns the integral of the Complete Basis vector inside a cuboid of lenghts l, centered on p
  **/

template<typename real>
inline void getCompleteBasisIntegralInCube(helper::vector<real>& basis, const Vec<3,real>& p, const Vec<3,real>& l, const unsigned int order)
{
    typedef Vec<3,real> Coord;

    unsigned int j,dim=(order+1)*(order+2)*(order+3)/6;

    basis.resize(dim);  for (j=0; j<dim; j++) basis[j]=0;

    unsigned int count=0;
    real v=l[0]*l[1]*l[2];

    // order 0
    basis[count]=v; count++;
    if (count==dim) return;

    // order 1
    basis[count]=p[0]*v; count++;
    basis[count]=p[1]*v; count++;
    basis[count]=p[2]*v; count++;
    if (count==dim) return;

    // order 2
    Coord l2;  for (j=0; j<3; j++) l2[j]=l[j]*l[j];
    Coord p2;  for (j=0; j<3; j++) p2[j]=p[j]*p[j];
    real inv12 = 1./12.;
    basis[count]=( inv12*l2[0] + p2[0] )*v; count++;
    basis[count]= p[0]*p[1]*v; count++;
    basis[count]= p[0]*p[2]*v; count++;
    basis[count]= ( inv12*l2[1] + p2[1] )*v; count++;
    basis[count]= p[1]*p[2]*v; count++;
    basis[count]= ( inv12*l2[2] + p2[2] )*v; count++;
    if (count==dim) return;

    // order 3
    Coord p3;    for (j=0; j<3; j++) p3[j]=p2[j]*p[j];
    basis[count]= p[0]*p[1]*p[2]*v; count++;
    basis[count]= ( 0.25*l2[0]*p[0] + p3[0] )*v; count++;
    basis[count]= ( inv12*l2[0]*p[1] + p2[0]*p[1] )*v; count++;
    basis[count]= ( inv12*l2[0]*p[2] + p2[0]*p[2] )*v; count++;
    basis[count]= ( inv12*p[0]*l2[1] + p[0]*p2[1] )*v; count++;
    basis[count]= ( 0.25*l2[1]*p[1] + p3[1] )*v; count++;
    basis[count]= ( inv12*l2[1]*p[2] + p2[1]*p[2] )*v; count++;
    basis[count]= ( inv12*p[0]*l2[2] + p[0]*p2[2] )*v; count++;
    basis[count]= ( inv12*p[1]*l2[2] + p[1]*p2[2] )*v; count++;
    basis[count]= ( 0.25*l2[2]*p[2] + p3[2] )*v; count++;
    if (count==dim) return;

    // order 4
    real inv144 = 1./144.;
    basis[count]= ( 0.0125*l2[0]*l2[0] + 0.5*l2[0]*p2[0] + p2[0]*p2[0] )*v; count++;
    basis[count]= ( inv144*l2[0]*l2[1] + inv12*l2[0]*p2[1] + inv12*p2[0]*l2[1] + p2[0]*p2[1] )*v; count++;
    basis[count]= ( inv144*l2[0]*l2[2] + inv12*l2[0]*p2[2] + inv12*p2[0]*l2[2] + p2[0]*p2[2] )*v; count++;
    basis[count]= ( 0.0125*l2[1]*l2[1] + 0.5*l2[1]*p2[1] + p2[1]*p2[1] )*v; count++;
    basis[count]= ( inv144*l2[1]*l2[2] + inv12*l2[1]*p2[2] + inv12*p2[1]*l2[2] + p2[1]*p2[2] )*v; count++;
    basis[count]= ( 0.0125*l2[2]*l2[2] + 0.5*l2[2]*p2[2] + p2[2]*p2[2] )*v; count++;
    basis[count]= ( inv12*l2[0]*p[1]*p[2] + p2[0]*p[1]*p[2] )*v; count++;
    basis[count]= ( inv12*p[0]*l2[1]*p[2] + p[0]*p2[1]*p[2] )*v; count++;
    basis[count]= ( inv12*p[0]*p[1]*l2[2] + p[0]*p[1]*p2[2] )*v; count++;
    basis[count]= ( 0.25*l2[0]*p[0]*p[1] + p3[0]*p[1] )*v; count++;
    basis[count]= ( 0.25*l2[0]*p[0]*p[2] + p3[0]*p[2] )*v; count++;
    basis[count]= ( 0.25*p[0]*l2[1]*p[1] + p[0]*p3[1] )*v; count++;
    basis[count]= ( 0.25*l2[1]*p[1]*p[2] + p3[1]*p[2] )*v; count++;
    basis[count]= ( 0.25*p[0]*l2[2]*p[2] + p[0]*p3[2] )*v; count++;
    basis[count]= ( 0.25*p[1]*l2[2]*p[2] + p[1]*p3[2] )*v; count++;
}


template<typename real>
Eigen::Matrix<real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> getCompleteBasis_TranslationMatrix(const Vec<3,real>& t,const unsigned int order)
{
//    typedef Vec<3,real> Coord;

    unsigned int j,dim=(order+1)*(order+2)*(order+3)/6;
    Eigen::Matrix<real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> A(dim,dim);
    A.setIdentity();

    // order 0
    unsigned int count=1;
    if (count==dim) return A;

    // order 1
    helper::vector<real> T; getCompleteBasis(T,t,order);
    for (j=1;j<dim;j++) A(j,0)=T[j]; // fill first column
    count=4;
    if (count==dim) return A;

    // order 2
    A(count,1)=2*T[1]; count++;
    A(count,1)=T[2]; A(count,2)=T[1]; count++;
    A(count,1)=T[3];    A(count,3)=T[1]; count++;
    A(count,2)=2*T[2]; count++;
    A(count,2)=T[3];    A(count,3)=T[2]; count++;
    A(count,3)=2*T[3]; count++;
    if (count==dim) return A;

    // order 3
    A(count,1)=T[8]; A(count,2)=T[6]; A(count,3)=T[5]; A(count,5)=T[3]; A(count,6)=T[2]; A(count,8)=T[1];  count++;
    A(count,1)=3*T[4]; A(count,4)=3*T[1]; count++;
    A(count,1)=2*T[5]; A(count,2)=T[4]; A(count,4)=T[2]; A(count,5)=2*T[1]; count++;
    A(count,1)=2*T[6]; A(count,3)=T[4]; A(count,4)=T[3]; A(count,6)=2*T[1]; count++;
    A(count,1)=T[7]; A(count,2)=2*T[5]; A(count,5)=2*T[2]; A(count,7)=T[1]; count++;
    A(count,2)=3*T[7]; A(count,7)=3*T[2]; count++;
    A(count,2)=2*T[8]; A(count,3)=T[7]; A(count,7)=T[3]; A(count,8)=2*T[2]; count++;
    A(count,1)=T[9]; A(count,3)=2*T[6]; A(count,6)=2*T[3]; A(count,9)=T[1]; count++;
    A(count,2)=T[9]; A(count,3)=2*T[8]; A(count,8)=2*T[3]; A(count,9)=T[2];  count++;
    A(count,3)=3*T[9]; A(count,9)=3*T[3];  count++;
    if (count==dim) return A;

    // order 4
    A(count,1)=4*T[11]; A(count,4)=6*T[4]; A(count,11)=4*T[1];  count++;
    A(count,1)=2*T[14]; A(count,2)=2*T[12]; A(count,4)=T[7]; A(count,5)=4*T[5]; A(count,7)=T[4]; A(count,12)=2*T[2]; A(count,14)=2*T[1];  count++;
    A(count,1)=2*T[17]; A(count,3)=2*T[13]; A(count,4)=T[9]; A(count,6)=4*T[6]; A(count,9)=T[4]; A(count,13)=2*T[3]; A(count,17)=2*T[1];  count++;
    A(count,2)=4*T[15]; A(count,7)=6*T[7];  A(count,15)=4*T[2];  count++;
    A(count,2)=2*T[18]; A(count,3)=2*T[16]; A(count,7)=T[9]; A(count,8)=4*T[8]; A(count,9)=T[7]; A(count,16)=2*T[3]; A(count,18)=2*T[2];  count++;
    A(count,3)=4*T[19]; A(count,9)=6*T[9];  A(count,19)=4*T[3];  count++;
    A(count,1)=2*T[10]; A(count,2)=T[13]; A(count,3)=T[12]; A(count,4)=T[8]; A(count,5)=2*T[6]; A(count,6)=2*T[5]; A(count,8)=T[4]; A(count,10)=2*T[1]; A(count,12)=T[3]; A(count,13)=T[2];  count++;
    A(count,1)=T[16]; A(count,2)=2*T[10]; A(count,3)=T[14]; A(count,5)=2*T[8]; A(count,6)=T[7]; A(count,7)=T[6]; A(count,8)=2*T[5]; A(count,10)=2*T[2]; A(count,14)=T[3]; A(count,16)=T[1];  count++;
    A(count,1)=T[18]; A(count,2)=T[17]; A(count,3)=2*T[10]; A(count,5)=T[9]; A(count,6)=2*T[8]; A(count,8)=2*T[6]; A(count,9)=T[5]; A(count,10)=2*T[3]; A(count,17)=T[2]; A(count,18)=T[1];  count++;
    A(count,1)=3*T[12]; A(count,2)=T[11]; A(count,4)=3*T[5]; A(count,5)=3*T[4]; A(count,11)=T[2]; A(count,12)=3*T[1];  count++;
    A(count,1)=3*T[13]; A(count,3)=T[11]; A(count,4)=3*T[6]; A(count,6)=3*T[4]; A(count,11)=T[3]; A(count,13)=3*T[1];  count++;
    A(count,1)=T[15]; A(count,2)=3*T[14]; A(count,5)=3*T[7]; A(count,7)=3*T[5]; A(count,14)=3*T[2]; A(count,15)=T[1];  count++;
    A(count,2)=3*T[16]; A(count,3)=T[15]; A(count,7)=3*T[8]; A(count,8)=3*T[7]; A(count,15)=T[3]; A(count,16)=3*T[2];  count++;
    A(count,1)=T[19]; A(count,3)=3*T[17]; A(count,6)=3*T[9]; A(count,9)=3*T[6]; A(count,17)=3*T[3]; A(count,19)=T[1];  count++;
    A(count,2)=T[19]; A(count,3)=3*T[18]; A(count,8)=3*T[9]; A(count,9)=3*T[8]; A(count,18)=3*T[3]; A(count,19)=T[2];  count++;

    return A; // order>4 not implemented...
}



template<typename real>
inline void getCompleteBasisGradient(helper::vector<Vec<3,real> >& basisDeriv, const Vec<3,real>& p,const unsigned int order)
{
    typedef Vec<3,real> Coord;

    unsigned int j,k,dim=(order+1)*(order+2)*(order+3)/6;

    basisDeriv.resize(dim);  for (j=0; j<dim; j++) basisDeriv[j].fill(0);

    Coord p2;  for (j=0; j<3; j++) p2[j]=p[j]*p[j];
    Coord p3;  for (j=0; j<3; j++) p3[j]=p2[j]*p[j];

    unsigned int count=0;
    // order 0
    count++;
    if (count==dim) return;
    // order 1
    for (j=0; j<3; j++)
    {
        basisDeriv[count][j]=1;
        count++;
    }
    if (count==dim) return;
    // order 2
    for (j=0; j<3; j++) for (k=j; k<3; k++)
    {
        basisDeriv[count][k]+=p[j];
        basisDeriv[count][j]+=p[k];
        count++;
    }
    if (count==dim) return;
    // order 3
    basisDeriv[count][0]=p[1]*p[2];
    basisDeriv[count][1]=p[0]*p[2];
    basisDeriv[count][2]=p[0]*p[1];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++)
    {
        basisDeriv[count][k]+=p2[j];
        basisDeriv[count][j]+=2*p[j]*p[k];
        count++;
    }
    if (count==dim) return;
    // order 4
    for (j=0; j<3; j++) for (k=j; k<3; k++)
    {
        basisDeriv[count][k]=2*p2[j]*p[k];
        basisDeriv[count][j]=2*p[j]*p2[k];
        count++;
    }
    basisDeriv[count][0]=2*p[0]*p[1]*p[2];
    basisDeriv[count][1]=p2[0]*p[2];
    basisDeriv[count][2]=p2[0]*p[1];
    count++;
    basisDeriv[count][0]=p2[1]*p[2];
    basisDeriv[count][1]=2*p[0]*p[1]*p[2];
    basisDeriv[count][2]=p[0]*p2[1];
    count++;
    basisDeriv[count][0]=p[1]*p2[2];
    basisDeriv[count][1]=p[0]*p2[2];
    basisDeriv[count][2]=2*p[0]*p[1]*p[2];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++) if (j!=k)
    {
        basisDeriv[count][k]=p3[j];
        basisDeriv[count][j]=3*p2[j]*p[k];
        count++;
    }
    if (count==dim) return;

    return; // order>4 not implemented...
}


template<typename real>
inline void getCompleteBasisHessian(helper::vector<MatSym<3,real> >& basisDeriv, const Vec<3,real>& p,const unsigned int order)
{
    typedef Vec<3,real> Coord;

    unsigned int j,k,dim=(order+1)*(order+2)*(order+3)/6;

    basisDeriv.resize(dim);    for (k=0; k<dim; k++) basisDeriv[k].fill(0);

    unsigned int count=0;
    // order 0
    count++;
    if (count==dim) return;
    // order 1
    count+=3;
    if (count==dim) return;
    // order 2
    for (j=0; j<3; j++) for (k=j; k<3; k++)
    {
        basisDeriv[count](k,j)+=1;
        if(k==j)  basisDeriv[count](k,j)+=1;
        count++;
    }
    if (count==dim) return;
    // order 3
    basisDeriv[count](0,1)=p[2];
    basisDeriv[count](0,2)=p[1];
    basisDeriv[count](1,2)=p[0];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++)
    {
        basisDeriv[count](k,j)+=2*p[j];
        if(k==j) basisDeriv[count](k,j)+=2*p[j];
        count++;
    }
    if (count==dim) return;
    // order 4
    Coord p2;  for (j=0; j<3; j++) p2[j]=p[j]*p[j];
    for (j=0; j<3; j++) for (k=j; k<3; k++)
    {
        basisDeriv[count](k,j)=4*p[j]*p[k];
        basisDeriv[count](k,k)=2*p2[j];
        basisDeriv[count](j,j)=2*p2[k];
        count++;
    }
    basisDeriv[count](0,0)=2*p[1]*p[2];
    basisDeriv[count](0,1)=2*p[0]*p[2];
    basisDeriv[count](0,2)=2*p[0]*p[1];
    basisDeriv[count](1,2)=p2[0];
    count++;
    basisDeriv[count](0,1)=2*p[1]*p[2];
    basisDeriv[count](0,2)=p2[1];
    basisDeriv[count][1][1]=2*p[0]*p[2];
    basisDeriv[count](1,2)=2*p[0]*p[1];
    count++;
    basisDeriv[count](0,1)=p2[2];
    basisDeriv[count](0,2)=2*p[1]*p[2];
    basisDeriv[count](1,2)=2*p[0]*p[2];
    basisDeriv[count](2,2)=2*p[0]*p[1];
    count++;

    for (j=0; j<3; j++) for (k=0; k<3; k++) if (j!=k)
    {
        basisDeriv[count](k,j)=3*p2[j];
        basisDeriv[count](j,j)=6*p[j]*p[k];
        count++;
    }
    if (count==dim) return;

    return; // order>4 not implemented...
}



template<class Real>
Vec<3,Real> getOrder1Factors(const helper::vector<Real>& v)
{
    Vec<3,Real> ret;
    if(v.size()>=4)
    {
        ret(0)=v[1];
        ret(1)=v[2];
        ret(2)=v[3];
    }
    return ret;
}

template<class Real>
Vec<6,Real> getOrder2Factors(const helper::vector<Real>& v)
{
    Vec<6,Real> ret;
    if(v.size()>=10)
    {
        ret(0)=v[4];    // x * x
        ret(1)=v[5];    // x * y
        ret(2)=v[6];    // x * z
        ret(3)=v[7];    // y * y
        ret(4)=v[8];    // y * z
        ret(5)=v[9];    // z * z
    }
    else //unit cube
    {
        ret(0)=ret(3)=ret(5)=(Real)1./(Real)12.;
    }
    return ret;
}

template<class Real>
Mat<3,6,Real> getOrder3Factors(const helper::vector<Real>& v)
{
    Mat<3,6,Real> ret;
    if(v.size()>=20)
    {
        //  x^2             xy              y^2             xz              yz              z^2
        ret(0,0)=v[11]; ret(0,1)=v[12]; ret(0,2)=v[14]; ret(0,3)=v[13]; ret(0,4)=v[10]; ret(0,5)=v[17];     // x
        ret(1,0)=v[12]; ret(1,1)=v[14]; ret(1,2)=v[15]; ret(1,3)=v[10]; ret(1,4)=v[16]; ret(1,5)=v[18];     // y
        ret(2,0)=v[13]; ret(2,1)=v[10]; ret(2,2)=v[16]; ret(2,3)=v[17]; ret(2,4)=v[18]; ret(2,5)=v[19];     // z
    }
    return ret;
}

template<class Real>
MatSym<6,Real> getOrder4Factors(const helper::vector<Real>& v)
{
    MatSym<6,Real> ret;
    if(v.size()>=35)
    {
        //  x^2             xy              y^2             xz              yz              z^2
        ret(0,0)=v[20];                                                                                     // x^2
        ret(1,0)=v[29]; ret(1,1)=v[21];                                                                     // xy
        ret(2,0)=v[21]; ret(2,1)=v[31]; ret(2,2)=v[23];                                                     // y^2
        ret(3,0)=v[30]; ret(3,1)=v[26]; ret(3,2)=v[27]; ret(3,3)=v[22];                                     // xz
        ret(4,0)=v[26]; ret(4,1)=v[27]; ret(4,2)=v[32]; ret(4,3)=v[28]; ret(4,4)=v[24];                     // yz
        ret(5,0)=v[22]; ret(5,1)=v[28]; ret(5,2)=v[24]; ret(5,3)=v[33]; ret(5,4)=v[34]; ret(5,5)=v[25];     // z^2
    }
    else //unit cube
    {
        ret(0,0)=ret(2,2)=ret(5,5)=(Real)1./(Real)80.;
        ret(1,1)=ret(3,3)=ret(4,4)=(Real)1./(Real)144.;
    }
    return ret;
}


/**
* get differential quantities (val(p0), grad(val) (p0), grad2(val) (p0)) given a polynomial fit centered on p0
*/

template<typename real>
void getPolynomialFit_differential(  const helper::vector<real>& coeff, real& Val, Vec<3,real> *Gradient=NULL, Mat<3,3,real>* Hessian=NULL)
{
    Val=coeff[0];
    if(Gradient && coeff.size()>3)  // = Coeff * CompleteBasisDeriv(0,0,0);
    {
        (*Gradient)[0]=coeff[1];
        (*Gradient)[1]=coeff[2];
        (*Gradient)[2]=coeff[3];
    }
    if(Hessian && coeff.size()>9) // = Coeff * CompleteBasisDeriv2(0,0,0);
    {
        (*Hessian)(0,0)=(coeff[4]*(real)2.);
        (*Hessian)(0,1)=(*Hessian)(1,0)=coeff[5];
        (*Hessian)(0,2)=(*Hessian)(2,0)=coeff[6];
        (*Hessian)(1,1)=(coeff[7]*(real)2.);
        (*Hessian)(1,2)=(*Hessian)(2,1)=coeff[8];
        (*Hessian)(2,2)=(coeff[9]*(real)2.);
    }
}

/**
* polynomial approximation of scalar value given 3d offsets (constant for order 0, linear for order 1, quadratic for order 2)
* least squares fit: \f$ argmin_coeff ( sum_i ( val_i - coeff^T.(pos_i)~ )^2 ) = argmin_coeff ||coeff-X.val||^2 \f$
* -> normal equation = \f$  coeff = (X^TX)^-1 X^T val  \f$
* with :
*   - \f$ val_i \f$ is the value to be approximated at location \f$ pos_i \f$
*   - \f$ ()~ \f$ is the polynomial basis of order @param order
*   - \f$ Xij = (pos_i)~ j i \f$
*   - @returns the polynomial coefficients coeff
*/

template<typename real>
void PolynomialFit(helper::vector<real>& coeff, const helper::vector<real>& val, const helper::vector<Vec<3,real> >& pos, const unsigned int order,const real MIN_COEFF=1E-5)
{
    typedef Eigen::Matrix<real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>  Matrix;
    typedef Eigen::Matrix<real,Eigen::Dynamic,1>  Vector;
    const int nbp=pos.size(),dim=(order+1)*(order+2)*(order+3)/6;

    Matrix X(nbp,dim);

    for (int i=0; i<nbp; i++)
    {
        helper::vector<real> basis;
        getCompleteBasis(basis,pos[i],order);
        memcpy(&X(i,0),&basis[0],dim*sizeof(real));
    }
    Eigen::JacobiSVD<Matrix> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

    if (svd.singularValues().minCoeff()<MIN_COEFF) { PolynomialFit(coeff,val, pos, order-1,MIN_COEFF); return;}

    Vector c=svd.solve( Eigen::Map<const Vector>(&val[0],nbp) );

    coeff.resize(dim);
    memcpy(&coeff[0],&c(0),dim*sizeof(real));
}

// returns error \f$ sum_i ( val_i - coeff^T.(pos_i)~ )^2 \f$
template<typename real>
real getPolynomialFit_Error(const helper::vector<real>& coeff, const helper::vector<real>& val, const helper::vector<Vec<3,real> >& pos)
{
    real error=0;
    for(unsigned int i=0; i<pos.size(); i++) error+=getPolynomialFit_Error(coeff, val[i], pos[i]);
    return error;
}

template<typename real>
real getPolynomialFit_Error(const helper::vector<real>& coeff, const real& val, const Vec<3,real>& pos)
{
    int dim=coeff.size(),order;
    if(dim==1) order=0; else if(dim==4) order=1; else if(dim==10) order=2; else if(dim==20) order=3; else order=4;
    dim=(order+1)*(order+2)*(order+3)/6;
    helper::vector<real> basis;
    getCompleteBasis(basis,pos,order);
    real v=0; for (int i=0; i<dim; i++) v+=coeff[i]*basis[i];
    real error=(v-(real)val)*(v-(real)val);
    return error;
}




/**
  Factors used for the computation of Polynomial Fit solution/error for joint domains
*/
template<typename real>
struct PolynomialFitFactors
{

    typedef Eigen::Matrix<real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>  Matrix;
    typedef Eigen::Matrix<real,Eigen::Dynamic,1>  Vector;

    Matrix  a;  // dim*dim matrix : sum_i (pos_i)~.(pos_i)~^T   = (X^TX)
    Matrix  b;  // num_nodes*dim matrix : b_j = sum_i (val_{ji}).(pos_i)~   =  (X^T.val_j)
    Matrix  c;  // num_nodes*num_nodes matrix : c_{jk} = sum_i val_{ji}.val_{ki}
    Matrix  coeff;  // num_nodes*dim matrix of the optimal coeffs minimizing    \sum_j ||coeff_j-X.val_j||^2

    unsigned int nb;                            // nb of voxels in the region
    std::set<unsigned int> voronoiIndices;      // value corresponding to the value in the voronoi image
    Vec<3,real> center;                              // centroid
    std::map<unsigned int , unsigned int> parentsToNodeIndex;    // map between indices of parents to node indices
    Vector vol;  // volume (and moments) of this region

    PolynomialFitFactors() :nb(0),center(0,0,0)  { }
    PolynomialFitFactors(const std::set<unsigned int>& parents, const unsigned int voronoiIndex) :nb(0),center(0,0,0)  { setParents(parents); voronoiIndices.insert(voronoiIndex); }

    void operator =(const PolynomialFitFactors& f)
    {
        a=f.a;
        b=f.b;
        c=f.c;
        coeff=f.coeff;

        nb=f.nb;
        voronoiIndices=f.voronoiIndices;
        center=f.center;
        parentsToNodeIndex=f.parentsToNodeIndex;
        vol=f.vol;
    }

    void setParents(const helper::vector<unsigned int>& parents)     { parentsToNodeIndex.clear();  for(unsigned int i=0;i<parents.size();i++)  parentsToNodeIndex[parents[i]]=i; }
    void setParents(const std::set<unsigned int>& parents)     { parentsToNodeIndex.clear();  unsigned int i=0; for( std::set<unsigned int>::const_iterator it=parents.begin();it!=parents.end();it++)  parentsToNodeIndex[*it]=i++; }

    // compute factors. vals is a num_nodes x nbp matrix
    void fill( const Matrix& val, const helper::vector<Vec<3,real> >& pos, const unsigned int order, const Vec<3,real>& voxelsize, const unsigned int volOrder)
    {
        unsigned int num_nodes = val.rows(); if(!num_nodes) return;

        unsigned int dim=dimFromOrder(order);

        unsigned int volDim=dimFromOrder(volOrder);
        vol.resize(volDim); vol.setZero();

        nb=pos.size();
        Matrix X(nb,dim);
        for (unsigned int i=0;i<nb;i++)
        {
            helper::vector<real> basis;
            defaulttype::getCompleteBasis(basis,pos[i]-center,order); Eigen::Map<Vector> ebasis(&basis[0],dim); X.row(i) = ebasis;
            defaulttype::getCompleteBasisIntegralInCube(basis,pos[i]-center,voxelsize,volOrder); Eigen::Map<Vector> ebasis2(&basis[0],volDim); vol += ebasis2; // treat voxels as volume elements
            //defaulttype::getCompleteBasis(basis,pos[i]-center,volOrder); Eigen::Map<Vector> ebasis2(&basis[0],volDim); vol += ebasis2*voxelsize[0]*voxelsize[1]*voxelsize[2]; // treat voxels as points (simpler but less accurate)
        }
        a = X.transpose()*X;
        b = val*X;
        c = val*val.transpose();
    }

    // direct solve of coeffs from point data
    void directSolve( const Matrix& val, const helper::vector<Vec<3,real> >& pos, const unsigned int order,const real MIN_COEFF=1E-5)
    {
        unsigned int num_nodes = val.rows(); if(!num_nodes) return;

        unsigned int dim=dimFromOrder(order);

        nb=pos.size();
        Matrix X(nb,dim);
        for (unsigned int i=0;i<nb;i++)
        {
            helper::vector<real> basis; defaulttype::getCompleteBasis(basis,pos[i]-center,order); Eigen::Map<Vector> ebasis(&basis[0],dim); X.row(i) = ebasis;
        }

        // jacobi svd
        coeff.resize(num_nodes,dim); coeff.setZero();
        Eigen::JacobiSVD<Matrix> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.singularValues().minCoeff()<MIN_COEFF)
        {
            PolynomialFitFactors<real> fact;
            fact.center=center;
            fact.directSolve(val,pos,order-1,MIN_COEFF);
            coeff.leftCols(dimFromOrder(order-1))=fact.coeff;
        }
        else for (unsigned int i=0;i<num_nodes;i++) coeff.row(i) = svd.solve(val.row(i).transpose());

        // fill a,b,c (for error checking)
        a = X.transpose()*X;
        b = val*X;
        c = val*val.transpose();
    }

    // least squares solution at a given order, using LLT.
    // a and b must be filled.
    // For node i: coeff(i) = a^1.b(i)
    void solve (const unsigned int order)
    {
        unsigned int num_nodes = b.rows(); if(!num_nodes) return;
        unsigned int dim = a.rows(); if(!dim) return;
        coeff.resize(num_nodes,dim);  coeff.setZero();

        unsigned int targetDim=dimFromOrder(order);
        Matrix A,B;
        if(targetDim<dim) {  A = a.topLeftCorner(targetDim,targetDim); B = b.leftCols(targetDim); }
        else {A=a; B=b; targetDim=dim;}

        if(order==0) { for (unsigned int i=0;i<num_nodes;i++) coeff(i,0)=(A(0,0)==0)?1./(real)num_nodes:(B(i,0)/A(0,0)); return; }

        Eigen::LLT<Matrix> llt(A);
        if(llt.info()==Eigen::Success) for (unsigned int i=0;i<num_nodes;i++) coeff.row(i).leftCols(targetDim) = llt.solve(B.row(i).transpose()).transpose();

        bool success=true;
        const real TOL=1E-2;
        if(llt.info()!=Eigen::Success) success=false;
        else if(fabs(coeff.col(0).sum()-1)>TOL) success=false;      // normally checking Eigen::Success should suffice, but for some reasons, some singular matrices remain undetected..
        else for(unsigned int i=1;i<dim;i++) if(fabs(coeff.col(i).sum())>TOL) success=false;

        if(!success) solve(order-1);
    }

    // error integral computation: \f$ ||coeff-X.val||^2 = coeff^T.a.coeff - 2.coeff^T.b + c \f$
    real getError(const int nodeIndex=-1) const
    {
        real err = 0;
        int num_nodes = b.rows();
        for (int i=0; i<num_nodes; i++) if(nodeIndex==-1 || nodeIndex==i) err += c(i,i) +  coeff.row(i) * a * coeff.row(i).transpose() - (real)2. * coeff.row(i) * b.row(i).transpose();
        return err;
    }

    // error computation at point p: \f$  ( val - coeff^T.(pos)~ )^2  \f$
    real getError(const Vec<3,real>& p, const Vector& val) const
    {
        unsigned int dim = coeff.cols(); if(!dim) return -1;
        unsigned int order=orderFromDim(dim);

        helper::vector<real> basis; defaulttype::getCompleteBasis(basis,p-center,order); Eigen::Map<Vector> ebasis(&basis[0],dim);

        int num_nodes = val.cols(); if(coeff.rows()!=num_nodes) return -1;
        real err = 0;
        for (int i=0; i<num_nodes; i++) { real e=(val(i) - coeff.row(i) * ebasis); err += e*e; }
        return err;
    }


    // get factors when all pos are translated by t
    void setCenter(const Vec<3,real>& ctr)
    {
        unsigned int dim = a.rows();
        if(dim)
        {
            Matrix T=defaulttype::getCompleteBasis_TranslationMatrix(center-ctr,orderFromDim(dim));
            a = T*a*T.transpose();
            b = b*T.transpose();
        }

        unsigned int volDim = vol.rows();
        if(volDim)  vol = defaulttype::getCompleteBasis_TranslationMatrix(center-ctr,orderFromDim(volDim))*vol;

        center=ctr;
    }

    // updates nodes given that vals for new node i is a weighted sum of old vals \sum val_j w(i,j)
    void updateNodes(const Matrix& w, const std::vector<unsigned int> newParents)
    {
        b = w*b;
        c = w*c*w.transpose();
        setParents(newParents);
    }

    void updateNodes(const std::map<unsigned int,Vector>& wmap)
    {
        typedef typename std::map<unsigned int,Vector>::const_iterator wMapIt;
        int num_nodes = b.rows(),count=0;
        Matrix w(wmap.size(),num_nodes);
        std::vector<unsigned int> newParents(wmap.size());
        for(wMapIt it=wmap.begin();it!=wmap.end();it++)
        {
            newParents[count]=it->first;
            w.row(count)=it->second;
            count++;
        }
        b = w*b;
        c = w*c*w.transpose();
        setParents(newParents);
    }

    // operators for two disjoint domains with same nodes
    // you should have : center = f.center,  parentsToNodeIndex = f.parentsToNodeIndex
    void operator +=(const PolynomialFitFactors& f)
    {
        a += f.a;
        b += f.b;
        c += f.c;
        nb += f.nb;
        vol = vol + f.vol;
        voronoiIndices.insert(f.voronoiIndices.begin(),f.voronoiIndices.end());
    }

    // returns differential coeffs for each parent
    void getMapping(helper::vector<unsigned int>& index,helper::vector<real>& w, helper::vector<Vec<3,real> >& dw, helper::vector<Mat<3,3,real> >& ddw)
    {
        unsigned int num_nodes = b.rows(); if(!num_nodes) return;
        unsigned int dim = a.rows(); if(!dim) return;

        index.resize(num_nodes); w.resize(num_nodes); dw.resize(num_nodes); ddw.resize(num_nodes);
        for(std::map<unsigned int , unsigned int>::iterator it=parentsToNodeIndex.begin(); it!=parentsToNodeIndex.end(); it++)
        {
            unsigned int i = it->second;
            index[i] = it->first;
            w[i]=coeff(i,0);
            if(dim>3)  // = Coeff * CompleteBasisDeriv(0,0,0);
            {
                dw[i][0]=coeff(i,1);
                dw[i][1]=coeff(i,2);
                dw[i][2]=coeff(i,3);
            }
            if(dim>9) // = Coeff * CompleteBasisDeriv2(0,0,0);
            {
                ddw[i](0,0)=(coeff(i,4)*(real)2.);
                ddw[i](0,1)=ddw[i](1,0)=coeff(i,5);
                ddw[i](0,2)=ddw[i](2,0)=coeff(i,6);
                ddw[i](1,1)=(coeff(i,7)*(real)2.);
                ddw[i](1,2)=ddw[i](2,1)=coeff(i,8);
                ddw[i](2,2)=(coeff(i,9)*(real)2.);
            }
        }
    }

    //helpers
    inline unsigned int orderFromDim(const unsigned int dim) const { if(dim==1) return 0; else if(dim==4) return 1; else if(dim==10) return 2; else if(dim==20) return 3; else return 4;}
    inline unsigned int dimFromOrder(const unsigned int order) const { return (order+1)*(order+2)*(order+3)/6; }


};


} // namespace defaulttype
} // namespace sofa


#endif
