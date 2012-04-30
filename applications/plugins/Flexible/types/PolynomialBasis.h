/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
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

#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace sofa
{
namespace defaulttype
{

using helper::vector;

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
inline void getCompleteBasis(vector<real>& basis, const Vec<3,real>& p,const unsigned int order)
{
    typedef Vec<3,real> Coord;

    unsigned int j,k,dim=(order+1)*(order+2)*(order+3)/6;

    basis.resize(dim);  for (j=0; j<dim; j++) basis[j]=0;

    Coord p2;    for (j=0; j<3; j++) p2[j]=p[j]*p[j];
    Coord p3;    for (j=0; j<3; j++) p3[j]=p2[j]*p[j];

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
    basis[count]=p[0]*p[1]*p[2];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++)
        {
            basis[count]=p2[j]*p[k];
            count++;
        }
    if (count==dim) return;
    // order 4
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


template<typename real>
inline void getCompleteBasisGradient(vector<Vec<3,real> >& basisDeriv, const Vec<3,real>& p,const unsigned int order)
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
inline void getCompleteBasisHessian(vector<MatSym<3,real> >& basisDeriv, const Vec<3,real>& p,const unsigned int order)
{
    typedef Vec<3,real> Coord;

    unsigned int j,k,dim=(order+1)*(order+2)*(order+3)/6;

    basisDeriv.resize(dim);    for (k=0; k<dim; k++) basisDeriv[k].fill(0);

    Coord p2;  for (j=0; j<3; j++) p2[j]=p[j]*p[j];

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
void PolynomialFit(vector<real>& coeff, const vector<real>& val, const vector<Vec<3,real> >& pos, const unsigned int order,const real MIN_COEFF=1E-5)
{
    typedef  Vec<3,real> Coord;
    typedef Eigen::Matrix<real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>  Matrix;
    typedef Eigen::Matrix<real,Eigen::Dynamic,1>  Vector;
    const int nbp=pos.size(),dim=(order+1)*(order+2)*(order+3)/6;

    Matrix X(nbp,dim);

    for (int i=0; i<nbp; i++)
    {
        vector<real> basis;
        getCompleteBasis(basis,pos[i],order);
        memcpy(&X(i,0),&basis[0],dim*sizeof(real));
    }
    Eigen::JacobiSVD<Matrix> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

    if (svd.singularValues().minCoeff()<MIN_COEFF) { PolynomialFit(coeff,val, pos, order-1,MIN_COEFF); return;}

    Vector c=svd.solve( Eigen::Map<const Vector>(&val[0],nbp) );
//    std::cout<<"X="<<X<<std::endl;
//    std::cout<<"y="<<y<<std::endl;
//std::cout<<"c="<<c<<std::endl;
//std::cout<<"nb="<<svd.singularValues ()<<std::endl;
    coeff.resize(dim);
    memcpy(&coeff[0],&c(0),dim*sizeof(real));
}


template<typename real>
real getPolynomialFit_Error(const vector<real>& coeff, const vector<real>& val, const vector<Vec<3,real> >& pos)
{
    real error=0;
    for(unsigned int i=0; i<pos.size(); i++) error+=getPolynomialFit_Error(coeff, val[i], pos[i]);
    return error;
}

template<typename real>
real getPolynomialFit_Error(const vector<real>& coeff, const real& val, const Vec<3,real>& pos)
{
    typedef  Vec<3,real> Coord;
    int dim=coeff.size(),order;
    if(dim==1) order=0; else if(dim==4) order=1; else if(dim==10) order=2; else if(dim==20) order=3; else order=4;
    dim=(order+1)*(order+2)*(order+3)/6;
    vector<real> basis;
    getCompleteBasis(basis,pos,order);
    real v=0; for (int i=0; i<dim; i++) v+=coeff[i]*basis[i];
    real error=(v-(real)val)*(v-(real)val);
    return error;
}


/**
* get differential quantities (val(p0), grad(val) (p0), grad2(val) (p0)) given a polynomial fit centered on p0
*/

template<typename real>
void getPolynomialFit_differential(  const vector<real>& coeff, real& Val, Vec<3,real> *Gradient=NULL, Mat<3,3,real>* Hessian=NULL)
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


} // namespace defaulttype
} // namespace sofa


#endif
