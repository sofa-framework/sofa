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
#ifndef FLEXIBLE_MLSJacobianBlock_H
#define FLEXIBLE_MLSJacobianBlock_H

#include "../BaseJacobian.h"

#include "../types/QuadraticTypes.h"
#include "../types/AffineTypes.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/MatSym.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace defaulttype
{

/** Template class used to implement one jacobian block for MLSMapping */
template<class TIn, class TOut>
class MLSJacobianBlock : public BaseJacobianBlock<TIn,TOut> {};





/** Helper to retrieve coordinates and MLS order from a variety of input types
*/

template< class Coord>
class InInfo
{
public:
    enum {dim = Coord::spatial_dimensions};
    typedef typename Coord::value_type Real;
    static const unsigned int order = 0;
    static Vec<dim,Real> getCenter(const Coord& x)    { return x; }
};

template<class TCoord, class TDeriv, class TReal>
class InInfo<defaulttype::StdVectorTypes<TCoord, TDeriv, TReal> >
{
public:
    enum {dim = TCoord::spatial_dimensions};
    typedef typename TCoord::value_type Real;
    static const unsigned int order = 0;
    static Vec<dim,Real> getCenter(const TCoord& x)    { return x; }
};

template<int _dim, typename _Real>
class InInfo<defaulttype::StdAffineTypes<_dim, _Real> >
{
public:
    typedef typename defaulttype::StdAffineTypes<_dim, _Real> T;
    enum {dim =_dim};
    typedef _Real Real;
    static const unsigned int order = 1;
    static Vec<dim,Real> getCenter(const typename T::Coord& x)    { return x.getCenter(); }
};

template<int _dim, typename _Real>
class InInfo<defaulttype::StdRigidTypes<_dim, _Real> >
{
public:
    typedef typename defaulttype::StdRigidTypes<_dim, _Real> T;
    enum {dim =_dim};
    typedef _Real Real;
    static const unsigned int order = 1;
    static Vec<dim,Real> getCenter(const typename T::Coord& x)    { return x.getCenter(); }
};


template<int _dim,typename _Real>
class InInfo<defaulttype::StdQuadraticTypes<_dim, _Real> >
{
public:
    typedef typename defaulttype::StdQuadraticTypes<_dim, _Real> T;
    enum {dim =_dim};
    typedef _Real Real;
    static const unsigned int order = 2;
    static Vec<dim,Real> getCenter(const typename T::Coord& x)    { return x.getCenter(); }
};

/** Class to compute basis and covariance matrix from a variety of input types (needed by MLS moment matrix)
  order=0 corresponds to MLS for points
  order=1 corresponds to GMLS for linear frames (affine, rigid)
  order=2 corresponds to GMLS for quadratic frames
*/

template<unsigned int dim,unsigned int order,typename _Real>
class MLSInfo  {};

template<unsigned int dim,typename _Real>
class MLSInfo<dim,0,_Real>
{
public:
    static const unsigned int bdim = 1+dim; ///< size of complete basis
    typedef _Real Real;
    typedef Vec<dim,Real> coord;
    typedef Vec<bdim,Real> basis;
    typedef MatSym<bdim,Real> moment;

    static basis getBasis(const coord& x)
    {
        basis b;
        b[0]=1; for(unsigned int i=0;i<dim;i++) b[i+1]=x[i];
        return b;
    }

    static basis getBasisGradient(const coord& ,const unsigned int axis)
    {
        basis b;
        b[1+axis]=1.;
        return b;
    }

    static basis getBasisHessian(const coord& ,const unsigned int ,const unsigned int )
    {
        basis b;
        return b;
    }

    static moment getCov(const coord& x)
    {
        return covN(getBasis(x));
    }
};

template<unsigned int dim,typename _Real>
class MLSInfo<dim,1,_Real>
{
public:
    static const unsigned int bdim = 1+dim; ///< size of complete basis
    typedef _Real Real;
    typedef Vec<dim,Real> coord;
    typedef Vec<bdim,Real> basis;
    typedef MatSym<bdim,Real> moment;

    static basis getBasis(const coord& x)
    { return MLSInfo<dim,0,Real>::getBasis(x); }

    static basis getBasisGradient(const coord& x,const unsigned int axis)
    { return MLSInfo<dim,0,Real>::getBasisGradient(x,axis); }

    static basis getBasisHessian(const coord& x,const unsigned int axis1,const unsigned int axis2)
    { return MLSInfo<dim,0,Real>::getBasisHessian(x,axis1,axis2); }

    static moment getCov(const coord& x)
    {
        moment M = covN(getBasis(x));
        for(unsigned int i=1;i<bdim;i++) M(i,i)+=1.; // GMLS term corresponding to first derivatives = sum Gradient.Gradient^T
        return M;
    }
};


template<unsigned int dim,typename _Real>
class MLSInfo<dim,2,_Real>
{
public:
    typedef defaulttype::StdQuadraticTypes<dim, _Real> Qtypes;
    static const unsigned int bdim = 1 + Qtypes::num_quadratic_terms; ///< size of complete basis
    typedef _Real Real;
    typedef Vec<dim,Real> coord;
    typedef Vec<bdim,Real> basis;
    typedef MatSym<bdim,Real> moment;

    static basis getBasis(const coord& x)
    {
        Vec<bdim-1,Real> x2 = defaulttype::convertSpatialToQuadraticCoord(x);
        basis b;
        b[0]=1;
        for(unsigned int i=1;i<bdim;i++) b[i]=x2[i-1];
        return b;
    }

    static basis getBasisGradient(const coord& x,const unsigned int axis)
    {
        Vec<bdim-1,Real> grad = defaulttype::SpatialToQuadraticCoordGradient(x).col(axis);
        basis b;
        for(unsigned int i=1;i<bdim;i++) b[i]=grad[i-1];
        return b;
    }

    static basis getBasisHessian(const coord& x,const unsigned int axis1,const unsigned int axis2)
    {
        basis b;
        if(axis1==axis2) {b[1+dim+axis1]=2.; return b;}
        if(axis2==axis1+1 || (dim==axis1+1 && axis2==0) )  {b[1+2*dim+axis1]=1.; return b;}
        return  getBasisHessian(x,axis2,axis1);
    }

    static moment getCov(const coord& x)
    {
        moment M = covN(getBasis(x));

        // GMLS term corresponding to first derivatives = sum Gradient.Gradient^T
        for(unsigned int j=0;j<dim;j++)
        {
            Vec<bdim,Real> b = getBasisGradient(x,j);
            M+=covN(b);
        }
        // GMLS term corresponding to second derivatives = sum(i,j) Hessian(i,j).Hessian(i,j)^T
        for(unsigned int i=dim+1;i<2*dim+1;i++) M(i,i)+=2.; // square terms
        for(unsigned int i=2*dim+1;i<bdim;i++) M(i,i)+=2.; // cross terms

        return M;
    }
};


template<class basis>
const Vec<basis::spatial_dimensions-1,typename basis::value_type>& BasisToCoord(const basis& v) { return *reinterpret_cast<const Vec<basis::spatial_dimensions-1,typename basis::value_type>*>(&v[1]); }



} // namespace defaulttype
} // namespace sofa



#endif
