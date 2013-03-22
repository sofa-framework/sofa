/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
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



/** Class to compute covariance matrix from a variety of input types (needed by MLS moment matrix)
*/

template< class Coord>
class MLSInfo
{
public:
    enum {dim = Coord::spatial_dimensions};
    static const unsigned int bdim = 1+dim; ///< size of complete basis
    typedef typename Coord::value_type Real;

    // returns X: the complete polynomial basis of order 1
    static Vec<bdim,Real> getBasis(const Coord& x)
    {
        Vec<bdim,Real> basis;
        basis[0]=1; for(unsigned int i=0;i<dim;i++) basis[i+1]=x[i];
        return basis;
    }

    // returns B.B^T
    static MatSym<bdim,Real> getCov(const Coord& x)
    {
        return covN(getBasis(x));
    }
};

template<unsigned int dim, typename _Real>
class MLSInfo<defaulttype::StdAffineTypes<dim, _Real> >
{
    typedef defaulttype::StdAffineTypes<3, _Real> Coord;
    static const unsigned int bdim = 1+dim; ///< size of complete basis
    typedef _Real Real;

    static Vec<bdim,Real> getBasis(const Coord& x)
    {
        Vec<bdim,Real> basis;
        basis[0]=1; for(unsigned int i=0;i<dim;i++) basis[i+1]=x.getCenter()[i];
        return basis;
    }

    static MatSym<bdim,Real> getCov(const Coord& x)
    {
        MatSym<bdim,Real> M = covN(getBasis(x));
        for(unsigned int i=1;i<bdim;i++) M(i,i)+=1.; // GMLS term corresponding to first derivatives
        return M;
    }
};

// same as affine
template<unsigned int dim, typename _Real>
class MLSInfo<defaulttype::StdRigidTypes<dim, _Real> >
{
    typedef defaulttype::StdRigidTypes<3, _Real> Coord;
    static const unsigned int bdim = 1+dim; ///< size of complete basis
    typedef _Real Real;

    static Vec<bdim,Real> getBasis(const Coord& x)
    {
        Vec<bdim,Real> basis;
        basis[0]=1; for(unsigned int i=0;i<dim;i++) basis[i+1]=x.getCenter()[i];
        return basis;
    }

    static MatSym<bdim,Real> getCov(const Coord& x)
    {
        MatSym<bdim,Real> M = covN(getBasis(x));
        for(unsigned int i=1;i<bdim;i++) M(i,i)+=1.; // GMLS term corresponding to first derivatives
        return M;
    }
};


template<unsigned int dim,typename _Real>
class MLSInfo<defaulttype::StdQuadraticTypes<dim, _Real> >
{
    typedef defaulttype::StdQuadraticTypes<dim, _Real> Coord;
    static const unsigned int bdim = 1 + Coord::num_quadratic_terms; ///< size of complete basis
    typedef _Real Real;

    static Vec<bdim,Real> getBasis(const Coord& x)
    {
        Vec<bdim-1,Real> x2 = defaulttype::convertSpatialToQuadraticCoord(x.getCenter());
        Vec<bdim,Real> basis;
        basis[0]=1;
        for(unsigned int i=1;i<bdim;i++) basis[i]=x2[i-1];
        return basis;
    }

    static MatSym<bdim,Real> getCov(const Coord& x)
    {
        MatSym<bdim,Real> M = covN(getBasis(x));

        // GMLS term corresponding to first derivatives
        Mat<bdim-1,dim,Real> grad = defaulttype::SpatialToQuadraticCoordGradient(x.getCenter());
        for(unsigned int j=0;j<dim;j++)
        {
            Vec<bdim,Real> b; for(unsigned int i=1;i<bdim;i++) b[i]=grad[i-1][j];
            M+=covN(b);
        }
        // GMLS term corresponding to second derivatives
        for(unsigned int i=dim+1;i<2*dim+1;i++) M(i,i)+=2.; // square terms
        for(unsigned int i=2*dim+1;i<bdim;i++) M(i,i)+=1.; // cross terms

        return M;
    }
};



} // namespace defaulttype
} // namespace sofa



#endif
