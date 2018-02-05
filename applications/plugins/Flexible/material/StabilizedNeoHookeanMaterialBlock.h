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

#ifndef FLEXIBLE_StabilizedNeoHookeanMaterialBlock_H
#define FLEXIBLE_StabilizedNeoHookeanMaterialBlock_H


#include "../material/BaseMaterial.h"
#include "StabilizedNeoHookeanMaterialBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"
#include <sofa/helper/decompose.h>
#include "../BaseJacobian.h"

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  default implementation for U331
//////////////////////////////////////////////////////////////////////////////////

template<class _T>
class StabilizedNeoHookeanMaterialBlock:
    public  BaseMaterialBlock< _T >
{
public:
    typedef _T T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    /**
      * DOFs: principal stretches U1,U2,U3   J=U1*U2*U3
      *
      * stabilized Neo-Hookean
      *     - W = mu/2(I1-3)-mu.ln(J)+lambda/2(ln(J))^2
      * see maple file ./doc/StabilizedNeoHookean_principalStretches.mw for derivative
      */

    static const bool constantK=false;

    Real lambdaVol;  ///<  0.5 * first coef * volume
    Real muVol;   ///<  0.5 * volume coef * volume

    mutable MatBlock _K;

    void init(const Real &youngM,const Real &poissonR)
    {
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];

        lambdaVol = vol * youngM*poissonR/((1-2*poissonR)*(1+poissonR)) ;
        muVol = vol * 0.5 * youngM/(1+poissonR);
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        const Real& U1 = x.getStrain()[0];
        const Real& U2 = x.getStrain()[1];
        const Real& U3 = x.getStrain()[2];

        Real J = U1*U2*U3;
        Real logJ = log(J);
        Real squareU[3] = { U1*U1, U2*U2, U3*U3 };

        return muVol*0.5*(squareU[0]+squareU[1]+squareU[2]-3) - muVol*logJ + 0.5*lambdaVol*logJ*logJ;
    }

    void addForce( Deriv& f, const Coord& x, const Deriv& /*v*/) const
    {
        const Real& U1 = x.getStrain()[0];
        const Real& U2 = x.getStrain()[1];
        const Real& U3 = x.getStrain()[2];

        Real invU[3] = { 1.0/U1, 1.0/U2, 1.0/U3 };

        Real J = U1 *  U2 * U3;
        Real logJ = log(J);

        Real t1 = -muVol + lambdaVol * logJ;
        f.getStrain()[0] -= t1 * invU[0] + muVol * U1;
        f.getStrain()[1] -= t1 * invU[1] + muVol * U2;
        f.getStrain()[2] -= t1 * invU[2] + muVol * U3;


        Real t2 = 0.1e1 - logJ;
        Real t3 = invU[0]*invU[0];
        Real t6 = lambdaVol * invU[1] * invU[0];
        Real t7 = lambdaVol * invU[2];
        t1 = t7 * invU[0];
        Real t8 = invU[1]*invU[1];
        Real t4 = t7 * invU[1];
        Real t5 = invU[2]*invU[2];
        _K[0][0] = t3 * t2 * lambdaVol + (0.1e1 + t3) * muVol;
        _K[0][1] = t6;
        _K[0][2] = t1;
        _K[1][0] = _K[0][1];
        _K[1][1] = t8 * t2 * lambdaVol + (0.1e1 + t8) * muVol;
        _K[1][2] = t4;
        _K[2][0] = _K[0][2];
        _K[2][1] = _K[1][2];;
        _K[2][2] = t5 * t2 * lambdaVol + (0.1e1 + t5) * muVol;


        // ensure _K is symmetric, positive semi-definite (even if it is not as good as positive definite) as suggested in [Teran05]
        helper::Decompose<Real>::PSDProjection( _K );
    }

    void addDForce( Deriv& df, const Deriv& dx, const SReal& kfactor, const SReal& /*bfactor*/ ) const
    {
        df.getStrain() -= _K * dx.getStrain() * kfactor;
    }

    MatBlock getK() const
    {
        return -_K;
    }

    MatBlock getC() const
    {
        MatBlock C = MatBlock();
        C.invert( _K );
        return C;
    }

    MatBlock getB() const
    {
        return MatBlock();
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  specialization for U321
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class StabilizedNeoHookeanMaterialBlock< U321(_Real) >:
    public  BaseMaterialBlock< U321(_Real) >
{
public:
    typedef U321(_Real) T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    /**
      * DOFs: principal stretches U1,U2   J=U1*U2
      *
      * classic Neo-Hookean
      *     - W = mu/2(U1^2+U2^1-2)-mu.ln(J)+lambda/2(ln(J))^2
      * see maple file ./doc/StabilizedNeoHookean_principalStretches.mw for derivative
      */

    static const bool constantK=false;

    Real lambdaVol;  ///<  0.5 * first coef * volume
    Real muVol;   ///<  0.5 * volume coef * volume

    mutable MatBlock _K;

    void init(const Real &youngM,const Real &poissonR)
    {
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];

        lambdaVol = vol * youngM*poissonR/((1-2*poissonR)*(1+poissonR)) ;
        muVol = vol * 0.5 * youngM/(1+poissonR);
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        const Real& U1 = x.getStrain()[0];
        const Real& U2 = x.getStrain()[1];

        const Real J = U1*U2;
        const Real logJ = log(J);
        const Real squareU[2] = { U1*U1, U2*U2 };

        return muVol*0.5*(squareU[0]+squareU[1]-2) - muVol*logJ + 0.5*lambdaVol*logJ*logJ;
    }

    void addForce( Deriv& f, const Coord& x, const Deriv& /*v*/) const
    {
        const Real& U1 = x.getStrain()[0];
        const Real& U2 = x.getStrain()[1];

        const Real invU[2] = { 1.0/U1, 1.0/U2 };
        const Real invSquareU[2] = { 1.0/(U1*U1), 1.0/(U2*U2) };

        const Real J = U1 *  U2;
        const Real logJ = log(J);

        Real t1 = -muVol + lambdaVol * logJ;
        f.getStrain()[0] -= t1 * invU[0] + muVol * U1;
        f.getStrain()[1] -= t1 * invU[1] + muVol * U2;


        Real lambdaLogJ = lambdaVol * logJ;

        _K[0][0] = muVol + (muVol+lambdaVol-lambdaLogJ)*invSquareU[0];
        _K[0][1] = lambdaVol / J;
        _K[1][0] = _K[0][1];
        _K[1][1] = muVol + (muVol+lambdaVol-lambdaLogJ)*invSquareU[1];


        // ensure _K is symmetric positive semi-definite (even if it is not as good as positive definite) as suggested in [Teran05]
        helper::Decompose<Real>::PSDProjection( _K );
    }

    void addDForce( Deriv& df, const Deriv& dx, const SReal& kfactor, const SReal& /*bfactor*/ ) const
    {
        df.getStrain() -= _K * dx.getStrain() * kfactor;
    }

    MatBlock getK() const
    {
        return -_K;
    }

    MatBlock getC() const
    {
        MatBlock C = MatBlock();
        C.invert( _K );
        return C;
    }

    MatBlock getB() const
    {
        return MatBlock();
    }
};







} // namespace defaulttype
} // namespace sofa



#endif

