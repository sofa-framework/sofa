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

#ifndef FLEXIBLE_StabilizedHookeMaterialBlock_H
#define FLEXIBLE_StabilizedHookeMaterialBlock_H


#include "../material/BaseMaterial.h"
#include "StabilizedHookeMaterialBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"
#include "../BaseJacobian.h"
#include <sofa/helper/decompose.h>

namespace sofa
{

namespace defaulttype
{


//////////////////////////////////////////////////////////////////////////////////
////  default implementation for U331
//////////////////////////////////////////////////////////////////////////////////

template<class _T>
class StabilizedHookeMaterialBlock:
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
      * stabilized Hooke's law
      *     - W  = mu.sum_i((Ui-1)^2) + lambda/2 * (J-1)^2
      *     - fi  = - ( 2mu(Ui-1) + lambda(J-1).Uj.Uk
      *     - dfi/dUi = -2mu - lambda Uj^2 Uk^2
      *       dfi/dUj = - lambda ( Ui.Uj.Uk^2 + (J-1).Uk )
      * see maple file ./doc/stabilizedHooke_principalStretches.mw
      */

    static const bool constantK=false;

    Real mu2Vol;  ///<  2 * mu * volume
    Real lambdaVol;  ///<  lambda * volume

    mutable MatBlock _K;

    void init( Real youngM, Real poissonR )
    {
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];

        lambdaVol = vol * youngM*poissonR/((1-2*poissonR)*(1+poissonR)) ;
        mu2Vol = vol * /*0.5* */youngM/(1+poissonR);
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        const Real& U1 = x.getStrain()[0];
        const Real& U2 = x.getStrain()[1];
        const Real& U3 = x.getStrain()[2];

        Real U1m1 = U1-1;
        Real U2m1 = U2-1;
        Real U3m1 = U3-1;

        Real J = U1*U2*U3;
        Real Jm1 = J-1;

        return 0.5*mu2Vol*( U1m1*U1m1 + U2m1*U2m1 + U3m1*U3m1 ) + lambdaVol * 0.5 * Jm1*Jm1;
    }

    void addForce( Deriv& f, const Coord& x, const Deriv& /*v*/) const
    {
        Real U[3] = { x.getStrain()[0], x.getStrain()[1],x.getStrain()[2]  };
        Real squareU[3] = { U[0]*U[0], U[1]*U[1], U[2]*U[2] };

        Real J = U[0]*U[1]*U[2];
        Real Jm1 = J-1;

        for( int i=0; i<3 ; ++i )
        {
            int j=(i+1)%3;
            int k=(j+1)%3;
            f.getStrain()[i] -= mu2Vol*(U[i]-1) + lambdaVol * Jm1 * U[j] * U[k]; // - ( 2mu(Ui-1) + lambda(J-1).Uj.Uk
            _K[i][i] = 2 * mu2Vol +lambdaVol*squareU[j]*squareU[k]; // -2mu - lambda Uj^2 Uk^2

            for( j=i+1 ; j<3 ; ++j )
            {
                k=(j+1)%3;
                _K[i][j] = _K[j][i] = lambdaVol* ( U[i]*U[j]*squareU[k] + Jm1*U[k] ); // - lambda ( Ui.Uj.Uk^2 + (J-1).Uk )
            }
        }

        // ensure _K is symmetric, positive semi-definite as suggested in [Teran05]
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

