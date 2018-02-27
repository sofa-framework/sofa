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

#ifndef FLEXIBLE_PolynomialMaterialBlock_H
#define FLEXIBLE_PolynomialMaterialBlock_H


#include "../material/BaseMaterial.h"
#include "../BaseJacobian.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"
#include <sofa/helper/decompose.h>

namespace sofa
{

namespace defaulttype
{

template<class T>
class PolynomialMaterialBlock : public BaseMaterialBlock<T> {};


//////////////////////////////////////////////////////////////////////////////////
////  I331
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class PolynomialMaterialBlock< I331(_Real) >:
    public  BaseMaterialBlock< I331(_Real) >
{
    public:
        typedef I331(_Real) T;

        typedef BaseMaterialBlock<T> Inherit;
        typedef typename Inherit::Coord Coord;
        typedef typename Inherit::Deriv Deriv;
        typedef typename Inherit::MatBlock MatBlock;
        typedef typename Inherit::Real Real;

        /**
          * DOFs: I1, I2, J
          *
          *     - W = vol* [ sum Cij ( I1/J^2/3 - 3)^i ( I2/J^4/3 - 3)^j + bulk/2 (J -1)^2 ]
          */

        static const bool constantK=false;

        Real C10Vol;
        Real C01Vol;
        Real C20Vol;
        Real C02Vol;
        Real C30Vol;
        Real C03Vol;
        Real C11Vol;
        Real bulkVol;
        mutable MatBlock K;

        void init(const Real &C10,const Real &C01,const Real &C20,const Real &C02,const Real &C30,const Real &C03,const Real &C11,const Real &bulk)
        {
            Real vol=1.;
            if(this->volume) vol=(*this->volume)[0];
            C10Vol = C10 * vol;
            C01Vol = C01 * vol;
            C20Vol = C20 * vol;
            C02Vol = C02 * vol;
            C30Vol = C30 * vol;
            C03Vol = C03 * vol;
            C11Vol = C11 * vol;
            bulkVol = bulk * vol;
        }

        Real getPotentialEnergy(const Coord& x) const
        {
            Real Jm23=pow(x.getStrain()[2],-(Real)2./(Real)3.);
            Real Jm43=Jm23*Jm23;

            return C10Vol*(x.getStrain()[0]*Jm23-(Real)3.) +
                   C01Vol*(x.getStrain()[1]*Jm43-(Real)3.) +
                   C20Vol*(x.getStrain()[0]*Jm23-(Real)3.)*(x.getStrain()[0]*Jm23-(Real)3.) +
                   C02Vol*(x.getStrain()[1]*Jm43-(Real)3.)*(x.getStrain()[1]*Jm43-(Real)3.) +
                   C30Vol*(x.getStrain()[0]*Jm23-(Real)3.)*(x.getStrain()[0]*Jm23-(Real)3.)*(x.getStrain()[0]*Jm23-(Real)3.) +
                   C03Vol*(x.getStrain()[1]*Jm43-(Real)3.)*(x.getStrain()[1]*Jm43-(Real)3.)*(x.getStrain()[1]*Jm43-(Real)3.) +
                   C11Vol*(x.getStrain()[0]*Jm23-(Real)3.)*(x.getStrain()[1]*Jm43-(Real)3.) +
                   bulkVol*(Real)0.5*(x.getStrain()[2]-1.0)*(x.getStrain()[2]-1.0);
        }

        void addForce( Deriv& f , const Coord& x , const Deriv& /*v*/) const
        {
            Real Jm13=pow(x.getStrain()[2],-(Real)1./(Real)3.);
            Real Jm23=Jm13*Jm13;
            Real Jm43=Jm23*Jm23;
            Real Jm53=Jm43*Jm13;
            Real Jm73=Jm53*Jm23;
            Real Jm83=Jm73*Jm13;
            Real Jm103=Jm83*Jm23;
            Real Jm113=Jm73*Jm43;
            Real Jm143=Jm103*Jm43;
            Real Jm2=Jm43*Jm23;
            Real Jm3=Jm73*Jm23;
            Real Jm4=Jm2*Jm2;
            Real Jm5=Jm3*Jm2;
            Real Jm6=Jm3*Jm3;

            K[0][0] = -2.*C20Vol*Jm43 - 6.*C30Vol*Jm2*x.getStrain()[0] + 18.*C30Vol*Jm43;
            K[0][1] = K[1][0] = -C11Vol*Jm2;
            K[0][2] = K[2][0] = 2./3.*C10Vol*Jm53 + 8./3.*C20Vol*Jm73*x.getStrain()[0] - 4.*C20Vol*Jm53 + 6.*C30Vol*Jm3*x.getStrain()[0]*x.getStrain()[0] - 24.*C30Vol*Jm73*x.getStrain()[0] + 18.*C30Vol*Jm53 + 2.*C11Vol*Jm3*x.getStrain()[1] - 2.*C11Vol*Jm53;
            K[1][1] = -2.*C02Vol*Jm83 - 6.*C03Vol*Jm4*x.getStrain()[1] + 18.*C03Vol*Jm83;
            K[1][2] = K[2][1] = 4./3.*C01Vol*Jm73 + 16./3.*C02Vol*Jm113*x.getStrain()[1] - 8.*C02Vol*Jm73 + 12.*C03Vol*Jm5*x.getStrain()[1]*x.getStrain()[1] - 48.*C03Vol*Jm113*x.getStrain()[1] + 36.*C03Vol*Jm73 + 2.*C11Vol*Jm3*x.getStrain()[0] - 4.*C11Vol*Jm73;
            K[2][2] = -10./9.*C10Vol*x.getStrain()[0]*Jm83 - 28./9.*C01Vol*x.getStrain()[1]*Jm103 - 28./9.*C20Vol*x.getStrain()[0]*x.getStrain()[0]*Jm103 + 20./3.*C20Vol*x.getStrain()[0]*Jm83 - 88./9.*C02Vol*x.getStrain()[1]*x.getStrain()[1]*Jm143 + 56./3.*C02Vol*x.getStrain()[1]*Jm103 - 6.*C30Vol*x.getStrain()[0]*x.getStrain()[0]*x.getStrain()[0]*Jm4 + 28.*C30Vol*x.getStrain()[0]*x.getStrain()[0]*Jm103 - 30.*C30Vol*x.getStrain()[0]*Jm83 - 20.*C03Vol*x.getStrain()[1]*x.getStrain()[1]*x.getStrain()[1]*Jm6 + 88.*C03Vol*x.getStrain()[1]*x.getStrain()[1]*Jm143 - 84.*C03Vol*x.getStrain()[1]*Jm103 - 6.*C11Vol*x.getStrain()[0]*Jm4*x.getStrain()[1] + 10./3.*C11Vol*x.getStrain()[0]*Jm83 + 28./3.*C11Vol*x.getStrain()[1]*Jm103 - bulkVol;

            f.getStrain()[0] += -C10Vol*Jm23 - 2.*C20Vol*Jm43*x.getStrain()[0] + 6.*C20Vol*Jm23 - 3.*C30Vol*Jm2*x.getStrain()[0]*x.getStrain()[0] + 18.*C30Vol*Jm43*x.getStrain()[0] - 27.*C30Vol*Jm23 - C11Vol*Jm2*x.getStrain()[1] + 3.*C11Vol*Jm23;
            f.getStrain()[1] += -C01Vol*Jm43 - 2.*C02Vol*Jm83*x.getStrain()[1] + 6.*C02Vol*Jm43 - 3.*C03Vol*Jm4*x.getStrain()[1]*x.getStrain()[1] + 18.*C03Vol*Jm83*x.getStrain()[1] - 27.*C03Vol*Jm43 - C11Vol*Jm2*x.getStrain()[0] + 3.*C11Vol*Jm43;
            f.getStrain()[2] += 2./3.*C10Vol*x.getStrain()[0]*Jm53 + 4./3.*C01Vol*x.getStrain()[1]*Jm73 + 4./3.*C20Vol*x.getStrain()[0]*x.getStrain()[0]*Jm73 - 4.*C20Vol*x.getStrain()[0]*Jm53 + 8./3.*C02Vol*x.getStrain()[1]*x.getStrain()[1]*Jm113 - 8.*C02Vol*x.getStrain()[1]*Jm73 + 2.*C30Vol*x.getStrain()[0]*x.getStrain()[0]*x.getStrain()[0]*Jm3 - 12.*C30Vol*x.getStrain()[0]*x.getStrain()[0]*Jm73 + 18.*C30Vol*x.getStrain()[0]*Jm53 + 4.*C03Vol*x.getStrain()[1]*x.getStrain()[1]*x.getStrain()[1]*Jm5 - 24.*C03Vol*x.getStrain()[1]*x.getStrain()[1]*Jm113 + 36.*C03Vol*x.getStrain()[1]*Jm73 + 2.*C11Vol*x.getStrain()[0]*Jm3*x.getStrain()[1] - 2.*C11Vol*x.getStrain()[0]*Jm53 - 4.*C11Vol*x.getStrain()[1]*Jm73 - bulkVol*x.getStrain()[2] + bulkVol;
        }

        void addDForce( Deriv&   df, const Deriv&   dx, const SReal& kfactor, const SReal& /*bfactor*/ ) const
        {
            df.getStrain()+=K*dx.getStrain()*kfactor;
        }

        MatBlock getK() const
        {
            return K;
        }

        MatBlock getC() const
        {
            MatBlock C = MatBlock();
            //C.invert(-K);
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

