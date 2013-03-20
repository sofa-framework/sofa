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
#ifndef FLEXIBLE_OgdenMaterialBlock_INL
#define FLEXIBLE_OgdenMaterialBlock_INL

#include "OgdenMaterialBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"
#include "../material/BaseMaterial.h"

namespace sofa
{

namespace defaulttype
{


//////////////////////////////////////////////////////////////////////////////////
////  default implementation for U331 D331
//////////////////////////////////////////////////////////////////////////////////

template<class _T>
class OgdenMaterialBlock :
    public BaseMaterialBlock< _T >
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
      * classic Ogden
      *     - W = sum(1<=i=<N) mui/alphai (~U1^alphai+~U2^alphai+~U3^alphai-3) + sum(1<=i=<N) 1/di(J-1)^{2i}
      * with       J = U1*U2*U3     and      ~Ui=J^{-1/3}Ui  deviatoric principal stretches
      * see maple file ./doc/Ogden_principalStretches.mw for derivative
      */

    static const bool constantK=true;

    Real mu1Volonalpha1, mu2Volonalpha2, mu3Volonalpha3, alpha1, alpha2, alpha3, volond1, volond2, volond3;

    mutable MatBlock _K;

    static const Real MIN_DETERMINANT() {return 0.001;} ///< threshold to clamp J to avoid undefined deviatoric expressions

    void init( Real mu1, Real mu2, Real mu3, Real _alpha1, Real _alpha2, Real _alpha3, Real d1, Real d2, Real d3 )
    {
        alpha1=_alpha1;
        alpha2=_alpha2;
        alpha3=_alpha3;
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        mu1Volonalpha1 = mu1*vol/alpha1;
        mu2Volonalpha2 = mu2*vol/alpha2;
        mu3Volonalpha3 = mu3*vol/alpha3;
        volond1 = vol/d1;
        volond2 = vol/d2;
        volond3 = vol/d3;
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        Real J = x.getStrain()[0]*x.getStrain()[1]*x.getStrain()[2];
        if( J<MIN_DETERMINANT() ) J = MIN_DETERMINANT();
        Real Jm1 = J-1;
        Real squareJm1 = Jm1*Jm1;
        Real fourJm1 = squareJm1*squareJm1;
        Real Jm13 = pow(J,-1.0/3.0);

        Real devU[3] = { Jm13*x.getStrain()[0], Jm13*x.getStrain()[1], Jm13*x.getStrain()[2] };

        return mu1Volonalpha1 * ( pow(devU[0],alpha1)+pow(devU[1],alpha1)+pow(devU[2],alpha1) - 3 ) +
            mu2Volonalpha2 * ( pow(devU[0],alpha2)+pow(devU[1],alpha2)+pow(devU[2],alpha2) - 3 ) +
            mu3Volonalpha3 * ( pow(devU[0],alpha3)+pow(devU[1],alpha3)+pow(devU[2],alpha3) - 3 ) +
            volond1 * squareJm1 +
            volond2 * fourJm1 +
            volond3 * squareJm1*fourJm1;
    }

    void addForce( Deriv& f, const Coord& x, const Deriv& /*v*/) const
    {
        const Real& U1 = x.getStrain()[0];
        const Real& U2 = x.getStrain()[1];
        const Real& U3 = x.getStrain()[2];

        // TODO optimize this REALLY crappy code generated from maple
        // there are a LOT of redondencies

        Real t1 = U1 * U2;

        Real J = t1 * U3; if( helper::rabs(J)<MIN_DETERMINANT() ) J=helper::sign(J)*MIN_DETERMINANT();
        Real Jm1 = J-1;

        Real J13 = pow(J,1.0/3.0);
        Real Jm13 = 1.0/J13;
        Real J23 = pow(J,2.0/3.0);
        Real Jm23 = 1.0/J23;
        Real Jm43 = pow(J,-4.0/3.0);
        Real Jm73 = pow(J,-7.0/3.0);

        Real t3 = Jm13;
        Real t4 = t3 * U1;
        Real t5 = pow(t4, alpha1);
        Real t6 = -Jm13 / 0.3e1 + t3;
        Real t7 = J13;
        Real t8 = t3 * U2;
        Real t9 = pow(t8, alpha1);
        t3 = t3 * U3;
        Real t10 = pow(t3, alpha1);
        Real t11 = 0.1e1 / U1;
        Real t12 = pow(t4, alpha2);
        Real t13 = pow(t8, alpha2);
        Real t14 = pow(t3, alpha2);
        t4 = pow(t4, alpha3);
        t8 = pow(t8, alpha3);
        t3 = pow(t3, alpha3);
        Real t15 = pow(Jm1, 0.2e1);
        t15 = (2 * volond1) + ((4 * volond2) + 0.6e1 * volond3 * t15) * t15;
        Real t16 = Jm1 * U3;
        Real t17 = 0.1e1 / U2;
        Real t18 = 0.1e1 / U3;
        f.getStrain()[0] -= t16 * U2 * t15 + mu1Volonalpha1 * t11 * alpha1 * (-t9 / 0.3e1 - t10 / 0.3e1 + t5 * t6 * t7) + mu2Volonalpha2 * t11 * alpha2 * (-t13 / 0.3e1 - t14 / 0.3e1 + t12 * t6 * t7) + mu3Volonalpha3 * t11 * alpha3 * (-t8 / 0.3e1 - t3 / 0.3e1 + t4 * t6 * t7);
        f.getStrain()[1] -= t16 * U1 * t15 + mu1Volonalpha1 * t17 * alpha1 * (-t5 / 0.3e1 - t10 / 0.3e1 + t9 * t6 * t7) + mu2Volonalpha2 * t17 * alpha2 * (-t12 / 0.3e1 - t14 / 0.3e1 + t13 * t6 * t7) + mu3Volonalpha3 * t17 * alpha3 * (-t4 / 0.3e1 - t3 / 0.3e1 + t8 * t6 * t7);
        f.getStrain()[2] -= t1 * Jm1 * t15 + mu1Volonalpha1 * t18 * alpha1 * (-t5 / 0.3e1 - t9 / 0.3e1 + t10 * t6 * t7) + mu2Volonalpha2 * t18 * alpha2 * (-t12 / 0.3e1 - t13 / 0.3e1 + t14 * t6 * t7) + mu3Volonalpha3 * t18 * alpha3 * (-t4 / 0.3e1 - t8 / 0.3e1 + t3 * t6 * t7);



        _K[0][0] = mu1Volonalpha1 * (pow(Jm13 * U1, alpha1) * alpha1 * alpha1 * pow(-Jm43 * J / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U1, -0.2e1) + pow(Jm13 * U1, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * U2 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U2 * U3) * J13 / U1 + pow(Jm13 * U1, alpha1) * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 / J / 0.3e1 - pow(Jm13 * U1, alpha1) * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * J13 * pow(U1, -0.2e1) + pow(Jm13 * U2, alpha1) * alpha1 * alpha1 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha1) * alpha1 * pow(U1, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha1) * alpha1 * alpha1 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha1) * alpha1 * pow(U1, -0.2e1) / 0.3e1) + mu2Volonalpha2 * (pow(Jm13 * U1, alpha2) * alpha2 * alpha2 * pow(-Jm43 * J / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U1, -0.2e1) + pow(Jm13 * U1, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * U2 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U2 * U3) * J13 / U1 + pow(Jm13 * U1, alpha2) * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 / J / 0.3e1 - pow(Jm13 * U1, alpha2) * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * J13 * pow(U1, -0.2e1) + pow(Jm13 * U2, alpha2) * alpha2 * alpha2 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha2) * alpha2 * pow(U1, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha2) * alpha2 * alpha2 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha2) * alpha2 * pow(U1, -0.2e1) / 0.3e1) + mu3Volonalpha3 * (pow(Jm13 * U1, alpha3) * alpha3 * alpha3 * pow(-Jm43 * J / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U1, -0.2e1) + pow(Jm13 * U1, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * U2 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U2 * U3) * J13 / U1 + pow(Jm13 * U1, alpha3) * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 / J / 0.3e1 - pow(Jm13 * U1, alpha3) * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 * pow(U1, -0.2e1) + pow(Jm13 * U2, alpha3) * alpha3 * alpha3 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha3) * alpha3 * pow(U1, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha3) * alpha3 * alpha3 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha3) * alpha3 * pow(U1, -0.2e1) / 0.3e1) + 0.2e1 * volond1 * U2 * U2 * U3 * U3 + 0.12e2 * volond2 * pow(Jm1, 0.2e1) * U2 * U2 * U3 * U3 + 0.30e2 * volond3 * pow(Jm1, 0.4e1) * U2 * U2 * U3 * U3;
        _K[0][1] = mu1Volonalpha1 * (-pow(Jm13 * U1, alpha1) * alpha1 * alpha1 / U2 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * J * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U1 + pow(Jm13 * U1, alpha1) * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 * U3 / 0.3e1 - pow(Jm13 * U2, alpha1) * alpha1 * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U2 / U1 / 0.3e1 + pow(Jm13 * U3, alpha1) * alpha1 * alpha1 / U2 / U1 / 0.9e1) + mu2Volonalpha2 * (-pow(Jm13 * U1, alpha2) * alpha2 * alpha2 / U2 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * J * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U1 + pow(Jm13 * U1, alpha2) * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 * U3 / 0.3e1 - pow(Jm13 * U2, alpha2) * alpha2 * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U2 / U1 / 0.3e1 + pow(Jm13 * U3, alpha2) * alpha2 * alpha2 / U2 / U1 / 0.9e1) + mu3Volonalpha3 * (-pow(Jm13 * U1, alpha3) * alpha3 * alpha3 / U2 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * J * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U1 + pow(Jm13 * U1, alpha3) * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 * U3 / 0.3e1 - pow(Jm13 * U2, alpha3) * alpha3 * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U2 / U1 / 0.3e1 + pow(Jm13 * U3, alpha3) * alpha3 * alpha3 / U2 / U1 / 0.9e1) + 0.2e1 * volond1 * U1 * U3 * U3 * U2 + 0.2e1 * volond1 * (Jm1) * U3 + 0.12e2 * volond2 * pow(Jm1, 0.2e1) * U2 * U3 * U3 * U1 + 0.4e1 * volond2 * pow(Jm1, 0.3e1) * U3 + 0.30e2 * volond3 * pow(Jm1, 0.4e1) * U2 * U3 * U3 * U1 + 0.6e1 * volond3 * pow(Jm1, 0.5e1) * U3;
        _K[0][2] = mu1Volonalpha1 * (-pow(Jm13 * U1, alpha1) * alpha1 * alpha1 / U3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U1 + pow(Jm13 * U1, alpha1) * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 * U2 / 0.3e1 + pow(Jm13 * U2, alpha1) * alpha1 * alpha1 / U3 / U1 / 0.9e1 - pow(Jm13 * U3, alpha1) * alpha1 * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U3 / U1 / 0.3e1) + mu2Volonalpha2 * (-pow(Jm13 * U1, alpha2) * alpha2 * alpha2 / U3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U1 + pow(Jm13 * U1, alpha2) * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 * U2 / 0.3e1 + pow(Jm13 * U2, alpha2) * alpha2 * alpha2 / U3 / U1 / 0.9e1 - pow(Jm13 * U3, alpha2) * alpha2 * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U3 / U1 / 0.3e1) + mu3Volonalpha3 * (-pow(Jm13 * U1, alpha3) * alpha3 * alpha3 / U3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U1 + pow(Jm13 * U1, alpha3) * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 * U2 / 0.3e1 + pow(Jm13 * U2, alpha3) * alpha3 * alpha3 / U3 / U1 / 0.9e1 - pow(Jm13 * U3, alpha3) * alpha3 * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U3 / U1 / 0.3e1) + 0.2e1 * volond1 * U1 * U2 * U2 * U3 + 0.2e1 * volond1 * (Jm1) * U2 + 0.12e2 * volond2 * pow(Jm1, 0.2e1) * U2 * U2 * U3 * U1 + 0.4e1 * volond2 * pow(Jm1, 0.3e1) * U2 + 0.30e2 * volond3 * pow(Jm1, 0.4e1) * U2 * U2 * U3 * U1 + 0.6e1 * volond3 * pow(Jm1, 0.5e1) * U2;
        _K[1][0] = _K[0][1];
        _K[1][1] = mu1Volonalpha1 * (pow(Jm13 * U1, alpha1) * alpha1 * alpha1 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha1) * alpha1 * pow(U2, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha1) * alpha1 * alpha1 * pow(-Jm43 * J / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U2, -0.2e1) + pow(Jm13 * U2, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * J * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U2 + pow(Jm13 * U2, alpha1) * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 / U2 * U1 * U3 / 0.3e1 - pow(Jm13 * U2, alpha1) * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * J13 * pow(U2, -0.2e1) + pow(Jm13 * U3, alpha1) * alpha1 * alpha1 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha1) * alpha1 * pow(U2, -0.2e1) / 0.3e1) + mu2Volonalpha2 * (pow(Jm13 * U1, alpha2) * alpha2 * alpha2 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha2) * alpha2 * pow(U2, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha2) * alpha2 * alpha2 * pow(-Jm43 * J / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U2, -0.2e1) + pow(Jm13 * U2, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * J * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U2 + pow(Jm13 * U2, alpha2) * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 / U2 * U1 * U3 / 0.3e1 - pow(Jm13 * U2, alpha2) * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * J13 * pow(U2, -0.2e1) + pow(Jm13 * U3, alpha2) * alpha2 * alpha2 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha2) * alpha2 * pow(U2, -0.2e1) / 0.3e1) + mu3Volonalpha3 * (pow(Jm13 * U1, alpha3) * alpha3 * alpha3 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha3) * alpha3 * pow(U2, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha3) * alpha3 * alpha3 * pow(-Jm43 * J / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U2, -0.2e1) + pow(Jm13 * U2, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * J * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U2 + pow(Jm13 * U2, alpha3) * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 / U2 * U1 * U3 / 0.3e1 - pow(Jm13 * U2, alpha3) * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 * pow(U2, -0.2e1) + pow(Jm13 * U3, alpha3) * alpha3 * alpha3 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha3) * alpha3 * pow(U2, -0.2e1) / 0.3e1) + 0.2e1 * volond1 * U1 * U1 * U3 * U3 + 0.12e2 * volond2 * pow(Jm1, 0.2e1) * U1 * U1 * U3 * U3 + 0.30e2 * volond3 * pow(Jm1, 0.4e1) * U1 * U1 * U3 * U3;
        _K[1][2] = mu1Volonalpha1 * (pow(Jm13 * U1, alpha1) * alpha1 * alpha1 / U3 / U2 / 0.9e1 - pow(Jm13 * U2, alpha1) * alpha1 * alpha1 / U3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U2 / 0.3e1 + pow(Jm13 * U2, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U2 + pow(Jm13 * U2, alpha1) * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 * U1 / 0.3e1 - pow(Jm13 * U3, alpha1) * alpha1 * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U3 / U2 / 0.3e1) + mu2Volonalpha2 * (pow(Jm13 * U1, alpha2) * alpha2 * alpha2 / U3 / U2 / 0.9e1 - pow(Jm13 * U2, alpha2) * alpha2 * alpha2 / U3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U2 / 0.3e1 + pow(Jm13 * U2, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U2 + pow(Jm13 * U2, alpha2) * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 * U1 / 0.3e1 - pow(Jm13 * U3, alpha2) * alpha2 * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U3 / U2 / 0.3e1) + mu3Volonalpha3 * (pow(Jm13 * U1, alpha3) * alpha3 * alpha3 / U3 / U2 / 0.9e1 - pow(Jm13 * U2, alpha3) * alpha3 * alpha3 / U3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U2 / 0.3e1 + pow(Jm13 * U2, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U2 + pow(Jm13 * U2, alpha3) * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 * U1 / 0.3e1 - pow(Jm13 * U3, alpha3) * alpha3 * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 / U3 / U2 / 0.3e1) + 0.2e1 * volond1 * U1 * J + 0.2e1 * volond1 * (Jm1) * U1 + 0.12e2 * volond2 * pow(Jm1, 0.2e1) * U1 * U1 * U3 * U2 + 0.4e1 * volond2 * pow(Jm1, 0.3e1) * U1 + 0.30e2 * volond3 * pow(Jm1, 0.4e1) * U1 * U1 * U3 * U2 + 0.6e1 * volond3 * pow(Jm1, 0.5e1) * U1;
        _K[2][0] = _K[0][2];
        _K[2][1] = _K[1][2];
        _K[2][2] = mu1Volonalpha1 * (pow(Jm13 * U1, alpha1) * alpha1 * alpha1 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha1) * alpha1 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha1) * alpha1 * alpha1 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha1) * alpha1 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha1) * alpha1 * alpha1 * pow(-Jm43 * J / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U3, -0.2e1) + pow(Jm13 * U3, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U3 + pow(Jm13 * U3, alpha1) * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 / U3 * U1 * U2 / 0.3e1 - pow(Jm13 * U3, alpha1) * alpha1 * (-Jm43 * J / 0.3e1 + Jm13) * J13 * pow(U3, -0.2e1)) + mu2Volonalpha2 * (pow(Jm13 * U1, alpha2) * alpha2 * alpha2 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha2) * alpha2 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha2) * alpha2 * alpha2 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha2) * alpha2 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha2) * alpha2 * alpha2 * pow(-Jm43 * J / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U3, -0.2e1) + pow(Jm13 * U3, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U3 + pow(Jm13 * U3, alpha2) * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 / U3 * U1 * U2 / 0.3e1 - pow(Jm13 * U3, alpha2) * alpha2 * (-Jm43 * J / 0.3e1 + Jm13) * J13 * pow(U3, -0.2e1)) + mu3Volonalpha3 * (pow(Jm13 * U1, alpha3) * alpha3 * alpha3 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha3) * alpha3 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha3) * alpha3 * alpha3 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha3) * alpha3 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha3) * alpha3 * alpha3 * pow(-Jm43 * J / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U3, -0.2e1) + pow(Jm13 * U3, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U3 + pow(Jm13 * U3, alpha3) * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * Jm23 / U3 * U1 * U2 / 0.3e1 - pow(Jm13 * U3, alpha3) * alpha3 * (-Jm43 * J / 0.3e1 + Jm13) * J13 * pow(U3, -0.2e1)) + 0.2e1 * volond1 * U1 * U1 * U2 * U2 + 0.12e2 * volond2 * pow(Jm1, 0.2e1) * U1 * U1 * U2 * U2 + 0.30e2 * volond3 * pow(Jm1, 0.4e1) * U1 * U1 * U2 * U2;

    }

    void addDForce( Deriv& df, const Deriv& dx, const double& kfactor, const double& /*bfactor*/ ) const
    {
        df.getStrain() = -_K * dx.getStrain() * kfactor;
    }

    MatBlock getK() const
    {
        return _K;
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
