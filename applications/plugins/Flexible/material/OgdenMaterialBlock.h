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
#ifndef FLEXIBLE_OgdenMaterialBlock_INL
#define FLEXIBLE_OgdenMaterialBlock_INL

#include "OgdenMaterialBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"
#include "../material/BaseMaterial.h"
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

    static const bool constantK=false;

    Real mu1Vol, mu2Vol, mu3Vol, alpha1, alpha2, alpha3, volond1, volond2, volond3;
    bool stabilization;

    mutable MatBlock _K;

    void init( Real mu1, Real mu2, Real mu3, Real _alpha1, Real _alpha2, Real _alpha3, Real d1, Real d2, Real d3, bool _stabilization )
    {
        alpha1=_alpha1;
        alpha2=_alpha2;
        alpha3=_alpha3;
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        mu1Vol = mu1*vol;
        mu2Vol = mu2*vol;
        mu3Vol = mu3*vol;
        volond1 = vol/d1;
        volond2 = vol/d2;
        volond3 = vol/d3;

        stabilization = _stabilization;
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        Real J = x.getStrain()[0]*x.getStrain()[1]*x.getStrain()[2];
        Real Jm1 = J-1;
        Real squareJm1 = Jm1*Jm1;
        Real fourJm1 = squareJm1*squareJm1;
        Real Jm13 = pow(J,-1.0/3.0);

        Real devU[3] = { Jm13*x.getStrain()[0], Jm13*x.getStrain()[1], Jm13*x.getStrain()[2] };

        return mu1Vol/alpha1 * ( pow(devU[0],alpha1)+pow(devU[1],alpha1)+pow(devU[2],alpha1) - 3 ) +
               mu2Vol/alpha1 * ( pow(devU[0],alpha2)+pow(devU[1],alpha2)+pow(devU[2],alpha2) - 3 ) +
               mu3Vol/alpha1 * ( pow(devU[0],alpha3)+pow(devU[1],alpha3)+pow(devU[2],alpha3) - 3 ) +
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

        Real J = t1 * U3;
        Real Jm1 = J-1;
        Real squareJm1 = Jm1*Jm1;
        Real cubeJm1 = squareJm1*Jm1;
        Real fourJm1 = squareJm1*squareJm1;
        Real fiveJm1 = fourJm1*Jm1;

        Real J13 = pow(J,1.0/3.0);
        Real Jm13 = 1.0/J13;
        Real J23 = pow(J,2.0/3.0);
        Real Jm23 = pow(J,-2.0/3.0);
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
        Real t15 = (2 * volond1) + ((4 * volond2) + 0.6e1 * volond3 * squareJm1) * squareJm1;
        Real t16 = Jm1 * U3;
        Real t17 = 0.1e1 / U2;
        Real t18 = 0.1e1 / U3;
        f.getStrain()[0] -= t16 * U2 * t15 + mu1Vol * t11 * (-t9 / 0.3e1 - t10 / 0.3e1 + t5 * t6 * t7) + mu2Vol * t11 * (-t13 / 0.3e1 - t14 / 0.3e1 + t12 * t6 * t7) + mu3Vol * t11 * (-t8 / 0.3e1 - t3 / 0.3e1 + t4 * t6 * t7);
        f.getStrain()[1] -= t16 * U1 * t15 + mu1Vol * t17 * (-t5 / 0.3e1 - t10 / 0.3e1 + t9 * t6 * t7) + mu2Vol * t17 * (-t12 / 0.3e1 - t14 / 0.3e1 + t13 * t6 * t7) + mu3Vol * t17 * (-t4 / 0.3e1 - t3 / 0.3e1 + t8 * t6 * t7);
        f.getStrain()[2] -= t1 * Jm1 * t15 + mu1Vol * t18 * (-t5 / 0.3e1 - t9 / 0.3e1 + t10 * t6 * t7) + mu2Vol * t18 * (-t12 / 0.3e1 - t13 / 0.3e1 + t14 * t6 * t7) + mu3Vol * t18 * (-t4 / 0.3e1 - t8 / 0.3e1 + t3 * t6 * t7);

        _K[0][0] = mu1Vol * (pow(Jm13 * U1, alpha1) * alpha1 * pow(-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U1, -0.2e1) + pow(Jm13 * U1, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * U2 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U2 * U3) * J13 / U1 + pow(Jm13 * U1, alpha1) * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 / U1 * U2 * U3 / 0.3e1 - pow(Jm13 * U1, alpha1) * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 * pow(U1, -0.2e1) + pow(Jm13 * U2, alpha1) * alpha1 * alpha1 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha1) * alpha1 * pow(U1, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha1) * alpha1 * alpha1 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha1) * alpha1 * pow(U1, -0.2e1) / 0.3e1) + mu2Vol * (pow(Jm13 * U1, alpha2) * alpha2 * pow(-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U1, -0.2e1) + pow(Jm13 * U1, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * U2 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U2 * U3) * J13 / U1 + pow(Jm13 * U1, alpha2) * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 / U1 * U2 * U3 / 0.3e1 - pow(Jm13 * U1, alpha2) * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 * pow(U1, -0.2e1) + pow(Jm13 * U2, alpha2) * alpha2 * alpha2 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha2) * alpha2 * pow(U1, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha2) * alpha2 * alpha2 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha2) * alpha2 * pow(U1, -0.2e1) / 0.3e1) + mu3Vol * (pow(Jm13 * U1, alpha3) * alpha3 * pow(-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U1, -0.2e1) + pow(Jm13 * U1, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * U2 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U2 * U3) * J13 / U1 + pow(Jm13 * U1, alpha3) * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 / U1 * U2 * U3 / 0.3e1 - pow(Jm13 * U1, alpha3) * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 * pow(U1, -0.2e1) + pow(Jm13 * U2, alpha3) * alpha3 * alpha3 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha3) * alpha3 * pow(U1, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha3) * alpha3 * alpha3 * pow(U1, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha3) * alpha3 * pow(U1, -0.2e1) / 0.3e1) + 0.2e1 * volond1 * U2 * U2 * U3 * U3 + 0.12e2 * volond2 * squareJm1 * U2 * U2 * U3 * U3 + 0.30e2 * volond3 * fourJm1 * U2 * U2 * U3 * U3;
        _K[0][1] = mu1Vol * (-pow(Jm13 * U1, alpha1) * alpha1 / U2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U1 + pow(Jm13 * U1, alpha1) * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 * U3 / 0.3e1 - pow(Jm13 * U2, alpha1) * alpha1 * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U2 / U1 / 0.3e1 + pow(Jm13 * U3, alpha1) * alpha1 * alpha1 / U2 / U1 / 0.9e1) + mu2Vol * (-pow(Jm13 * U1, alpha2) * alpha2 / U2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U1 + pow(Jm13 * U1, alpha2) * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 * U3 / 0.3e1 - pow(Jm13 * U2, alpha2) * alpha2 * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U2 / U1 / 0.3e1 + pow(Jm13 * U3, alpha2) * alpha2 * alpha2 / U2 / U1 / 0.9e1) + mu3Vol * (-pow(Jm13 * U1, alpha3) * alpha3 / U2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U1 + pow(Jm13 * U1, alpha3) * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 * U3 / 0.3e1 - pow(Jm13 * U2, alpha3) * alpha3 * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U2 / U1 / 0.3e1 + pow(Jm13 * U3, alpha3) * alpha3 * alpha3 / U2 / U1 / 0.9e1) + 0.2e1 * volond1 * U1 * U3 * U3 * U2 + 0.2e1 * volond1 * (U1 * U2 * U3 - 0.1e1) * U3 + 0.12e2 * volond2 * squareJm1 * U2 * U3 * U3 * U1 + 0.4e1 * volond2 * cubeJm1 * U3 + 0.30e2 * volond3 * fourJm1 * U2 * U3 * U3 * U1 + 0.6e1 * volond3 * fiveJm1 * U3;
        _K[0][2] = mu1Vol * (-pow(Jm13 * U1, alpha1) * alpha1 / U3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U1 + pow(Jm13 * U1, alpha1) * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 * U2 / 0.3e1 + pow(Jm13 * U2, alpha1) * alpha1 * alpha1 / U3 / U1 / 0.9e1 - pow(Jm13 * U3, alpha1) * alpha1 * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U3 / U1 / 0.3e1) + mu2Vol * (-pow(Jm13 * U1, alpha2) * alpha2 / U3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U1 + pow(Jm13 * U1, alpha2) * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 * U2 / 0.3e1 + pow(Jm13 * U2, alpha2) * alpha2 * alpha2 / U3 / U1 / 0.9e1 - pow(Jm13 * U3, alpha2) * alpha2 * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U3 / U1 / 0.3e1) + mu3Vol * (-pow(Jm13 * U1, alpha3) * alpha3 / U3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U1 / 0.3e1 + pow(Jm13 * U1, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U1 + pow(Jm13 * U1, alpha3) * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 * U2 / 0.3e1 + pow(Jm13 * U2, alpha3) * alpha3 * alpha3 / U3 / U1 / 0.9e1 - pow(Jm13 * U3, alpha3) * alpha3 * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U3 / U1 / 0.3e1) + 0.2e1 * volond1 * U1 * U2 * U2 * U3 + 0.2e1 * volond1 * (U1 * U2 * U3 - 0.1e1) * U2 + 0.12e2 * volond2 * squareJm1 * U2 * U2 * U3 * U1 + 0.4e1 * volond2 * cubeJm1 * U2 + 0.30e2 * volond3 * fourJm1 * U2 * U2 * U3 * U1 + 0.6e1 * volond3 * fiveJm1 * U2;
        _K[1][0] = _K[0][1];
        _K[1][1] = mu1Vol * (pow(Jm13 * U1, alpha1) * alpha1 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha1) * alpha1 * pow(U2, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha1) * alpha1 * alpha1 * pow(-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U2, -0.2e1) + pow(Jm13 * U2, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U2 + pow(Jm13 * U2, alpha1) * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 / U2 * U1 * U3 / 0.3e1 - pow(Jm13 * U2, alpha1) * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 * pow(U2, -0.2e1) + pow(Jm13 * U3, alpha1) * alpha1 * alpha1 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha1) * alpha1 * pow(U2, -0.2e1) / 0.3e1) + mu2Vol * (pow(Jm13 * U1, alpha2) * alpha2 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha2) * alpha2 * pow(U2, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha2) * alpha2 * alpha2 * pow(-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U2, -0.2e1) + pow(Jm13 * U2, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U2 + pow(Jm13 * U2, alpha2) * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 / U2 * U1 * U3 / 0.3e1 - pow(Jm13 * U2, alpha2) * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 * pow(U2, -0.2e1) + pow(Jm13 * U3, alpha2) * alpha2 * alpha2 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha2) * alpha2 * pow(U2, -0.2e1) / 0.3e1) + mu3Vol * (pow(Jm13 * U1, alpha3) * alpha3 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha3) * alpha3 * pow(U2, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha3) * alpha3 * alpha3 * pow(-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U2, -0.2e1) + pow(Jm13 * U2, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U3 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U3) * J13 / U2 + pow(Jm13 * U2, alpha3) * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 / U2 * U1 * U3 / 0.3e1 - pow(Jm13 * U2, alpha3) * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 * pow(U2, -0.2e1) + pow(Jm13 * U3, alpha3) * alpha3 * alpha3 * pow(U2, -0.2e1) / 0.9e1 + pow(Jm13 * U3, alpha3) * alpha3 * pow(U2, -0.2e1) / 0.3e1) + 0.2e1 * volond1 * U1 * U1 * U3 * U3 + 0.12e2 * volond2 * squareJm1 * U1 * U1 * U3 * U3 + 0.30e2 * volond3 * fourJm1 * U1 * U1 * U3 * U3;
        _K[1][2] = mu1Vol * (pow(Jm13 * U1, alpha1) * alpha1 / U3 / U2 / 0.9e1 - pow(Jm13 * U2, alpha1) * alpha1 * alpha1 / U3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U2 / 0.3e1 + pow(Jm13 * U2, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U2 + pow(Jm13 * U2, alpha1) * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 * U1 / 0.3e1 - pow(Jm13 * U3, alpha1) * alpha1 * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U3 / U2 / 0.3e1) + mu2Vol * (pow(Jm13 * U1, alpha2) * alpha2 / U3 / U2 / 0.9e1 - pow(Jm13 * U2, alpha2) * alpha2 * alpha2 / U3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U2 / 0.3e1 + pow(Jm13 * U2, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U2 + pow(Jm13 * U2, alpha2) * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 * U1 / 0.3e1 - pow(Jm13 * U3, alpha2) * alpha2 * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U3 / U2 / 0.3e1) + mu3Vol * (pow(Jm13 * U1, alpha3) * alpha3 / U3 / U2 / 0.9e1 - pow(Jm13 * U2, alpha3) * alpha3 * alpha3 / U3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U2 / 0.3e1 + pow(Jm13 * U2, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U2 + pow(Jm13 * U2, alpha3) * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 * U1 / 0.3e1 - pow(Jm13 * U3, alpha3) * alpha3 * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 / U3 / U2 / 0.3e1) + 0.2e1 * volond1 * U1 * U1 * U2 * U3 + 0.2e1 * volond1 * (U1 * U2 * U3 - 0.1e1) * U1 + 0.12e2 * volond2 * squareJm1 * U1 * U1 * U3 * U2 + 0.4e1 * volond2 * cubeJm1 * U1 + 0.30e2 * volond3 * fourJm1 * U1 * U1 * U3 * U2 + 0.6e1 * volond3 * fiveJm1 * U1;
        _K[2][0] = _K[0][2];
        _K[2][1] = _K[1][2];
        _K[2][2] = mu1Vol * (pow(Jm13 * U1, alpha1) * alpha1 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha1) * alpha1 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha1) * alpha1 * alpha1 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha1) * alpha1 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha1) * alpha1 * alpha1 * pow(-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U3, -0.2e1) + pow(Jm13 * U3, alpha1) * alpha1 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U3 + pow(Jm13 * U3, alpha1) * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 / U3 * U1 * U2 / 0.3e1 - pow(Jm13 * U3, alpha1) * alpha1 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 * pow(U3, -0.2e1)) + mu2Vol * (pow(Jm13 * U1, alpha2) * alpha2 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha2) * alpha2 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha2) * alpha2 * alpha2 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha2) * alpha2 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha2) * alpha2 * alpha2 * pow(-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U3, -0.2e1) + pow(Jm13 * U3, alpha2) * alpha2 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U3 + pow(Jm13 * U3, alpha2) * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 / U3 * U1 * U2 / 0.3e1 - pow(Jm13 * U3, alpha2) * alpha2 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 * pow(U3, -0.2e1)) + mu3Vol * (pow(Jm13 * U1, alpha3) * alpha3 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U1, alpha3) * alpha3 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U2, alpha3) * alpha3 * alpha3 * pow(U3, -0.2e1) / 0.9e1 + pow(Jm13 * U2, alpha3) * alpha3 * pow(U3, -0.2e1) / 0.3e1 + pow(Jm13 * U3, alpha3) * alpha3 * alpha3 * pow(-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13, 0.2e1) * J23 * pow(U3, -0.2e1) + pow(Jm13 * U3, alpha3) * alpha3 * (0.4e1 / 0.9e1 * Jm73 * U1 * U1 * U2 * U2 * U3 - 0.2e1 / 0.3e1 * Jm43 * U1 * U2) * J13 / U3 + pow(Jm13 * U3, alpha3) * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * Jm23 / U3 * U1 * U2 / 0.3e1 - pow(Jm13 * U3, alpha3) * alpha3 * (-Jm43 * U1 * U2 * U3 / 0.3e1 + Jm13) * J13 * pow(U3, -0.2e1)) + 0.2e1 * volond1 * U1 * U1 * U2 * U2 + 0.12e2 * volond2 * squareJm1 * U1 * U1 * U2 * U2 + 0.30e2 * volond3 * fourJm1 * U1 * U1 * U2 * U2;

        /// ensure _K is symmetric positive semi-definite (even if it is not as good as positive definite) as suggested in [Teran05]
        if( stabilization ) helper::Decompose<Real>::PSDProjection( _K );

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
////  U321
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class OgdenMaterialBlock< U321(_Real) > :
    public BaseMaterialBlock< U321(_Real) >
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
      * classic Ogden
      *     - W = sum(1<=i=<N) mui/alphai (~U1^alphai+~U2^alphai-2) + sum(1<=i=<N) 1/di(J-1)^{2i}
      * with       ~Ui=J^{-1/3}Ui  deviatoric principal stretches
      * see maple file ./doc/Ogden_principalStretches.mw for derivative
      */

    static const bool constantK=false;

    Real mu1Vol, mu2Vol, mu3Vol, alpha1, alpha2, alpha3, volond1, volond2, volond3;
    bool stabilization;

    mutable MatBlock _K;

    void init( Real mu1, Real mu2, Real mu3, Real _alpha1, Real _alpha2, Real _alpha3, Real d1, Real d2, Real d3, bool _stabilization )
    {
        alpha1=_alpha1;
        alpha2=_alpha2;
        alpha3=_alpha3;
        Real vol=1.;
        if(this->volume) vol=(*this->volume)[0];
        mu1Vol = mu1*vol;
        mu2Vol = mu2*vol;
        mu3Vol = mu3*vol;
        volond1 = vol/d1;
        volond2 = vol/d2;
        volond3 = vol/d3;

        stabilization = _stabilization;
    }

    Real getPotentialEnergy(const Coord& x) const
    {
        Real J = x.getStrain()[0]*x.getStrain()[1];
        Real Jm1 = J-1;
        Real squareJm1 = Jm1*Jm1;
        Real fourJm1 = squareJm1*squareJm1;
        Real Jm13 = pow(J,-1.0/3.0);

        Real devU[2] = { Jm13*x.getStrain()[0], Jm13*x.getStrain()[1] };

        return mu1Vol/alpha1 * ( pow(devU[0],alpha1)+pow(devU[1],alpha1) - 2 ) +
               mu2Vol/alpha2 * ( pow(devU[0],alpha2)+pow(devU[1],alpha2) - 2 ) +
               mu3Vol/alpha3 * ( pow(devU[0],alpha3)+pow(devU[1],alpha3) - 2 ) +
               volond1 * squareJm1 +
               volond2 * fourJm1 +
               volond3 * squareJm1*fourJm1;
    }

    void addForce( Deriv& f, const Coord& x, const Deriv& /*v*/) const
    {
        const Real& U1 = x.getStrain()[0];
        const Real& U2 = x.getStrain()[1];

        // TODO optimize this REALLY crappy code generated from maple
        // there are a LOT of redondencies

        Real J = U1 * U2;
        Real Jm1 = J-1;
        Real squareJm1 = Jm1*Jm1;

        Real J13 = pow(J,1.0/3.0);
        Real Jm13 = 1.0/J13;
        Real J23 = pow(J,2.0/3.0);
        Real Jm23 = pow(J, -2.0/3.0); // note that taking 1.0/J23 is loosing too much precision
        Real Jm43 = pow(J,-4.0/3.0);


        Real t2 = Jm13;
        Real t3 = t2 * U1;
        Real t4 = pow(t3, alpha1);
        Real t5 = -Jm13 / 0.3e1 + t2;
        Real t6 = J13;
        Real t7 = t2 * U2;
        Real t8 = pow(t7, alpha1);
        Real t9 = pow(t2, alpha1);
        Real t10 = 0.1e1 / U1;
        Real t11 = pow(t3, alpha2);
        Real t12 = pow(t7, alpha2);
        Real t13 = pow(t2, alpha2);
        t3 = pow(t3, alpha3);
        t7 = pow(t7, alpha3);
        t2 = pow(t2, alpha3);
        Real t14 =  (2 * volond1) + ( (4 * volond2) + 0.6e1 * volond3 * squareJm1) * squareJm1;
        Real t18 = 0.1e1 / U2;
        f.getStrain()[0] -= Jm1 * U2 * t14 + mu1Vol * t10 * (-t8 / 0.3e1 - t9 / 0.3e1 + t4 * t5 * t6) + mu2Vol * t10 * (-t12 / 0.3e1 - t13 / 0.3e1 + t11 * t5 * t6) + mu3Vol * t10 * (-t7 / 0.3e1 - t2 / 0.3e1 + t3 * t5 * t6);
        f.getStrain()[0] -= Jm1 * U1 * t14 + mu1Vol * t18 * (-t4 / 0.3e1 - t9 / 0.3e1 + t8 * t5 * t6) + mu2Vol * t18 * (-t11 / 0.3e1 - t13 / 0.3e1 + t12 * t5 * t6) + mu3Vol * t18 * (-t3 / 0.3e1 - t2 / 0.3e1 + t7 * t5 * t6);



        t3 = Jm13 * U1;
        t4 = pow(t3, alpha1);
        t6 = -J * Jm43 / 0.3e1 + Jm13;
        t5 = 0.4e1 / 0.9e1 * Jm43 - 0.2e1 / 0.3e1 * Jm43;
        t10 = Jm13 * U2;
        t11 = pow(t10, alpha1);
        t12 = pow(Jm13, alpha1);
        t13 = 0.1e1 / U1;
        t14 = t11 + t12;
        Real t15 = J / 0.3e1;
        Real t16 = t15 * U2;
        Real t17 = J13 * t13;
        t18 = t13 * alpha1;
        Real t19 = U2 * t5 * J13;
        Real t20 = t13 / 0.3e1;
        Real t21 = pow(t3, alpha2);
        Real t22 = pow(t10, alpha2);
        Real t23 = pow(Jm13, alpha2);
        Real t24 = t22 + t23;
        Real  t25 = t13 * alpha2;
        t3 = pow(t3, alpha3);
        t10 = pow(t10, alpha3);
        t2 = pow(Jm13, alpha3);
        Real  t26 = t10 + t2;
        Real t27 = t13 * alpha3;
        Real t30 = 2 * volond1;
        Real t31 =  t30 + ( (12 * volond2) + 0.30e2 * volond3 * squareJm1) * squareJm1;
        Real t35 = 0.1e1 / U2;
        Real t36 = J13 * t35;
        t5 = U1 * t5 * J13;

        _K[0][1] = J * t31 + ( t30 + ( (4 * volond2) + 0.6e1 * volond3 * squareJm1) * squareJm1) * Jm1 + mu1Vol * ((t36 * t18 * (-t11 - t4) / 0.3e1 + t4 * Jm23 / 0.3e1) * t6 + t13 * (t12 * alpha1 * t35 / 0.9e1 + t5 * t4)) + mu2Vol * ((t36 * t25 * (-t22 - t21) / 0.3e1 + t21 * Jm23 / 0.3e1) * t6 + t13 * (t23 * alpha2 * t35 / 0.9e1 + t5 * t21)) + mu3Vol * ((t36 * t27 * (-t10 - t3) / 0.3e1 + t3 * Jm23 / 0.3e1) * t6 + t13 * (t2 * alpha3 * t35 / 0.9e1 + t5 * t3));

        t8 = t4 + t12;
        t9 = t15 * U1;
        t12 = t35 * alpha1;
        t15 = t21 + t23;
        t23 = t35 * alpha2;
        t2 = t3 + t2;
        Real t28 = t35 * alpha3;
        Real t200 = t35 / 0.3e1;

        _K[0][0] = U2 * U2 * t31 + mu1Vol * t13 * ((t19 + (t16 - t17 + t18 * J23 * t6) * t6) * t4 + t18 * t14 / 0.9e1 + t20 * t14) + mu2Vol * t13 * ((t19 + (t16 - t17 + t25 * J23 * t6) * t6) * t21 + t25 * t24 / 0.9e1 + t20 * t24) + mu3Vol * t13 * ((t19 + (t16 - t17 + t27 * J23 * t6) * t6) * t3 + t27 * t26 / 0.9e1 + t20 * t26);
        _K[1][0] = _K[0][1];
        _K[1][1] = U1 * U1 * t31 + mu1Vol * t35 * ((t5 + (t9 - t36 + t12 * J23 * t6) * t6) * t11 + t12 * t8 / 0.9e1 + t8 * t200) + mu2Vol * t35 * ((t5 + (t9 - t36 + t23 * J23 * t6) * t6) * t22 + t23 * t15 / 0.9e1 + t200 * t15) + mu3Vol * t35 * ((t5 + (t9 - t36 + t28 * J23 * t6) * t6) * t10 + t28 * t2 / 0.9e1 + t200 * t2);


        // ensure _K is symmetric positive semi-definite (even if it is not as good as positive definite) as suggested in [Teran05]
        if( stabilization ) helper::Decompose<Real>::PSDProjection( _K );

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
