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
#pragma once
#include <BeamPlastic/config.h>

#include "PlasticConstitutiveLaw.h"
#include <sofa/type/Mat.h>


namespace sofa::plugin::beamplastic::component::constitutivelaw
{

using type::Mat;

template<class DataTypes>
class RambergOsgood : public PlasticConstitutiveLaw<DataTypes> {

public:

    typedef typename DataTypes::Coord::value_type Real;
    typedef Mat<3, 3, Real> Matrix3;
    typedef Mat<6, 6, Real> Matrix6;

    RambergOsgood(Real E, Real yieldStress,
                  Real eps1 = (Real)0.00384942, Real sig1 = (Real)6.0e8,
                  Real eps2 = (Real)0.0030727273, Real sig2 = (Real)5.6e8,
                  unsigned int n = 15, Real A = (Real)0.002)
    {
        // Initialsation of generic material parameters
        _E = E;
        _yieldStress = yieldStress;

        // Initialisation of Ramberg-Osgood parameters
        _A = A;
        _n = n;
        _K = A * pow(E / yieldStress, n);

        // Initialisation of Ramberg-Osgood alternative form parameters
        Real m1 = sig1 / (E*eps1);
        Real m2 = sig2 / (E*eps2);
        _N = 1 + (log((m1 - 1) / (m2 - 1)) / log(eps1 / eps2));
        _altK = (m1 - 1) / pow(eps1,_N - 1);
    }

    Real getTangentModulusFromStress(const double effStress) override
    {
        double En1 = pow(_E, _n - 1);
        Real tangentModulus = (_E*En1) / ( En1 + _K*_n*pow(effStress,_n-1) );
        return tangentModulus;
    }

    Real getTangentModulusFromStrain(const double effPlasticStrain) override
    {
        Real tangentModulus = _E*(1 + _altK*_N*pow(effPlasticStrain,_N-1));
        return tangentModulus;
    }

protected:

    // Material
    Real _E;
    Real _yieldStress;

    // Ramberg-Osgood model as in Ramberg and Osgood, 1943
    Real _A;
    Real _K;
    unsigned int _n;

    // Alternative form of Ramberg-Osgood as in Venkateswara Rao and Krishna Murty, 1971
    Real _altK;
    Real _N;

};


} // namespace sofa::plugin::beamplastic::component::constitutivelaw
