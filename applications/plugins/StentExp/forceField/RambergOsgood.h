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
#ifndef SOFA_COMPONENT_FEM_RAMBERGOSGOOD_H
#define SOFA_COMPONENT_FEM_RAMBERGOSGOOD_H
#include "../config.h"

#include "PlasticConstitutiveLaw.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>


namespace sofa
{

namespace component
{

namespace fem
{

template<class DataTypes>
class RambergOsgood : public PlasticConstitutiveLaw<DataTypes> {

public:

    typedef typename DataTypes::Coord::value_type Real;
    typedef defaulttype::Mat<3, 3, Real> Matrix3;
    typedef defaulttype::Mat<6, 6, Real> Matrix6;

    RambergOsgood(Real E, Real yieldStress, unsigned int n=15, Real A=(Real)0.002)
    {
        _A = A;
        _n = n;
        _E = E;
        _yieldStress = yieldStress;

        // Computing _K
        _K = A * pow(E / yieldStress, n);
    }

    virtual Real getTangentModulus(const double yieldStress) {

        double En1 = pow(_E, _n - 1);
        Real tangentModulus = (_E*En1) / ( En1 + _K*_n*pow(yieldStress,_n-1) );
        return tangentModulus;
    }

protected:

    Real _A;
    Real _K;
    unsigned int _n;
    Real _E;
    Real _yieldStress;

};


} // namespace fem

} // namespace component

} // namespace sofa

#endif // ifndef SOFA_COMPONENT_FEM_RAMBERGOSGOOD_H