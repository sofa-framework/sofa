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
#include <Eigen/Core>

namespace sofa::plugin::beamplastic::component::constitutivelaw
{

template<class DataTypes>
class PlasticConstitutiveLaw
{
public:

    typedef typename DataTypes::Coord Coord;
    typedef typename Coord::value_type Real;
    typedef Eigen::Matrix<double, 6, 1> VoigtTensor;

    virtual ~PlasticConstitutiveLaw() {}

    /* Returns the slope of effective stress VS effective plastic strains, from the stress value */
    virtual Real getTangentModulusFromStress(const double effStress) = 0;

    /* Returns the slope of effective stress VS effective plastic strains, from the strain value*/
    virtual Real getTangentModulusFromStrain(const double effPlasticStrain) = 0;

};

} // namespace sofa::plugin::beamplastic::component::constitutivelaw
