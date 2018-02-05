/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_Elasticity_test_deprecated_H
#define SOFA_Elasticity_test_deprecated_H


#include "Elasticity_test.h"

namespace sofa {

template< class DataTypes>
struct SOFA_SOFATEST_API Elasticity_test_deprecated: public Elasticity_test<DataTypes>
{
    typedef component::container::MechanicalObject<DataTypes> DOFs;
    typedef typename DOFs::Coord  Coord;
    typedef typename DOFs::Deriv  Deriv;

/// Create sun-planet system
simulation::Node::SPtr createSunPlanetSystem(
        simulation::Node::SPtr root,
        double mSun,
        double mPlanet,
        double g,
        Coord xSun,
        Deriv vSun,
        Coord xPlanet,
        Deriv vPlanet);

};




} // namespace sofa

#endif
