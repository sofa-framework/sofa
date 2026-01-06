/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#ifndef SOFA_CORE_BEHAVIOR_ROTATIONFINDER_H
#define SOFA_CORE_BEHAVIOR_ROTATIONFINDER_H

#include <sofa/core/behavior/BaseRotationFinder.h>
#include <sofa/type/Mat.h>


namespace sofa::core::behavior
{

template <class DataTypes>
class RotationFinder : public BaseRotationFinder
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RotationFinder, DataTypes), BaseRotationFinder);

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef type::Mat< 3, 3, Real > Mat3x3;

    using BaseRotationFinder::getRotations;
    virtual const type::vector< Mat3x3 >& getRotations() = 0;
};

} // namespace sofa::core::behavior


#endif // SOFA_CORE_BEHAVIOR_ROTATIONFINDER_H
