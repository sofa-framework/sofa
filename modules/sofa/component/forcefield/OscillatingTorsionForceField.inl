/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_OSCILLATINGTORSIONFORCEFIELD_INL
#define SOFA_COMPONENT_OSCILLATINGTORSIONFORCEFIELD_INL

#include "OscillatingTorsionForceField.h"
#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
//#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/simulation/common/Simulation.h>




namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
OscillatingTorsionForceField<DataTypes>::OscillatingTorsionForceField()
    : axis(initData(&axis, "axis", "axis of rotation"))
    , force(initData(&force, "force", "applied moment force to rigid object around center"))
    , frequency(initData(&frequency, "frequency", "frequency of oscillation"))
    , arrowSizeCoef(initData(&arrowSizeCoef, 0.0f, "arrowSizeCoef", "Size of the drawn arrows (0->no arrows, sign->direction of drawing"))
{

}


template<class DataTypes>
double OscillatingTorsionForceField<DataTypes>::getDynamicAngle()
{
    double t = this->getContext()->getTime();
    double angle = cos( 6.2831853 * frequency.getValue() * t );
    return angle;
}


template<class DataTypes>
void OscillatingTorsionForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv&)
{
    f1.resize( 1 );  // class is only defined for rigid objects
    Quat quat( axis.getValue(), this->getDynamicAngle() );
    f1[0].getAngular() += quat.toEulerVector() * force.getValue();
}


template<class DataTypes>
void OscillatingTorsionForceField<DataTypes>::addDForce (VecDeriv &df, const VecDeriv &dx )
{
    // force does not depend on relative position -> do not modify df
}


template <class DataTypes>
double OscillatingTorsionForceField<DataTypes>::getPotentialEnergy(const VecCoord& x)
{
    serr << "OscillatingTorsionForceField::getPotentialEnergy() not implemented !!!" << sendl;
    return 0;
}



template<class DataTypes>
void OscillatingTorsionForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
}



} // namespace forcefield

} // namespace component

} // namespace sofa

#endif



/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_OSCILLATINGTORSIONFORCEFIELD_INL
#define SOFA_COMPONENT_OSCILLATINGTORSIONFORCEFIELD_INL

#include "OscillatingTorsionForceField.h"
#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
//#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/simulation/common/Simulation.h>




namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
OscillatingTorsionForceField<DataTypes>::OscillatingTorsionForceField()
    : axis(initData(&axis, "axis", "axis of rotation"))
    , force(initData(&force, "force", "applied moment force to rigid object around center"))
    , frequency(initData(&frequency, "frequency", "frequency of oscillation"))
    , arrowSizeCoef(initData(&arrowSizeCoef, 0.0f, "arrowSizeCoef", "Size of the drawn arrows (0->no arrows, sign->direction of drawing"))
{

}


template<class DataTypes>
double OscillatingTorsionForceField<DataTypes>::getDynamicAngle()
{
    double t = this->getContext()->getTime();
    double angle = cos( 6.2831853 * frequency.getValue() * t );
    return angle;
}


template<class DataTypes>
void OscillatingTorsionForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv&)
{
    f1.resize( 1 );  // class is only defined for rigid objects
    Quat quat( axis.getValue(), this->getDynamicAngle() );
    f1[0].getAngular() += quat.toEulerVector() * force.getValue();
}


template<class DataTypes>
void OscillatingTorsionForceField<DataTypes>::addDForce (VecDeriv &df, const VecDeriv &dx )
{
    // force does not depend on relative position -> do not modify df
}


template <class DataTypes>
double OscillatingTorsionForceField<DataTypes>::getPotentialEnergy(const VecCoord& x)
{
    serr << "OscillatingTorsionForceField::getPotentialEnergy() not implemented !!!" << sendl;
    return 0;
}



template<class DataTypes>
void OscillatingTorsionForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
}



} // namespace forcefield

} // namespace component

} // namespace sofa

#endif



