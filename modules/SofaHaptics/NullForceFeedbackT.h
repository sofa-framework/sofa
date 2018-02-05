/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_CONTROLLER_NULLFORCEFEEDBACKT_H
#define SOFA_COMPONENT_CONTROLLER_NULLFORCEFEEDBACKT_H
#include "config.h"

#include <SofaHaptics/MechanicalStateForceFeedback.h>

namespace sofa
{

namespace component
{

namespace controller
{

/**
* Device driver force field
*/
template<class TDataTypes>
class SOFA_HAPTICS_API NullForceFeedbackT : public sofa::component::controller::MechanicalStateForceFeedback<TDataTypes>
{
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

public:
    SOFA_CLASS(SOFA_TEMPLATE(NullForceFeedbackT,TDataTypes),sofa::component::controller::MechanicalStateForceFeedback<TDataTypes>);
    void init() override {this->ForceFeedback::init();};

    virtual void computeForce(SReal, SReal, SReal, SReal, SReal, SReal, SReal, SReal& fx, SReal& fy, SReal& fz) override
    {
        fx = fy = fz = 0.0;
    };
    virtual void computeForce(const  VecCoord &,  VecDeriv &) override {};
    virtual void computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &, const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &, sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world ) override {W_tool_world.clear();};


};

} // namespace controller

} // namespace component

} // namespace sofa

#endif
