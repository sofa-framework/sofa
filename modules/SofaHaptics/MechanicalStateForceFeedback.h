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
#ifndef SOFA_COMPONENT_CONTROLLER_MECHANICALSTATEFORCEFEEDBACK_H
#define SOFA_COMPONENT_CONTROLLER_MECHANICALSTATEFORCEFEEDBACK_H
#include "config.h"

#include <SofaHaptics/ForceFeedback.h>



namespace sofa
{

namespace component
{

namespace controller
{

template<class TDataTypes>
class SOFA_HAPTICS_API MechanicalStateForceFeedback : public sofa::component::controller::ForceFeedback
{

public:

    SOFA_CLASS(SOFA_TEMPLATE(MechanicalStateForceFeedback,TDataTypes),sofa::component::controller::ForceFeedback);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    simulation::Node *context;
protected:
    MechanicalStateForceFeedback(void) {};
public:
    virtual void init() override {context = dynamic_cast<simulation::Node *>(this->getContext());};
    virtual void computeForce(SReal x, SReal y, SReal z, SReal u, SReal v, SReal w, SReal q, SReal& fx, SReal& fy, SReal& fz) override = 0;
    virtual void computeForce(const  VecCoord& state,  VecDeriv& forces) = 0;
    virtual void computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &, const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &, sofa::defaulttype::SolidTypes<SReal>::SpatialVector & ) override = 0;

    virtual void setReferencePosition(sofa::defaulttype::SolidTypes<SReal>::Transform& /*referencePosition*/) override {};
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif
