/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CONTROLLER_FORCEFEEDBACK_H
#define SOFA_COMPONENT_CONTROLLER_FORCEFEEDBACK_H
#include "config.h"

#include <sofa/simulation/Node.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace controller
{

/// Base class implementing forcefeedback as a force field
class SOFA_SOFAHAPTICS_API ForceFeedback : public core::behavior::BaseController
{

public:
    SOFA_CLASS(ForceFeedback,core::behavior::BaseController);
    Data<bool> f_activate; ///< boolean to activate or deactivate the forcefeedback
    Data<int> indice; ///< Tool indice in the OmniDriver

    simulation::Node *context;

    void init() override;

    virtual void computeForce(SReal x, SReal y, SReal z,
                              SReal u, SReal v, SReal w,
                              SReal q, SReal& fx, SReal& fy, SReal& fz) = 0;

    virtual void computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &,
                               const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &,
                               sofa::defaulttype::SolidTypes<SReal>::SpatialVector & )=0;

    virtual void setReferencePosition(sofa::defaulttype::SolidTypes<SReal>::Transform& referencePosition);
    virtual bool isEnabled();

protected:
    ForceFeedback();
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif
