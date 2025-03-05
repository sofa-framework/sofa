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
#pragma once

#include <sofa/component/haptics/config.h>

#include <sofa/simulation/fwd.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/type/Transform.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::haptics
{

/// Base class implementing forcefeedback as a force field
class SOFA_COMPONENT_HAPTICS_API ForceFeedback : public virtual core::behavior::BaseController
{

public:
    SOFA_ABSTRACT_CLASS(ForceFeedback,core::behavior::BaseController);
    Data<bool> d_activate; ///< boolean to activate or deactivate the forcefeedback
    Data<int> d_indice; ///< Tool indice in the OmniDriver

    simulation::Node *context;

    void init() override;

    virtual void computeForce(SReal x, SReal y, SReal z,
                              SReal u, SReal v, SReal w,
                              SReal q, SReal& fx, SReal& fy, SReal& fz) = 0;

    virtual void computeWrench(const sofa::type::Transform<SReal> &,
                               const sofa::type::SpatialVector<SReal> &,
                               sofa::type::SpatialVector<SReal> & )=0;

    virtual void setReferencePosition(sofa::type::Transform<SReal>& referencePosition);
    virtual bool isEnabled();

    /// Abstract method to lock or unlock the force feedback computation. To be implemented by child class if needed
    virtual void setLock(bool value)
    {
        SOFA_UNUSED(value);
    }

protected:
    ForceFeedback();
};

} // namespace sofa::component::haptics
