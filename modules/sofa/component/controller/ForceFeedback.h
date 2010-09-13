/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_CONTROLLER_FORCEFEEDBACK_H
#define SOFA_COMPONENT_CONTROLLER_FORCEFEEDBACK_H

#include <sofa/simulation/common/Node.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/component/component.h>

using namespace sofa::defaulttype;
namespace sofa
{

namespace component
{

namespace controller
{


/**
* Omni driver force field
*/
class SOFA_COMPONENT_CONTROLLER_API ForceFeedback : public core::behavior::BaseController
{

public:
    SOFA_CLASS(ForceFeedback,core::behavior::BaseController);
    Data<bool> f_activate;

    simulation::Node *context;

    ForceFeedback():
        f_activate(initData(&f_activate, false, "activate", "boolean to activate or deactivate the forcefeedback"))
    {
    }

    virtual void init() {context = dynamic_cast<simulation::Node *>(this->getContext());};
    virtual void computeForce(double x, double y, double z, double u, double v, double w, double q, double& fx, double& fy, double& fz) = 0;
    virtual void computeWrench(const SolidTypes<double>::Transform &, const SolidTypes<double>::SpatialVector &, SolidTypes<double>::SpatialVector & )=0;

    virtual void setReferencePosition(SolidTypes<double>::Transform& /*referencePosition*/) {};
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif
