/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_CONTROLLER_MECHANICALSTATEFORCEFEEDBACK_H
#define SOFA_COMPONENT_CONTROLLER_MECHANICALSTATEFORCEFEEDBACK_H

#include <sofa/component/component.h>
#include <sofa/component/controller/ForceFeedback.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/constraintset/ConstraintSolverImpl.h>

namespace sofa
{

namespace component
{

namespace controller
{

using namespace std;
using namespace helper::system::thread;
using namespace core::behavior;
using namespace core;


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
    virtual void init() {context = dynamic_cast<simulation::Node *>(this->getContext());};
    virtual void computeForce(SReal x, SReal y, SReal z, SReal u, SReal v, SReal w, SReal q, SReal& fx, SReal& fy, SReal& fz) = 0;
    virtual void computeForce(const  VecCoord& state,  VecDeriv& forces) = 0;
    virtual void computeWrench(const SolidTypes<SReal>::Transform &, const SolidTypes<SReal>::SpatialVector &, SolidTypes<SReal>::SpatialVector & )=0;

    virtual void setReferencePosition(SolidTypes<SReal>::Transform& /*referencePosition*/) {};
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif
