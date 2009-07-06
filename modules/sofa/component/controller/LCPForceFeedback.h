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
#ifndef SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_H
#define SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_H

#include <sofa/component/controller/ForceFeedback.h>
#include <sofa/component/container/MechanicalObject.h>

namespace sofa
{

namespace component
{
namespace odesolver
{
class MasterContactSolver;
class LCP;
}

namespace controller
{
using namespace std;

/**
* Omni driver force field
*/
class LCPForceFeedback : public sofa::component::controller::ForceFeedback
{

public:
    typedef defaulttype::SparseConstraint<defaulttype::RigidTypes::Deriv> SparseConstraint;
    typedef SparseConstraint::const_data_iterator ConstraintIterator;

    void init();
    Data<double> forceCoef;
    virtual void computeForce(double x, double y, double z, double u, double v, double w, double q, double& fx, double& fy, double& fz);

    LCPForceFeedback()
        : forceCoef(initData(&forceCoef, 0.03, "forceCoef","multiply haptic force by this coef."))
    {}

protected:
    component::odesolver::LCP* lcp;
    core::componentmodel::behavior::MechanicalState<defaulttype::Rigid3Types> *mState; ///< The omni try to follow this mechanical state.
    sofa::component::odesolver::MasterContactSolver* mastersolver;
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif
