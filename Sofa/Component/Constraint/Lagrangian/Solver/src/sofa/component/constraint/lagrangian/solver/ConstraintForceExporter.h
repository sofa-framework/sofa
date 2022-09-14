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
#include <sofa/component/constraint/lagrangian/solver/config.h>

#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/core/fwd.h>

namespace sofa::component::constraint::lagrangian::solver
{

/**
* Small utility component to compute forces (and wrenches in case of rigids)
* from constraints computed by a constraint solver on a mechanical state,
* at the end of each animation step.
* Can also draw the forces for debug purpose (but it does not draw the wrenches if applicable).
*/
template <typename DataTypes>
class ConstraintForceExporter : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(ConstraintForceExporter, sofa::core::objectmodel::BaseObject);

    using Inherited = sofa::core::objectmodel::BaseObject;
    using Deriv = typename DataTypes::Deriv;
    using VecDeriv = typename DataTypes::VecDeriv;

protected:
    ConstraintForceExporter();
    ~ConstraintForceExporter() override = default;
public:
    void init() override;
    void handleEvent(sofa::core::objectmodel::Event* event) override;
    void draw(const sofa::core::visual::VisualParams* vparams) override;

    Data<VecDeriv> d_constraintForces; ///< (output) Forces (or wrenches if rigids) computed from the mechanical state using the constraint solver.
    Data<bool> d_draw; ///< (debug) draw forces (as an arrow) for each position of the mechanical object.
    Data<double> d_drawForceScale; ///< (debug) Scale to apply on the force (draw as an arrow).

    SingleLink<ConstraintForceExporter<DataTypes>, sofa::core::behavior::MechanicalState<DataTypes>, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_mechanicalState; ///< Link to the mechanical state storing the constraint matrix.
    SingleLink<ConstraintForceExporter<DataTypes>, ConstraintSolverImpl, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_constraintSolver; ///< Link to the constraint solver.

    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        sofa::core::behavior::MechanicalState<DataTypes>* mechanicalState = nullptr;

        std::string mechanicalStatePath{};

        if (arg->getAttribute("mechanicalState"))
        {
            mechanicalStatePath = arg->getAttribute("mechanicalState");
            context->findLinkDest(mechanicalState, mechanicalStatePath, nullptr);

            if (mechanicalState == nullptr)
            {
                arg->logError("Data attribute 'mechanicalState' does not point to a mechanical state of data type '" + std::string(DataTypes::Name()) + "'.");
                return false;
            }
        }
        else
        {
            mechanicalStatePath = "@./";
            arg->setAttribute("mechanicalState", mechanicalStatePath);
            context->findLinkDest(mechanicalState, mechanicalStatePath, nullptr);

            if (mechanicalState == nullptr)
            {
                arg->logError("Data attribute 'mechanicalState' has not been set and none can be found in the parent node context.");
                return false;
            }
        }

        return Inherited::canCreate(obj, context, arg);
    }

private:
    void computeForce();
};

#if !defined(SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_CONSTRAINTFORCEEXPORTER_CPP)
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API ConstraintForceExporter<sofa::defaulttype::Vec3Types>;
template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API ConstraintForceExporter<sofa::defaulttype::Rigid3Types>;
#endif

} //namespace sofa::component::constraint::lagrangian::solver
