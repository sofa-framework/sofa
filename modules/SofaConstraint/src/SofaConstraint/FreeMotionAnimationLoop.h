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
#include <SofaConstraint/config.h>

#include <sofa/simulation/CollisionAnimationLoop.h>

namespace sofa::core::behavior
{
    class ConstraintSolver;
}

namespace sofa::component::animationloop
{

class SOFA_SOFACONSTRAINT_API FreeMotionAnimationLoop : public sofa::simulation::CollisionAnimationLoop
{
public:
    SOFA_CLASS(FreeMotionAnimationLoop, sofa::simulation::CollisionAnimationLoop);

public:
    void step (const sofa::core::ExecParams* params, SReal dt) override;
    void init() override;
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override;

    /// Construction method called by ObjectFactory. An animation loop can only
    /// be created if
    template<class T>
    static typename T::SPtr create(T*, BaseContext* context, BaseObjectDescription* arg)
    {
        simulation::Node* gnode = dynamic_cast<simulation::Node*>(context);
        typename T::SPtr obj = sofa::core::objectmodel::New<T>(gnode);
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
        return obj;
    }

    Data<bool> m_solveVelocityConstraintFirst; ///< solve separately velocity constraint violations before position constraint violations
    Data<bool> d_threadSafeVisitor;

protected:
    FreeMotionAnimationLoop(simulation::Node* gnode);
    ~FreeMotionAnimationLoop() override ;

    ///< pointer towards a possible ConstraintSolver present in the scene graph
    sofa::core::behavior::ConstraintSolver *constraintSolver;

    ///< pointer towards a default ConstraintSolver (LCPConstraintSolver) used in case none was found in the scene graph
    sofa::core::sptr<sofa::core::behavior::ConstraintSolver> defaultSolver;
};

} // namespace sofa::component::animationloop
