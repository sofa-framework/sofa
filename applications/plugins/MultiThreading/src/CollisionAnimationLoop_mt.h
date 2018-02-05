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
#ifndef SOFA_SIMULATION_TREE_COLLISIONANIMATIONLOOP_MT_H
#define SOFA_SIMULATION_TREE_COLLISIONANIMATIONLOOP_MT_H

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include "BaseAnimationLoop_mt.h"
#include <sofa/core/ExecParams.h>
#include <SofaSimulationCommon/common.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>


//using namespace sofa::core::objectmodel;
//using namespace sofa::core::behavior;

namespace sofa
{

namespace simulation
{

using namespace sofa;

/**
 *  \brief Component responsible for main simulation algorithms, managing how
 *  and when collisions and integrations computations happen.
 *
 *  This class can optionally replace the default computation scheme of computing
 *  collisions then doing an integration step.
 *
 *  Note that it is in a preliminary stage, hence its fonctionnalities and API will
 *  certainly change soon.
 *
 */


class CollisionAnimationLoop_mt : public sofa::core::behavior::BaseAnimationLoop_mt
{
public:
    typedef BaseAnimationLoop_mt Inherit;

protected:
    CollisionAnimationLoop_mt(simulation::Node* gnode);

    virtual ~CollisionAnimationLoop_mt();


public:

	virtual void step(const core::ExecParams* params /* PARAMS FIRST =ExecParams::defaultInstance()*/, double dt) = 0;

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T*, BaseContext* context, BaseObjectDescription* arg)
    {
        simulation::Node* gnode = dynamic_cast<simulation::Node*>(context);
        typename T::SPtr obj = core::objectmodel::New<T>(gnode);
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
        return obj;
    }

protected:

    /// @name Visitors
    /// These methods provides an abstract view of the mechanical system to animate.
    /// They are implemented by executing Visitors in the subtree of the scene-graph below this solver.
    /// @{

    /// Activate collision pipeline
	virtual void collisionReset(const core::ExecParams* params = core::ExecParams::defaultInstance());
	
    virtual void collisionCompute(const core::ExecParams* params = core::ExecParams::defaultInstance());

	virtual void collisionResponse(const core::ExecParams* params = core::ExecParams::defaultInstance());


    /// Activate OdeSolvers
    virtual void integrate(const core::ExecParams* params /* PARAMS FIRST  = core::ExecParams::defaultInstance()*/, double dt);


    typedef simulation::Node::Sequence<core::behavior::OdeSolver> Solvers;
    typedef core::collision::Pipeline Pipeline;
    const Solvers& getSolverSequence();

    // the parent Node of CollisionAnimationLoop its self (usually, this parent node is the root node of the simulation graph)
    // This pointer is initialized one time at the construction, avoiding dynamic_cast<Node*>(context) every time step
    simulation::Node* gnode;
    /// @}
};

} // namespace simulation

} // namespace sofa

#endif /* SOFA_SIMULATION_TREE_COLLISIONANIMATIONLOOP_MT_H */
