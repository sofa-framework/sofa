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
#include <sofa/component/mastersolver/LMContactConstraintSolver.h>
#include <sofa/component/constraintset/LMConstraintSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/CollisionBeginEvent.h>
#include <sofa/simulation/common/CollisionEndEvent.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/gpu/cuda/CudaRasterizer.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>


//TODO: hope that CudaRasterizer will derive from a non template class Rasterizer. Need an access to the volume detected

namespace sofa
{

namespace component
{

namespace mastersolver
{

int LMContactConstraintSolverClass = core::RegisterObject("invert the Sofa simulation pipeline: first integration, than collision detection until no more collision is found.")
        .add< LMContactConstraintSolver >()
        ;

SOFA_DECL_CLASS(LMContactConstraintSolver);

LMContactConstraintSolver::LMContactConstraintSolver()
    : maxCollisionSteps( initData(&maxCollisionSteps,(unsigned int)1,"maxSteps", "number of collision steps between each frame rendering") )
{
}

LMContactConstraintSolver::~LMContactConstraintSolver()
{
}


void LMContactConstraintSolver::bwdInit()
{
//  sout << "collision" << sendl;
    computeCollision();
}

bool LMContactConstraintSolver::needPriorStatePropagation()
{
    using core::behavior::BaseLMConstraint;
    bool needPriorPropagation=false;
    {
        helper::vector< BaseLMConstraint* > c;
        this->getContext()->get<BaseLMConstraint>(&c, core::objectmodel::BaseContext::SearchDown);
        for (unsigned int i=0; i<c.size(); ++i)
        {
            if (!c[i]->isCorrectionComputedWithSimulatedDOF())
            {
                needPriorPropagation=true;
                sout << "Propagating the State because of "<< c[i]->getName() << sendl;
                break;
            }
        }
    }
    return needPriorPropagation;
}

void LMContactConstraintSolver::solveConstraints(bool needPropagation)
{
//  sout << "apply constraints" << sendl;
    simulation::Node *node = (simulation::Node*)getContext();
    simulation::MechanicalExpressJacobianVisitor JacobianVisitor(node);
    JacobianVisitor.execute(node);

    helper::vector< constraintset::LMConstraintSolver* > listSolver;
    node->get<constraintset::LMConstraintSolver>(&listSolver, core::objectmodel::BaseContext::SearchDown);

    helper::vector< bool > constraintActive(listSolver.size(), false);
    for (unsigned int i=0; i<listSolver.size(); ++i)
    {
        if (listSolver[i]->constraintPos.getValue()) constraintActive[i]=true;
        else                                         listSolver[i]->constraintPos.setValue(true);
    }

    core::behavior::BaseMechanicalState::VecId positionState=core::behavior::BaseMechanicalState::VecId::position();
    simulation::MechanicalSolveLMConstraintVisitor solveConstraintsPosition(positionState,needPropagation, false);
    solveConstraintsPosition.execute(node);

    for (unsigned int i=0; i<listSolver.size(); ++i)
    {
        listSolver[i]->constraintPos.setValue(constraintActive[i]);
    }

    simulation::MechanicalPropagatePositionVisitor propagateState;
    propagateState.ignoreMask=false;
    propagateState.execute(node);
}

bool LMContactConstraintSolver::isCollisionDetected()
{
//  sout << "collision" << sendl;
    {
        simulation::CollisionBeginEvent evBegin;
        simulation::PropagateEventVisitor eventPropagation(&evBegin);
        eventPropagation.execute(getContext());
    }
    ((simulation::Node*) getContext())->execute<simulation::CollisionDetectionVisitor>();
    {
        simulation::CollisionEndEvent evEnd;
        simulation::PropagateEventVisitor eventPropagation(&evEnd);
        eventPropagation.execute(getContext());
    }




    gpu::cuda::CudaRasterizer< defaulttype::Vec3dTypes > *rasterizer;
    this->getContext()->get(rasterizer);

    if (!rasterizer) return false;

//  sout << "intersections : " << rasterizer->getNbPairs() << sendl;
    return (rasterizer->getNbPairs() != 0);
}

void LMContactConstraintSolver::step(double dt)
{
    simulation::Node *node = (simulation::Node*)getContext();
    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }

    sofa::helper::AdvancedTimer::stepBegin("MasterSolverStep");
    const unsigned int maxSteps = maxCollisionSteps.getValue();

    // Then integrate the time step
    //    sout << "integration" << sendl;
    integrate(dt);

    bool propagateState=needPriorStatePropagation();
    for (unsigned int step=0; step<maxSteps; ++step)
    {
        node->execute<simulation::MechanicalResetConstraintVisitor>();
        node->execute<simulation::CollisionResetVisitor>();
        if (isCollisionDetected())
        {
            node->execute<simulation::CollisionResetVisitor>();
            node->execute<simulation::CollisionResponseVisitor>();
            solveConstraints(propagateState);
        }
        else
        {
            //No collision --> no constraint
            break;
        }
    }
    sofa::helper::AdvancedTimer::stepEnd("MasterSolverStep");

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }
}

} // namespace mastersolver

} // namespace component

} // namespace sofa

