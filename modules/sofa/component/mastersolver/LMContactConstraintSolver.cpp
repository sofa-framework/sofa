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
#include <sofa/simulation/common/MechanicalVisitor.h>
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
    : maxCollisionSteps( initData(&maxCollisionSteps,1,"maxSteps", "number of collision steps between each frame rendering") )
{
}

LMContactConstraintSolver::~LMContactConstraintSolver()
{
}


void LMContactConstraintSolver::bwdInit()
{
    sout << "collision" << sendl;
    computeCollision();
}

bool LMContactConstraintSolver::needPriorStatePropagation()
{
    using core::componentmodel::behavior::BaseLMConstraint;
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
    sout << "apply constraints" << sendl;
    simulation::MechanicalExpressJacobianVisitor JacobianVisitor;
    JacobianVisitor.execute(this->getContext());

    simulation::MechanicalSolveLMConstraintVisitor solveConstraints(needPropagation);
    solveConstraints.execute(this->getContext());

    simulation::MechanicalResetConstraintVisitor resetConstraints;
    resetConstraints.execute(this->getContext());

    simulation::MechanicalPropagatePositionAndVelocityVisitor propagateState;
    propagateState.execute(this->getContext());
}

bool LMContactConstraintSolver::isCollisionDetected()
{
    sout << "collision" << sendl;
    computeCollision();
    gpu::cuda::CudaRasterizer< defaulttype::Vec3dTypes > *rasterizer;
    this->getContext()->get(rasterizer);

    if (!rasterizer) return false;

    sout << "intersections : " << rasterizer->getDVDX_index().size() << sendl;
    return (rasterizer->getDVDX_index().size() != 0);
}

void LMContactConstraintSolver::step(double dt)
{
    const int maxSteps = maxCollisionSteps.getValue();

    // Then integrate the time step
    sout << "integration" << sendl;
    integrate(dt);

    bool propagateState=needPriorStatePropagation();

    int numStep=0;
    while ((numStep < maxSteps && isCollisionDetected()) || numStep==0)
    {
        sout << "Iteration " << numStep << sendl;
        solveConstraints(propagateState);
        ++numStep;
    }
}

} // namespace mastersolver

} // namespace component

} // namespace sofa

