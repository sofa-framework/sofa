/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/Node.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace simulation
{


Visitor::Result MechanicalVisitor::processNodeTopDown(simulation::Node* node)
{
    Result res = RESULT_CONTINUE;
    /*    if (node->solver != NULL) {
            ctime_t t0 = begin(node, node->solver);
            res = this->fwdOdeSolver(node, node->solver);
            end(node, node->solver, t0);
        }*/
    for (unsigned i=0; i<node->solver.size() && res!=RESULT_PRUNE; i++ )
    {
        ctime_t t0 = begin(node, node->solver[i]);
        res = this->fwdOdeSolver(node, node->solver[i]);
        end(node, node->solver[i], t0);
    }
    if (res != RESULT_PRUNE)
    {
        if (node->mechanicalState != NULL)
        {
            if (node->mechanicalMapping != NULL)
            {
                //cerr<<"MechanicalVisitor::processNodeTopDown, node "<<node->getName()<<" is a mapped model"<<endl;
                if (!node->mechanicalMapping->isMechanical())
                {
                    // stop all mechhanical computations
                    return RESULT_PRUNE;
                }
                Result res2;
                ctime_t t0 = begin(node, node->mechanicalMapping);
                res = this->fwdMechanicalMapping(node, node->mechanicalMapping);
                end(node, node->mechanicalMapping, t0);
                t0 = begin(node, node->mechanicalState);
                res2 = this->fwdMappedMechanicalState(node, node->mechanicalState);
                end(node, node->mechanicalState, t0);
                if (res2 == RESULT_PRUNE)
                    res = res2;
            }
            else
            {
                //cerr<<"MechanicalVisitor::processNodeTopDown, node "<<node->getName()<<" is a no-map model"<<endl;
                ctime_t t0 = begin(node, node->mechanicalState);
                res = this->fwdMechanicalState(node, node->mechanicalState);
                end(node, node->mechanicalState, t0);
            }
        }
    }
    if (res != RESULT_PRUNE)
    {
        if (node->mass != NULL)
        {
            ctime_t t0 = begin(node, node->mass);
            res = this->fwdMass(node, node->mass);
            end(node, node->mass, t0);
        }
    }
    if (res != RESULT_PRUNE)
    {
        res = for_each_r(this, node, node->forceField, &MechanicalVisitor::fwdForceField);
    }
    if (res != RESULT_PRUNE)
    {
        res = for_each_r(this, node, node->interactionForceField, &MechanicalVisitor::fwdInteractionForceField);
    }
    if (res != RESULT_PRUNE)
    {
        res = for_each_r(this, node, node->constraint, &MechanicalVisitor::fwdConstraint);
    }
    return res;
}

void MechanicalVisitor::processNodeBottomUp(simulation::Node* node)
{
    for_each(this, node, node->constraint, &MechanicalVisitor::bwdConstraint);
    if (node->mechanicalState != NULL)
    {
        if (node->mechanicalMapping != NULL)
        {
            if (node->mechanicalMapping->isMechanical())
            {
                ctime_t t0 = begin(node, node->mechanicalState);
                this->bwdMappedMechanicalState(node, node->mechanicalState);
                end(node, node->mechanicalState, t0);
                t0 = begin(node, node->mechanicalMapping);
                this->bwdMechanicalMapping(node, node->mechanicalMapping);
                end(node, node->mechanicalMapping, t0);
            }
        }
        else
        {
            ctime_t t0 = begin(node, node->mechanicalState);
            this->bwdMechanicalState(node, node->mechanicalState);
            end(node, node->mechanicalState, t0);
        }
    }
    /*    if (node->solver != NULL) {
            ctime_t t0 = begin(node, node->solver);
            this->bwdOdeSolver(node, node->solver);
            end(node, node->solver, t0);
        }*/
    for (unsigned i=0; i<node->solver.size(); i++ )
    {
        ctime_t t0 = begin(node, node->solver[i]);
        this->bwdOdeSolver(node, node->solver[i]);
        end(node, node->solver[i], t0);
    }
}


Visitor::Result MechanicalIntegrationVisitor::fwdOdeSolver(simulation::Node* node, core::componentmodel::behavior::OdeSolver* obj)
{
    double nextTime = node->getTime() + dt;
    MechanicalBeginIntegrationVisitor beginVisitor(dt);
    node->execute(&beginVisitor);

    //cerr<<"MechanicalIntegrationVisitor::fwdOdeSolver start solve obj"<<endl;
    obj->solve(dt);
    //cerr<<"MechanicalIntegrationVisitor::fwdOdeSolver end solve obj"<<endl;
    obj->propagatePositionAndVelocity(nextTime,core::componentmodel::behavior::OdeSolver::VecId::position(),core::componentmodel::behavior::OdeSolver::VecId::velocity());

    MechanicalEndIntegrationVisitor endVisitor(dt);
    node->execute(&endVisitor);
    return RESULT_PRUNE;
}



Visitor::Result  MechanicalVAvailVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->vAvail(v);
    return RESULT_CONTINUE;
}

Visitor::Result  MechanicalVAvailVisitor::fwdConstraint(simulation::Node* /*node*/,  core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
        mm->vAvail(v);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalVAllocVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->vAlloc(v);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalVAllocVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
        mm->vAlloc(v);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalVFreeVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->vFree(v);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalVFreeVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
        mm->vFree(v);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalVOpVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMechanicalState, model "<<mm->getName()<<endl;
    mm->vOp(v,a,b,f);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalVOpVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* /*mm*/)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMappedMechanicalState, model "<<mm->getName()<<endl;
    //mm->vOp(v,a,b,f);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalVOpVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
        mm->vOp(v,a,b,f);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalVMultiOpVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMechanicalState, model "<<mm->getName()<<endl;
    mm->vMultiOp(ops);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalVMultiOpVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* /*mm*/)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMappedMechanicalState, model "<<mm->getName()<<endl;
    //mm->vMultiOp(ops);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalVMultiOpVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
        mm->vMultiOp(ops);
    return RESULT_CONTINUE;
}

/// Sequential code
Visitor::Result MechanicalVDotVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    *total += mm->vDot(a,b);
    return RESULT_CONTINUE;
}
/// Sequential code
Visitor::Result MechanicalVDotVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
        *total += mm->vDot(a,b);
    return RESULT_CONTINUE;
}


/// Parallel code
Visitor::Result MechanicalVDotVisitor::processNodeTopDown(simulation::Node* node, LocalStorage* stack)
{
    double* localTotal = new double(0.0);
    stack->push(localTotal);
    if (node->mechanicalState && !node->mechanicalMapping)
    {
        core::componentmodel::behavior::BaseMechanicalState* mm = node->mechanicalState;
        *localTotal += mm->vDot(a,b);
    }
    for (simulation::Node::Sequence<core::componentmodel::behavior::BaseConstraint>::iterator it = node->
            constraint.begin();
            it != node->constraint.end();
            ++it)
    {
        core::componentmodel::behavior::BaseConstraint* c = *it;
        core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
        if (mm)
            *localTotal += mm->vDot(a,b);
    }
    return RESULT_CONTINUE;
}
/// Parallel code
void MechanicalVDotVisitor::processNodeBottomUp(simulation::Node* /*node*/, LocalStorage* stack)
{
    double* localTotal = static_cast<double*>(stack->pop());
    double* parentTotal = static_cast<double*>(stack->top());
    if (!parentTotal)
        *total += *localTotal; // root
    else
        *parentTotal += *localTotal;
    delete localTotal;
}


Visitor::Result MechanicalPropagateDxVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setDx(dx);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateDxVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    map->propagateDx();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateDxVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
        mm->setDx(dx);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPropagateDxAndResetForceVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setDx(dx);
    mm->setF(f);
    mm->resetForce();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateDxAndResetForceVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    map->propagateDx();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateDxAndResetForceVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->resetForce();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateDxAndResetForceVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->setDx(dx);
        mm->setF(f);
        mm->resetForce();
    }
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPropagateAndAddDxVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    map->propagateDx();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateAndAddDxVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    //mm->printDOF(VecId::dx());
    mm->addDxToCollisionModel();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateAndAddDxVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
        mm->setDx(dx);
    c->projectPosition();
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalAddMDxVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setF(res);
    if (!dx.isNull())
        mm->setDx(dx);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAddMDxVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->setF(res);
        if (!dx.isNull())
            mm->setDx(dx);
    }
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAddMDxVisitor::fwdMass(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMass* mass)
{
    mass->addMDx(factor);
    return RESULT_PRUNE;
}
Visitor::Result MechanicalAccFromFVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setDx(a);
    mm->setF(f);
    /// \todo Check presence of Mass
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAccFromFVisitor::fwdMass(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMass* mass)
{
    mass->accFromF();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAccFromFVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->setDx(a);
        mm->setF(f);
    }
    return RESULT_CONTINUE;
}


MechanicalPropagatePositionAndVelocityVisitor::MechanicalPropagatePositionAndVelocityVisitor(double t, VecId x, VecId v) : t(t), x(x), v(v)
{
    //cerr<<"::MechanicalPropagatePositionAndVelocityVisitor"<<endl;
}


Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::processNodeTopDown(simulation::Node* node)
{
    //cerr<<" MechanicalPropagatePositionAndVelocityVisitor::processNodeTopDown "<<node->getName()<<endl;
    node->setTime(t);
    node->updateSimulationContext();
    return MechanicalVisitor::processNodeTopDown( node);
}

void MechanicalPropagatePositionAndVelocityVisitor::processNodeBottomUp(simulation::Node* node)
{
    //cerr<<" MechanicalPropagatePositionAndVelocityVisitor::processNodeBottomUp "<<node->getName()<<endl;
    //for_each(this, node, node->constraint, &MechanicalPropagatePositionAndVelocityVisitor::bwdConstraint);
    MechanicalVisitor::processNodeBottomUp( node);

}


Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setX(x);
    mm->setV(v);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    map->propagateX();
    map->propagateV();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    // FF, 14/07/06: I do not understand the purpose of this method
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->setX(x);
        mm->setV(v);
    }
    c->projectPosition();
    c->projectVelocity();
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPropagateFreePositionVisitor::processNodeTopDown(simulation::Node* node)
{
    node->setTime(t);
    node->updateSimulationContext();
    return MechanicalVisitor::processNodeTopDown( node);
}
Visitor::Result MechanicalPropagateFreePositionVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setXfree(x);
    mm->setVfree(v);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateFreePositionVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    map->propagateXfree();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateFreePositionVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    c->projectFreePosition();
    c->projectFreeVelocity();
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalResetForceVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setF(res);
    mm->resetForce();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalResetForceVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->resetForce();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalResetForceVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->setF(res);
        mm->resetForce();
    }
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalComputeForceVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setF(res);
    mm->accumulateForce();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalComputeForceVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->accumulateForce();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalComputeForceVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->setF(res);
        mm->accumulateForce();
    }
    return RESULT_CONTINUE;
}
//virtual Result fwdMass(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMass* mass)
//{
//	mass->computeForce();
//	return RESULT_CONTINUE;
//}

Visitor::Result MechanicalComputeForceVisitor::fwdForceField(simulation::Node* /*node*/, core::componentmodel::behavior::BaseForceField* ff)
{
    //cerr<<"MechanicalComputeForceVisitor::fwdForceField "<<ff->getName()<<endl;
    ff->addForce();
    return RESULT_CONTINUE;
}

void MechanicalComputeForceVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    //cerr<<"MechanicalComputeForceVisitor::bwdMechanicalMapping "<<map->getName()<<endl;
    map->accumulateForce();
}

Visitor::Result MechanicalComputeDfVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setF(res);
    mm->accumulateDf();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalComputeDfVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->accumulateDf();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalComputeDfVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->setF(res);
        mm->accumulateDf();
    }
    return RESULT_CONTINUE;
}
//virtual Result fwdMass(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMass* mass)
//{
//	mass->computeDf();
//	return RESULT_CONTINUE;
//}
Visitor::Result MechanicalComputeDfVisitor::fwdForceField(simulation::Node* /*node*/, core::componentmodel::behavior::BaseForceField* ff)
{
    if (useV)
        ff->addDForceV();
    else
        ff->addDForce();
    return RESULT_CONTINUE;
}
void MechanicalComputeDfVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    map->accumulateDf();
}



Visitor::Result MechanicalAddMBKdxVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setF(res);
    mm->accumulateDf();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAddMBKdxVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->accumulateDf();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAddMBKdxVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->setF(res);
        mm->accumulateDf();
    }
    return RESULT_CONTINUE;
}
//virtual Result fwdMass(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMass* mass)
//{
//	mass->computeDf();
//	return RESULT_CONTINUE;
//}
Visitor::Result MechanicalAddMBKdxVisitor::fwdForceField(simulation::Node* /*node*/, core::componentmodel::behavior::BaseForceField* ff)
{
    if (useV)
        ff->addMBKv(mFactor, bFactor, kFactor);
    else
        ff->addMBKdx(mFactor, bFactor, kFactor);
    return RESULT_CONTINUE;
}

void MechanicalAddMBKdxVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    map->accumulateDf();
}

Visitor::Result MechanicalResetConstraintVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    // mm->setC(res);
    mm->resetConstraint();
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalResetConstraintVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->resetConstraint();
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalAccumulateConstraint::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    c->applyConstraint(contactId, mu);
    return RESULT_CONTINUE;
}

void MechanicalAccumulateConstraint::bwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    map->accumulateConstraint();
}


Visitor::Result MechanicalApplyConstraintsVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setDx(res);
    //mm->projectResponse();
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalApplyConstraintsVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* /*mm*/)
{
    //mm->projectResponse();
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalApplyConstraintsVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->setDx(res);
    }

    return RESULT_CONTINUE;
}

void MechanicalApplyConstraintsVisitor::bwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    c->projectResponse();

    if (W != NULL)
    {
        c->projectResponse(W);
    }
}

Visitor::Result MechanicalBeginIntegrationVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->beginIntegration(dt);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalBeginIntegrationVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->beginIntegration(dt);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalBeginIntegrationVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->beginIntegration(dt);
    }
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalEndIntegrationVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->endIntegration(dt);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalEndIntegrationVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->endIntegration(dt);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalEndIntegrationVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->endIntegration(dt);
    }
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalComputeComplianceVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* ms)
{
    ms->getCompliance(_W);
    return RESULT_PRUNE;
}
Visitor::Result MechanicalComputeComplianceVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* ms)
{
    ms->getCompliance(_W);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalComputeContactForceVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->setF(res);
    mm->accumulateForce();
    return RESULT_PRUNE;
}

Visitor::Result MechanicalComputeContactForceVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->accumulateForce();
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalComputeContactForceVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        mm->setF(res);
        mm->accumulateForce();
    }
    return RESULT_CONTINUE;
}

void MechanicalComputeContactForceVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    map->accumulateForce();
}

Visitor::Result MechanicalAddSeparateGravityVisitor::fwdMass(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMass* mass)
{
    if( mass->m_separateGravity.getValue() )
        mass->addGravityToV(dt);

    return RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa

