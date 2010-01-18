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
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/Node.h>
#include <iostream>

namespace sofa
{

namespace simulation
{
using std::cerr;
using std::endl;
//Max size for vector to be allowed to be dumped
#define DUMP_VISITOR_MAX_SIZE_VECTOR 50

Visitor::Result MechanicalVisitor::processNodeTopDown(simulation::Node* node)
{
    Result res = RESULT_CONTINUE;
    /*    if (node->solver != NULL) {
            ctime_t t0 = beginProcess(node, node->solver);
            res = this->fwdOdeSolver(node, node->solver);
            endProcess(node, node->solver, t0);
        }*/
    for (unsigned i=0; i<node->solver.size() && res!=RESULT_PRUNE; i++ )
    {
        if(testTags(node->solver[i]))
        {
            debug_write_state_before(node->solver[i]);
            res = this->fwdOdeSolver(node, node->solver[i]);
            debug_write_state_after(node->solver[i]);
        }
    }

    if (res != RESULT_PRUNE)
    {
        if (node->mechanicalState != NULL)
        {
            if (node->mechanicalMapping != NULL)
            {
                //cerr<<"MechanicalVisitor::processNodeTopDown, node "<<node->getName()<<" is a mapped model"<<endl;
                if (stopAtMechanicalMapping(node, node->mechanicalMapping))
                {
                    // stop all mechanical computations
                    std::cerr << "Pruning " << this->getClassName() << " at " << node->getPathName() << " because of mapping" << std::endl;
                    return RESULT_PRUNE;
                }
                Result res2 = RESULT_CONTINUE;
                if(testTags(node->mechanicalMapping))
                {
                    debug_write_state_before(node->mechanicalMapping);
                    res = this->fwdMechanicalMapping(node, node->mechanicalMapping);
                    debug_write_state_after(node->mechanicalMapping);
                }

                if(testTags(node->mechanicalState))
                {
                    debug_write_state_before(node->mechanicalState);
                    res2 = this->fwdMappedMechanicalState(node, node->mechanicalState);
                    debug_write_state_after(node->mechanicalState);
                }


                if (res2 == RESULT_PRUNE)
                    res = res2;
            }
            else
            {
                if(testTags(node->mechanicalState))
                {
                    //cerr<<"MechanicalVisitor::processNodeTopDown, node "<<node->getName()<<" is a no-map model"<<endl;
                    debug_write_state_before(node->mechanicalState);
                    res = this->fwdMechanicalState(node, node->mechanicalState);
                    debug_write_state_after(node->mechanicalState);
                }
            }
        }
    }
    if (res != RESULT_PRUNE)
    {
        if (node->mass != NULL)
        {
            if(testTags(node->mass))
            {
                debug_write_state_before(node->mass);
                res = this->fwdMass(node, node->mass);
                debug_write_state_after(node->mass);
            }
        }
    }
    if (res != RESULT_PRUNE)
    {
        res = for_each_r(this, node, node->constraintSolver, &MechanicalVisitor::fwdConstraintSolver);
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
    if (res != RESULT_PRUNE)
    {
        res = for_each_r(this, node, node->LMConstraint, &MechanicalVisitor::fwdLMConstraint);
    }
    return res;
}

void MechanicalVisitor::processNodeBottomUp(simulation::Node* node)
{
    for_each(this, node, node->constraint, &MechanicalVisitor::bwdConstraint);
    for_each(this, node, node->LMConstraint, &MechanicalVisitor::bwdLMConstraint);
    for_each(this, node, node->constraintSolver, &MechanicalVisitor::bwdConstraintSolver);
    if (node->mechanicalState != NULL)
    {
        if (node->mechanicalMapping != NULL)
        {
            if (!stopAtMechanicalMapping(node, node->mechanicalMapping))
            {
                if(testTags(node->mechanicalState))
                {
                    this->bwdMappedMechanicalState(node, node->mechanicalState);
                    this->bwdMechanicalMapping(node, node->mechanicalMapping);
                }
            }
        }
        else
        {
            if(testTags(node->mechanicalState))
                this->bwdMechanicalState(node, node->mechanicalState);
        }

    }
    /*    if (node->solver != NULL) {
            ctime_t t0 = beginProcess(node, node->solver);
            this->bwdOdeSolver(node, node->solver);
            endProcess(node, node->solver, t0);
        }*/
    for (unsigned i=0; i<node->solver.size(); i++ )
    {
        if(testTags(node->solver[i]))
            this->bwdOdeSolver(node, node->solver[i]);
    }
}
#ifdef SOFA_DUMP_VISITOR_INFO
void MechanicalVisitor::printReadVectors(core::componentmodel::behavior::BaseMechanicalState* mm)
{
    if (!mm || !readVector.size() || !Visitor::printActivated) return;

    printNode("Input");

    for (unsigned int i=0; i<readVector.size(); ++i)
    {
        std::ostringstream infoStream;
        TRACE_ARGUMENT arg;
        if (mm->getSize() < DUMP_VISITOR_MAX_SIZE_VECTOR)
        {
            mm->printDOF(readVector[i], infoStream);
            arg.push_back(std::make_pair("value", infoStream.str()));
        }

        printNode("Vector", readVector[i].getName(), arg);
        printCloseNode("Vector");
    }

    printCloseNode("Input");
}

void MechanicalVisitor::printWriteVectors(core::componentmodel::behavior::BaseMechanicalState* mm)
{
    if (!mm || !writeVector.size() || !Visitor::printActivated) return;

    printNode("Output");

    for (unsigned int i=0; i<writeVector.size(); ++i)
    {
        std::ostringstream infoStream;
        TRACE_ARGUMENT arg;
        if (mm->getSize() < DUMP_VISITOR_MAX_SIZE_VECTOR)
        {
            mm->printDOF(writeVector[i], infoStream);
            arg.push_back(std::make_pair("value", infoStream.str()));
        }

        printNode("Vector", writeVector[i].getName(), arg);
        printCloseNode("Vector");
    }

    printCloseNode("Output");
}

void MechanicalVisitor::printReadVectors(simulation::Node* node, core::objectmodel::BaseObject* obj)
{
    if (!Visitor::printActivated) return;
    if (readVector.size())
    {
        core::componentmodel::behavior::BaseMechanicalState *dof1, *dof2;
        if ( sofa::core::componentmodel::behavior::InteractionForceField* interact = dynamic_cast<sofa::core::componentmodel::behavior::InteractionForceField*> (obj))
        {
            dof1=interact->getMechModel1();
            dof2=interact->getMechModel2();
        }
        else if (sofa::core::componentmodel::behavior::InteractionConstraint* interact = dynamic_cast<sofa::core::componentmodel::behavior::InteractionConstraint*> (obj))
        {
            dof1=interact->getMechModel1();
            dof2=interact->getMechModel2();
        }
        else if (sofa::core::componentmodel::behavior::BaseLMConstraint* interact = dynamic_cast<sofa::core::componentmodel::behavior::BaseLMConstraint*> (obj))
        {
            dof1=interact->getConstrainedMechModel1();
            dof2=interact->getConstrainedMechModel2();
        }
        else
        {
            printReadVectors(node->mechanicalState);
            return;
        }

        TRACE_ARGUMENT arg1;
        arg1.push_back(std::make_pair("type", dof1->getClassName()));
        printNode("Components", dof1->getName(), arg1);
        printReadVectors(dof1);
        printCloseNode("Components");

        TRACE_ARGUMENT arg2;
        arg2.push_back(std::make_pair("type", dof2->getClassName()));
        printNode("Components", dof2->getName(), arg2);
        printReadVectors(dof2);
        printCloseNode("Components");
    }
}
void MechanicalVisitor::printWriteVectors(simulation::Node* node, core::objectmodel::BaseObject* obj)
{
    if (!Visitor::printActivated) return;
    if (writeVector.size())
    {
        core::componentmodel::behavior::BaseMechanicalState *dof1, *dof2;
        if ( sofa::core::componentmodel::behavior::InteractionForceField* interact = dynamic_cast<sofa::core::componentmodel::behavior::InteractionForceField*> (obj))
        {
            dof1=interact->getMechModel1();
            dof2=interact->getMechModel2();
        }
        else if (sofa::core::componentmodel::behavior::InteractionConstraint* interact = dynamic_cast<sofa::core::componentmodel::behavior::InteractionConstraint*> (obj))
        {
            dof1=interact->getMechModel1();
            dof2=interact->getMechModel2();
        }
        else if (sofa::core::componentmodel::behavior::BaseLMConstraint* interact = dynamic_cast<sofa::core::componentmodel::behavior::BaseLMConstraint*> (obj))
        {
            dof1=interact->getConstrainedMechModel1();
            dof2=interact->getConstrainedMechModel2();
        }
        else
        {
            printWriteVectors(node->mechanicalState);
            return;
        }

        TRACE_ARGUMENT arg1;
        arg1.push_back(std::make_pair("type", dof1->getClassName()));
        printNode("Components", dof1->getName(), arg1);
        printWriteVectors(dof1);
        printCloseNode("Components");

        TRACE_ARGUMENT arg2;
        arg2.push_back(std::make_pair("type", dof2->getClassName()));
        printNode("Components", dof2->getName(), arg2);
        printWriteVectors(dof2);
        printCloseNode("Components");
    }
}
#endif


simulation::Node::ctime_t MechanicalVisitor::beginProcess(simulation::Node* node, core::objectmodel::BaseObject* obj)
{
    ctime_t t=begin(node, obj);
#ifdef SOFA_DUMP_VISITOR_INFO
    printReadVectors(node, obj);
#endif
    return t;
}

void MechanicalVisitor::endProcess(simulation::Node* node, core::objectmodel::BaseObject* obj, ctime_t t0)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    printWriteVectors(node, obj);
#endif
    return end(node, obj, t0);
}



Visitor::Result MechanicalIntegrationVisitor::fwdOdeSolver(simulation::Node* node, core::componentmodel::behavior::OdeSolver* obj)
{
    ctime_t t0 = beginProcess(node, obj);
    double nextTime = node->getTime() + dt;
    MechanicalBeginIntegrationVisitor beginVisitor(dt);
    node->execute(&beginVisitor);

    //cerr<<"MechanicalIntegrationVisitor::fwdOdeSolver start solve obj"<<endl;
    obj->solve(dt);
// 	cerr<<"MechanicalIntegrationVisitor::fwdOdeSolver endVisitor ok"<<endl;

    //cerr<<"MechanicalIntegrationVisitor::fwdOdeSolver end solve obj"<<endl;
    obj->propagatePositionAndVelocity(nextTime,core::componentmodel::behavior::OdeSolver::VecId::position(),core::componentmodel::behavior::OdeSolver::VecId::velocity());

    MechanicalEndIntegrationVisitor endVisitor(dt);
    node->execute(&endVisitor);

    endProcess(node, obj, t0);
    return RESULT_PRUNE;
}



Visitor::Result  MechanicalVAvailVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->vAvail(v);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalVAllocVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->vAlloc(v);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalVFreeVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->vFree(v);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalVOpVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMechanicalState, model "<<mm->getName()<<endl;
    ctime_t t0 = beginProcess(node, mm);
    mm->vOp(v,a,b,f);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalVOpVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* /*mm*/)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMappedMechanicalState, model "<<mm->getName()<<endl;
    //mm->vOp(v,a,b,f);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalVMultiOpVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMechanicalState, model "<<mm->getName()<<endl;
    ctime_t t0 = beginProcess(node, mm);
    mm->vMultiOp(ops);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalVMultiOpVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* /*mm*/)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMappedMechanicalState, model "<<mm->getName()<<endl;
    //mm->vMultiOp(ops);
    return RESULT_CONTINUE;
}

/// Sequential code
Visitor::Result MechanicalVDotVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    *total += mm->vDot(a,b);
    endProcess(node, mm, t0);
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


Visitor::Result MechanicalPropagateDxVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setDx(dx);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateDxVisitor::fwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);
    if (!ignoreMask)
    {
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->propagateDx();
        map->getMechTo()->forceMask.activate(false);
    }
    else map->propagateDx();

    endProcess(node, map, t0);
    return RESULT_CONTINUE;
}

void MechanicalPropagateDxVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}


Visitor::Result MechanicalPropagateVVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setV(v);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateVVisitor::fwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);
    if (!ignoreMask)
    {
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->propagateV();
        map->getMechTo()->forceMask.activate(false);
    }
    else map->propagateV();

    endProcess(node, map, t0);
    return RESULT_CONTINUE;
}
void MechanicalPropagateVVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}


Visitor::Result MechanicalPropagateXVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setX(x);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateXVisitor::fwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);
    if (!ignoreMask)
    {
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->propagateX();
        map->getMechTo()->forceMask.activate(false);
    }
    else map->propagateX();

    endProcess(node, map, t0);
    return RESULT_CONTINUE;
}
void MechanicalPropagateXVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}


Visitor::Result MechanicalPropagateDxAndResetForceVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setDx(dx);
    mm->setF(f);
    mm->resetForce();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateDxAndResetForceVisitor::fwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);
    if (!ignoreMask)
    {
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->propagateDx();
        map->getMechTo()->forceMask.activate(false);
    }
    else map->propagateDx();

    endProcess(node, map, t0);
    return RESULT_CONTINUE;
}
void MechanicalPropagateDxAndResetForceVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}
Visitor::Result MechanicalPropagateDxAndResetForceVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->resetForce();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPropagateXAndResetForceVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setX(x);
    mm->setF(f);
    mm->resetForce();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateXAndResetForceVisitor::fwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);

    if (!ignoreMask)
    {
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->propagateX();
        map->getMechTo()->forceMask.activate(false);
    }
    else map->propagateX();

    endProcess(node, map, t0);
    return RESULT_CONTINUE;
}
void MechanicalPropagateXAndResetForceVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}
Visitor::Result MechanicalPropagateXAndResetForceVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->resetForce();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPropagateAndAddDxVisitor::fwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);

    if (!ignoreMask)
    {
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->propagateDx();
        map->propagateV();
        map->getMechTo()->forceMask.activate(false);
    }
    else
    {
        map->propagateDx();
        map->propagateV();
    }
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
}
void MechanicalPropagateAndAddDxVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}
Visitor::Result MechanicalPropagateAndAddDxVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    //mm->printDOF(VecId::dx());
    mm->addDxToCollisionModel();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalAddMDxVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setF(res);
    if (!dx.isNull())
        mm->setDx(dx);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAddMDxVisitor::fwdMass(simulation::Node* node, core::componentmodel::behavior::BaseMass* mass)
{
    ctime_t t0 = beginProcess(node, mass);
    mass->addMDx(factor);
    endProcess(node, mass, t0);
    return RESULT_PRUNE;
}
Visitor::Result MechanicalAccFromFVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setDx(a);
    mm->setF(f);
    endProcess(node, mm, t0);
    /// \todo Check presence of Mass
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAccFromFVisitor::fwdMass(simulation::Node* node, core::componentmodel::behavior::BaseMass* mass)
{
    ctime_t t0 = beginProcess(node, mass);
    mass->accFromF();
    endProcess(node, mass, t0);
    return RESULT_CONTINUE;
}

#ifdef SOFA_SUPPORT_MAPPED_MASS
MechanicalPropagatePositionAndVelocityVisitor::MechanicalPropagatePositionAndVelocityVisitor(double t, VecId x, VecId v, VecId a, bool m) : t(t), x(x), v(v), a(a), ignoreMask(m)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
    //cerr<<"::MechanicalPropagatePositionAndVelocityVisitor"<<endl;
}
#else
MechanicalPropagatePositionAndVelocityVisitor::MechanicalPropagatePositionAndVelocityVisitor(double t, VecId x, VecId v, bool m) : t(t), x(x), v(v), ignoreMask(m)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
    //cerr<<"::MechanicalPropagatePositionAndVelocityVisitor"<<endl;
}
#endif

MechanicalPropagatePositionVisitor::MechanicalPropagatePositionVisitor(double t, VecId x, bool m) : t(t), x(x), ignoreMask(m)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
    //cerr<<"::MechanicalPropagatePositionAndVelocityVisitor"<<endl;
}

#ifdef SOFA_SUPPORT_MAPPED_MASS
Visitor::Result MechanicalAddMDxVisitor::fwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    if (!dx.isNull())
    {
        ctime_t t0 = beginProcess(node, map);

        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->propagateDx();
        map->getMechTo()->forceMask.activate(false);

        endProcess(node, map, t0);
    }
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAddMDxVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->resetForce();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
void MechanicalAddMDxVisitor::bwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);
    map->getMechFrom()->forceMask.activate(true);
    map->getMechTo()->forceMask.activate(true);
    map->accumulateForce();
    map->getMechTo()->forceMask.activate(false);
    endProcess(node, map, t0);
}
void MechanicalAddMDxVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}
#else
Visitor::Result MechanicalAddMDxVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* /*map*/)
{
    return RESULT_PRUNE;
}
Visitor::Result MechanicalAddMDxVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* /*mm*/)
{
    return RESULT_PRUNE;
}
#endif


Visitor::Result MechanicalPropagatePositionVisitor::processNodeTopDown(simulation::Node* node)
{
    //cerr<<" MechanicalPropagatePositionVisitor::processNodeTopDown "<<node->getName()<<endl;
    node->updateSimulationContext();
    return MechanicalVisitor::processNodeTopDown( node);
}

void MechanicalPropagatePositionVisitor::processNodeBottomUp(simulation::Node* node)
{
    //cerr<<" MechanicalPropagatePositionVisitor::processNodeBottomUp "<<node->getName()<<endl;
    //for_each(this, node, node->constraint, &MechanicalPropagatePositionVisitor::bwdConstraint);
    MechanicalVisitor::processNodeBottomUp( node);

}


Visitor::Result MechanicalPropagatePositionVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setX(x);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagatePositionVisitor::fwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);

    if (!ignoreMask)
    {
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->propagateX();
        map->getMechTo()->forceMask.activate(false);
    }
    else
    {
        map->propagateX();
    }
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
}
void MechanicalPropagatePositionVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}
Visitor::Result MechanicalPropagatePositionVisitor::fwdConstraint(simulation::Node* node, core::componentmodel::behavior::BaseConstraint* c)
{
    ctime_t t0 = beginProcess(node, c);
    c->projectPosition();
    endProcess(node, c, t0);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::processNodeTopDown(simulation::Node* node)
{
    //cerr<<" MechanicalPropagatePositionAndVelocityVisitor::processNodeTopDown "<<node->getName()<<endl;
    node->updateSimulationContext();
    return MechanicalVisitor::processNodeTopDown( node);
}

void MechanicalPropagatePositionAndVelocityVisitor::processNodeBottomUp(simulation::Node* node)
{
    //cerr<<" MechanicalPropagatePositionAndVelocityVisitor::processNodeBottomUp "<<node->getName()<<endl;
    //for_each(this, node, node->constraint, &MechanicalPropagatePositionAndVelocityVisitor::bwdConstraint);
    MechanicalVisitor::processNodeBottomUp( node);

}


Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setX(x);
    mm->setV(v);
#ifdef SOFA_SUPPORT_MAPPED_MASS
    mm->setDx(a);
    mm->resetAcc();
#endif
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::fwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);
    if (!ignoreMask)
    {
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->propagateX();
        map->propagateV();
#ifdef SOFA_SUPPORT_MAPPED_MASS
        map->propagateA();
#endif
        map->getMechTo()->forceMask.activate(false);
    }
    else
    {
        map->propagateX();
        map->propagateV();
#ifdef SOFA_SUPPORT_MAPPED_MASS
        map->propagateA();
#endif
    }
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
}
void MechanicalPropagatePositionAndVelocityVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}
Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::fwdConstraint(simulation::Node* node, core::componentmodel::behavior::BaseConstraint* c)
{
    ctime_t t0 = beginProcess(node, c);
    c->projectPosition();
    c->projectVelocity();
    endProcess(node, c, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPropagateFreePositionVisitor::processNodeTopDown(simulation::Node* node)
{
    node->updateSimulationContext();
    return MechanicalVisitor::processNodeTopDown( node);
}
Visitor::Result MechanicalPropagateFreePositionVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setXfree(x);
    mm->setVfree(v);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPropagateFreePositionVisitor::fwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);
    if (!ignoreMask)
    {
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->propagateXfree();
        map->getMechTo()->forceMask.activate(false);
    }
    else
    {
        map->propagateXfree();
    }
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
}
void MechanicalPropagateFreePositionVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}
Visitor::Result MechanicalPropagateFreePositionVisitor::fwdConstraint(simulation::Node* node, core::componentmodel::behavior::BaseConstraint* c)
{
    ctime_t t0 = beginProcess(node, c);
    c->projectFreePosition();
    c->projectFreeVelocity();
    endProcess(node, c, t0);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalResetForceVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setF(res);
    if (!onlyMapped)
        mm->resetForce();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalResetForceVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->resetForce();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalComputeForceVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setF(res);
    mm->accumulateForce();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalComputeForceVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->accumulateForce();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalComputeForceVisitor::fwdForceField(simulation::Node* node, core::componentmodel::behavior::BaseForceField* ff)
{
    //cerr<<"MechanicalComputeForceVisitor::fwdForceField "<<ff->getName()<<endl;
    ctime_t t0 = beginProcess(node, ff);
    ff->addForce();
    endProcess(node, ff, t0);
    return RESULT_CONTINUE;
}

void MechanicalComputeForceVisitor::bwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
//       cerr<<"MechanicalComputeForceVisitor::bwdMechanicalMapping "<<map->getName()<<endl;
    if (accumulate)
    {
        ctime_t t0 = beginProcess(node, map);

        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->accumulateForce();
        map->getMechTo()->forceMask.activate(false);

        endProcess(node, map, t0);
    }
}

void MechanicalComputeForceVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}


Visitor::Result MechanicalComputeDfVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setF(res);
    mm->accumulateDf();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalComputeDfVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->accumulateDf();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalComputeDfVisitor::fwdForceField(simulation::Node* node, core::componentmodel::behavior::BaseForceField* ff)
{
    ctime_t t0 = beginProcess(node, ff);
    if (useV)
        ff->addDForceV();
    else
        ff->addDForce();
    endProcess(node, ff, t0);
    return RESULT_CONTINUE;
}
void MechanicalComputeDfVisitor::bwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    if (accumulate)
    {
        ctime_t t0 = beginProcess(node, map);
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->accumulateDf();
        map->getMechTo()->forceMask.activate(false);
        endProcess(node, map, t0);
    }
}

void MechanicalComputeDfVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}



Visitor::Result MechanicalAddMBKdxVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setF(res);
    mm->accumulateDf();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAddMBKdxVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->accumulateDf();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalAddMBKdxVisitor::fwdForceField(simulation::Node* node, core::componentmodel::behavior::BaseForceField* ff)
{
    ctime_t t0 = beginProcess(node, ff);
    if (useV)
        ff->addMBKv(mFactor, bFactor, kFactor);
    else
        ff->addMBKdx(mFactor, bFactor, kFactor);
    endProcess(node, ff, t0);
    return RESULT_CONTINUE;
}

void MechanicalAddMBKdxVisitor::bwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    if (accumulate)
    {
        ctime_t t0 = beginProcess(node, map);
        map->getMechFrom()->forceMask.activate(true);
        map->getMechTo()->forceMask.activate(true);
        map->accumulateDf();
        map->getMechTo()->forceMask.activate(false);
        endProcess(node, map, t0);
    }
}

void MechanicalAddMBKdxVisitor::bwdMechanicalState(simulation::Node* , core::componentmodel::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}


Visitor::Result MechanicalResetConstraintVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    // mm->setC(res);
    ctime_t t0 = beginProcess(node, mm);
    mm->resetConstraint();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalResetConstraintVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->resetConstraint();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalResetConstraintVisitor::fwdLMConstraint(simulation::Node* node, core::componentmodel::behavior::BaseLMConstraint* c)
{
    // mm->setC(res);
    ctime_t t0 = beginProcess(node, c);
    c->resetConstraint();
    endProcess(node, c, t0);
    return RESULT_CONTINUE;
}

#ifdef SOFA_HAVE_EIGEN2

MechanicalExpressJacobianVisitor::MechanicalExpressJacobianVisitor(simulation::Node* n)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
    helper::vector<core::componentmodel::behavior::BaseLMConstraint*> listC;
    n->get<core::componentmodel::behavior::BaseLMConstraint>(&listC, core::objectmodel::BaseContext::SearchDown);
    for (unsigned int i=0; i<listC.size(); ++i)
    {
        // simulation::Node *node=(simulation::Node*) listC[i]->getContext();
        // ctime_t t0 = beginProcess(node, listC[i]);
        listC[i]->buildJacobian();
        // endProcess(node, listC[i], t0);
    }
    for (unsigned int i=0; i<listC.size(); ++i)
    {
        // simulation::Node *node=(simulation::Node*) listC[i]->getContext();
        // ctime_t t0 = beginProcess(node, listC[i]);
        listC[i]->propagateJacobian();
        // endProcess(node, listC[i], t0);
    }
}

void MechanicalExpressJacobianVisitor::bwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);

    map->accumulateConstraint();
    endProcess(node, map, t0);
}



Visitor::Result MechanicalSolveLMConstraintVisitor::fwdConstraintSolver(simulation::Node* node, core::componentmodel::behavior::ConstraintSolver* s)
{
    typedef core::componentmodel::behavior::BaseMechanicalState::VecId VecId;
    ctime_t t0 = beginProcess(node, s);
    s->solveConstraint(propagateState,state);
    endProcess(node, s, t0);
    return RESULT_PRUNE;
}

Visitor::Result MechanicalWriteLMConstraint::fwdLMConstraint(simulation::Node* node, core::componentmodel::behavior::BaseLMConstraint* c)
{
    ctime_t t0 = beginProcess(node, c);
    c->writeConstraintEquations(order);

    datasC.push_back(c);

    endProcess(node, c, t0);
    return RESULT_CONTINUE;
}

#endif

Visitor::Result MechanicalAccumulateConstraint::fwdConstraint(simulation::Node* node, core::componentmodel::behavior::BaseConstraint* c)
{
    ctime_t t0 = beginProcess(node, c);
    c->applyConstraint(contactId);
    endProcess(node, c, t0);
    return RESULT_CONTINUE;
}

void MechanicalAccumulateConstraint::bwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);
    map->accumulateConstraint();
    endProcess(node, map, t0);
}


Visitor::Result MechanicalApplyConstraintsVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setDx(res);
    //mm->projectResponse();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalApplyConstraintsVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalState* /*mm*/)
{
    //mm->projectResponse();
    return RESULT_CONTINUE;
}

void MechanicalApplyConstraintsVisitor::bwdConstraint(simulation::Node* node, core::componentmodel::behavior::BaseConstraint* c)
{
    ctime_t t0 = beginProcess(node, c);
    c->projectResponse();
    if (W != NULL)
    {
        c->projectResponse(W);
    }
    endProcess(node, c, t0);
}

Visitor::Result MechanicalBeginIntegrationVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->beginIntegration(dt);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalBeginIntegrationVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->beginIntegration(dt);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalEndIntegrationVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->endIntegration(dt);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalEndIntegrationVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->endIntegration(dt);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalComputeComplianceVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* ms)
{
    ctime_t t0 = beginProcess(node, ms);
    ms->getCompliance(_W);
    endProcess(node, ms, t0);
    return RESULT_PRUNE;
}
Visitor::Result MechanicalComputeComplianceVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* ms)
{
    ctime_t t0 = beginProcess(node, ms);
    ms->getCompliance(_W);
    endProcess(node, ms, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalComputeContactForceVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->setF(res);
    mm->accumulateForce();
    endProcess(node, mm, t0);
    return RESULT_PRUNE;
}

Visitor::Result MechanicalComputeContactForceVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    mm->accumulateForce();
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

void MechanicalComputeContactForceVisitor::bwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);
    map->getMechFrom()->forceMask.activate(true);
    map->getMechTo()->forceMask.activate(true);
    map->accumulateForce();
    map->getMechTo()->forceMask.activate(false);
    endProcess(node, map, t0);
}

Visitor::Result MechanicalAddSeparateGravityVisitor::fwdMass(simulation::Node* node, core::componentmodel::behavior::BaseMass* mass)
{
    if( mass->m_separateGravity.getValue() )
    {
        ctime_t t0 = beginProcess(node, mass);
        if (! (res == VecId::velocity())) dynamic_cast<core::componentmodel::behavior::BaseMechanicalState*>(node->getMechanicalState())->setV(res);
        mass->addGravityToV(dt);
        if (! (res == VecId::velocity())) dynamic_cast<core::componentmodel::behavior::BaseMechanicalState*>(node->getMechanicalState())->setV(VecId::velocity());

        endProcess(node, mass, t0);
    }
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPickParticlesVisitor::fwdMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    ctime_t t0 = beginProcess(node, mm);
    //std::cout << "Picking particles on state " << mm->getName() << " within radius " << radius0 << " + dist * " << dRadius << std::endl;

    //We deactivate the Picking with static objects (not simulated)
    core::CollisionModel *c;
    mm->getContext()->get(c, core::objectmodel::BaseContext::Local);
    if (c && !c->isSimulated()) //If it is an obstacle, we don't try to pick
    {
        endProcess(node, mm, t0);
        return RESULT_CONTINUE;
    }

    mm->pickParticles(rayOrigin[0], rayOrigin[1], rayOrigin[2], rayDirection[0], rayDirection[1], rayDirection[2], radius0, dRadius, particles);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPickParticlesVisitor::fwdMappedMechanicalState(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalState* mm)
{
    if (node->mechanicalMapping  && !node->mechanicalMapping->isMechanical())
        return RESULT_PRUNE;
    ctime_t t0 = beginProcess(node, mm);
    mm->pickParticles(rayOrigin[0], rayOrigin[1], rayOrigin[2], rayDirection[0], rayDirection[1], rayDirection[2], radius0, dRadius, particles);
    endProcess(node, mm, t0);
    return RESULT_CONTINUE;
}
Visitor::Result MechanicalPickParticlesVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    if (!map->isMechanical())
        return RESULT_PRUNE;
    return RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa

