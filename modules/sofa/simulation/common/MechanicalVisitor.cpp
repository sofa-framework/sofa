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
using std::cerr;
using std::endl;

namespace sofa
{

namespace simulation
{
//Max size for vector to be allowed to be dumped
#define DUMP_VISITOR_MAX_SIZE_VECTOR 20

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
                if (!node->mechanicalMapping->isMechanical())
                {
                    // stop all mechanical computations
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
    if (node->mechanicalState != NULL)
    {
        if (node->mechanicalMapping != NULL)
        {
            if (node->mechanicalMapping->isMechanical())
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
void MechanicalVisitor::printReadVectors(core::componentmodel::behavior::BaseMechanicalState* mm, std::string &info)
{
    if (!mm || !readVector.size() || !Visitor::printActivated) return;


    for (unsigned int i=0; i<Visitor::depthLevel; ++i) info += "\t";
    info += "<Input>\n";

    for (unsigned int i=0; i<readVector.size(); ++i)
    {
        std::ostringstream infoStream;
        for (unsigned int j=0; j<Visitor::depthLevel+1; ++j) info += "\t";

        info += "<Vector name=\"" + readVector[i].getName() + "\"";
        if (mm->getSize() < DUMP_VISITOR_MAX_SIZE_VECTOR)
        {
            mm->printDOF(readVector[i], infoStream);
            info += "value=\"" + infoStream.str() + "\"";
        }
        info += "/>\n";
    }

    for (unsigned int i=0; i<Visitor::depthLevel; ++i) info += "\t";
    info += "</Input>\n";
}

void MechanicalVisitor::printWriteVectors(core::componentmodel::behavior::BaseMechanicalState* mm, std::string &info)
{
    if (!mm || !writeVector.size() || !Visitor::printActivated) return;

    for (unsigned int i=0; i<Visitor::depthLevel; ++i) info += "\t";
    info += "<Output>\n";

    for (unsigned int i=0; i<writeVector.size(); ++i)
    {
        std::ostringstream infoStream;
        for (unsigned int j=0; j<Visitor::depthLevel+1; ++j) info += "\t";
        info += "<Vector name=\"" + writeVector[i].getName() + "\"";

        if (mm->getSize() < DUMP_VISITOR_MAX_SIZE_VECTOR)
        {
            mm->printDOF(writeVector[i], infoStream);
            info += "value=\"" + infoStream.str() + "\"";
        }
        info += "/>\n";
    }

    for (unsigned int i=0; i<Visitor::depthLevel; ++i) info += "\t";
    info += "</Output>\n";
}

void MechanicalVisitor::printReadVectors(simulation::Node* node, core::objectmodel::BaseObject* obj)
{
    if (!Visitor::printActivated) return;
    if (readVector.size())
    {
        std::string info;
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
            dof1=interact->getMechModel1();
            dof2=interact->getMechModel2();
        }
        else
        {
            printReadVectors(node->mechanicalState, info);
            dumpInfo(info);
            return;
        }
        for (unsigned int i=0; i<Visitor::depthLevel; ++i) info +="\t";
        info +="<Components type=\"" + dof1->getClassName() + "\" name=\"" + dof1->getName() + "\">\n";
        printReadVectors(dof1,info);
        for (unsigned int i=0; i<Visitor::depthLevel; ++i) info +="\t";
        info +="</Components>\n";

        for (unsigned int i=0; i<Visitor::depthLevel; ++i) info +="\t";
        info +="<Components type=\"" + dof2->getClassName() + "\" name=\"" + dof2->getName() + "\">\n";
        printReadVectors(dof2,info);
        for (unsigned int i=0; i<Visitor::depthLevel; ++i) info +="\t";
        info +="</Components>\n";
        dumpInfo(info);
    }
}
void MechanicalVisitor::printWriteVectors(simulation::Node* node, core::objectmodel::BaseObject* obj)
{
    if (!Visitor::printActivated) return;
    if (writeVector.size())
    {
        std::string info;
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
            dof1=interact->getMechModel1();
            dof2=interact->getMechModel2();
        }
        else
        {
            printWriteVectors(node->mechanicalState, info);
            dumpInfo(info);
            return;
        }

        for (unsigned int i=0; i<Visitor::depthLevel; ++i) info +="\t";
        info +="<Components type=\"" + dof1->getClassName() + "\" name=\"" + dof1->getName() + "\">\n";
        printWriteVectors(dof1,info);
        for (unsigned int i=0; i<Visitor::depthLevel; ++i) info +="\t";
        info +="</Components>\n";

        for (unsigned int i=0; i<Visitor::depthLevel; ++i) info +="\t";
        info +="<Components type=\"" + dof2->getClassName() + "\" name=\"" + dof2->getName() + "\">\n";
        printWriteVectors(dof2,info);
        for (unsigned int i=0; i<Visitor::depthLevel; ++i) info +="\t";
        info +="</Components>\n";
        dumpInfo(info);
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
    map->propagateDx();
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
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
    map->propagateV();
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
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
    map->propagateX();
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
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
    map->propagateDx();
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
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
    map->propagateX();
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
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
    map->propagateDx();
    map->propagateV();
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
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
MechanicalPropagatePositionAndVelocityVisitor::MechanicalPropagatePositionAndVelocityVisitor(double t, VecId x, VecId v, VecId a) : t(t), x(x), v(v), a(a)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
    //cerr<<"::MechanicalPropagatePositionAndVelocityVisitor"<<endl;
}
#else
MechanicalPropagatePositionAndVelocityVisitor::MechanicalPropagatePositionAndVelocityVisitor(double t, VecId x, VecId v) : t(t), x(x), v(v)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
    //cerr<<"::MechanicalPropagatePositionAndVelocityVisitor"<<endl;
}
#endif

MechanicalPropagatePositionVisitor::MechanicalPropagatePositionVisitor(double t, VecId x) : t(t), x(x)
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
        map->propagateDx();
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
    map->accumulateForce();
    endProcess(node, map, t0);
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
    node->setTime(t);
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
    map->propagateX();
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
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
    map->propagateX();
    map->propagateV();
#ifdef SOFA_SUPPORT_MAPPED_MASS
    map->propagateA();
#endif
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
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
    node->setTime(t);
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
    map->propagateXfree();
    endProcess(node, map, t0);
    return RESULT_CONTINUE;
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
    map->accumulateMask();
    if (accumulate)
    {
        ctime_t t0 = beginProcess(node, map);
        map->accumulateForce();
        endProcess(node, map, t0);
    }
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
        map->accumulateDf();
        endProcess(node, map, t0);
    }
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
        map->accumulateDf();
        endProcess(node, map, t0);
    }
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

Visitor::Result MechanicalAccumulateLMConstraint::fwdLMConstraint(simulation::Node* node, core::componentmodel::behavior::BaseLMConstraint* c)
{

    ctime_t t0 = beginProcess(node, c);
    c->writeConstraintEquations(id);

    datasC.push_back(ConstraintData());
    ConstraintData &entry=datasC[datasC.size()-1];

    //get the corrections to apply
    entry.independentMState[0]=c->getMechModel1();
    entry.independentMState[1]=c->getMechModel2();

    entry.data=c;
    endProcess(node, c, t0);
    return RESULT_CONTINUE;
}

void MechanicalAccumulateLMConstraint::bwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
{
    ctime_t t0 = beginProcess(node, map);
    map->accumulateConstraint();

    for (unsigned int i=0; i<datasC.size(); ++i)
    {
        if ( datasC[i].independentMState[0] == map->getMechTo()) datasC[i].independentMState[0]=map->getMechFrom();
        if ( datasC[i].independentMState[1] == map->getMechTo()) datasC[i].independentMState[1]=map->getMechFrom();
    }

    endProcess(node, map, t0);
}

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
    map->accumulateForce();
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

