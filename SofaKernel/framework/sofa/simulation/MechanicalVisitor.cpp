/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_SIMULATION_MECHANICALVISITOR_CPP
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/Node.h>
#include <iostream>

namespace sofa
{

namespace simulation
{
using std::cerr;
using std::endl;
using namespace sofa::core;

Visitor::Result BaseMechanicalVisitor::processNodeTopDown(simulation::Node* node, VisitorContext* ctx)
{
    Result res = RESULT_CONTINUE;

    for (unsigned i=0; i<node->solver.size() && res!=RESULT_PRUNE; i++ )
    {
        if(testTags(node->solver[i]))
        {
            debug_write_state_before(node->solver[i]);
            ctime_t t = begin(node, node->solver[i], "fwd");
            res = this->fwdOdeSolver(ctx, node->solver[i]);
            end(node, node->solver[i], t);
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
                    //std::cerr << "Pruning " << this->getClassName() << " at " << node->getPathName() << " with non-mechanical mapping" << std::endl;
                    return RESULT_PRUNE;
                }
                //else if (!node->mechanicalMapping->isMechanical()) std::cerr << "Continuing " << this->getClassName() << " at " << node->getPathName() << " with non-mechanical mapping" << std::endl;

                Result res2 = RESULT_CONTINUE;
                if(testTags(node->mechanicalMapping))
                {
                    debug_write_state_before(node->mechanicalMapping);
                    ctime_t t = begin(node, node->mechanicalMapping, "fwd");
                    res = this->fwdMechanicalMapping(ctx, node->mechanicalMapping);
                    end(node, node->mechanicalMapping , t);
                    debug_write_state_after(node->mechanicalMapping);
                }

                if(testTags(node->mechanicalState))
                {
                    debug_write_state_before(node->mechanicalState);
                    ctime_t t = begin(node, node->mechanicalState, "fwd");
                    res2 = this->fwdMappedMechanicalState(ctx, node->mechanicalState);
                    end(node, node->mechanicalState, t);
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
                    ctime_t t = begin(node, node->mechanicalState, "fwd");
                    res = this->fwdMechanicalState(ctx, node->mechanicalState);
                    end(node, node->mechanicalState, t);
                    debug_write_state_after(node->mechanicalState);
                }
            }
        }
        else if (node->mechanicalMapping != NULL) // rare case of a mechanical mapping which controls a state located elsewhere.
        {
            if (stopAtMechanicalMapping(node, node->mechanicalMapping))
            {
                // stop all mechanical computations
                //std::cerr << "Pruning " << this->getClassName() << " at " << node->getPathName() << " with non-mechanical mapping" << std::endl;
                return RESULT_PRUNE;
            }
            //else if (!node->mechanicalMapping->isMechanical()) std::cerr << "Continuing " << this->getClassName() << " at " << node->getPathName() << " with non-mechanical mapping" << std::endl;

            Result res2 = RESULT_CONTINUE;
            if(testTags(node->mechanicalMapping))
            {
                debug_write_state_before(node->mechanicalMapping);
                ctime_t t = begin(node, node->mechanicalMapping, "fwd");
                res = this->fwdMechanicalMapping(ctx, node->mechanicalMapping);
                end(node, node->mechanicalMapping , t);
                debug_write_state_after(node->mechanicalMapping);
            }

            if (res2 == RESULT_PRUNE)
                res = res2;
        }
    }

    if (res != RESULT_PRUNE)
    {
        if (node->mass != NULL)
        {
            if(testTags(node->mass))
            {
                debug_write_state_before(node->mass);
                ctime_t t = begin(node, node->mass, "fwd");
                res = this->fwdMass(ctx, node->mass);
                end(node, node->mass, t);
                debug_write_state_after(node->mass);
            }
        }
    }

    if (res != RESULT_PRUNE)
    {
        res = for_each_r<BaseMechanicalVisitor,VisitorContext,sofa::simulation::Node::Sequence<core::behavior::ConstraintSolver>,core::behavior::ConstraintSolver>(this, ctx, node->constraintSolver, &MechanicalVisitor::fwdConstraintSolver);
    }

    if (res != RESULT_PRUNE)
    {
        res = for_each_r<BaseMechanicalVisitor,VisitorContext,sofa::simulation::Node::Sequence<core::behavior::BaseForceField>,core::behavior::BaseForceField>(this, ctx, node->forceField, &MechanicalVisitor::fwdForceField);
    }

    if (res != RESULT_PRUNE)
    {
        res = for_each_r<BaseMechanicalVisitor,VisitorContext,sofa::simulation::Node::Sequence<core::behavior::BaseInteractionForceField>,core::behavior::BaseInteractionForceField>(this, ctx, node->interactionForceField, &MechanicalVisitor::fwdInteractionForceField);
    }


    if (res != RESULT_PRUNE)
    {
        res = for_each_r<BaseMechanicalVisitor,VisitorContext,sofa::simulation::Node::Sequence<core::behavior::BaseProjectiveConstraintSet>,core::behavior::BaseProjectiveConstraintSet>(this, ctx, node->projectiveConstraintSet, &MechanicalVisitor::fwdProjectiveConstraintSet);
    }

    if (res != RESULT_PRUNE)
    {
        res = for_each_r<BaseMechanicalVisitor,VisitorContext,sofa::simulation::Node::Sequence<core::behavior::BaseConstraintSet>,core::behavior::BaseConstraintSet>(this, ctx, node->constraintSet, &MechanicalVisitor::fwdConstraintSet);
    }


    return res;
}


void BaseMechanicalVisitor::processNodeBottomUp(simulation::Node* node, VisitorContext* ctx)
{
    for_each<BaseMechanicalVisitor,VisitorContext,sofa::simulation::Node::Sequence<core::behavior::BaseProjectiveConstraintSet>,core::behavior::BaseProjectiveConstraintSet>(this, ctx, node->projectiveConstraintSet, &MechanicalVisitor::bwdProjectiveConstraintSet);
    for_each<BaseMechanicalVisitor,VisitorContext,sofa::simulation::Node::Sequence<core::behavior::BaseConstraintSet>,core::behavior::BaseConstraintSet>(this, ctx, node->constraintSet, &MechanicalVisitor::bwdConstraintSet);
    for_each<BaseMechanicalVisitor,VisitorContext,sofa::simulation::Node::Sequence<core::behavior::ConstraintSolver>,core::behavior::ConstraintSolver>(this, ctx, node->constraintSolver, &MechanicalVisitor::bwdConstraintSolver);

    if (node->mechanicalState != NULL)
    {
        if (node->mechanicalMapping != NULL)
        {
            if (!stopAtMechanicalMapping(node, node->mechanicalMapping))
            {
                if(testTags(node->mechanicalState))
                {
                    ctime_t t = begin(node, node->mechanicalState, "bwd");
                    this->bwdMappedMechanicalState(ctx, node->mechanicalState);
                    end(node, node->mechanicalState, t);
                    t = begin(node, node->mechanicalMapping, "bwd");
                    this->bwdMechanicalMapping(ctx, node->mechanicalMapping);
                    end(node, node->mechanicalMapping, t);
                }
            }
        }
        else
        {
            if(testTags(node->mechanicalState))
            {
                ctime_t t = begin(node, node->mechanicalState, "bwd");
                this->bwdMechanicalState(ctx, node->mechanicalState);
                end(node, node->mechanicalState, t);
            }
        }

    }

    for (unsigned i=0; i<node->solver.size(); i++ )
    {
        if(testTags(node->solver[i]))
        {
            ctime_t t = begin(node, node->solver[i], "bwd");
            this->bwdOdeSolver(ctx, node->solver[i]);
            end(node, node->solver[i], t);
        }
    }

    if (node == root)
    {
        root = NULL;
    }
}


Visitor::Result BaseMechanicalVisitor::processNodeTopDown(simulation::Node* node)
{
    if (root == NULL)
    {
        root = node;
    }

    VisitorContext ctx;
    ctx.root = root;
    ctx.node = node;
    ctx.nodeData = rootData;
    return processNodeTopDown(node, &ctx);
}


void BaseMechanicalVisitor::processNodeBottomUp(simulation::Node* node)
{
    VisitorContext ctx;
    ctx.root = root;
    ctx.node = node;
    ctx.nodeData = rootData;
    processNodeBottomUp(node, &ctx);
}


Visitor::Result BaseMechanicalVisitor::processNodeTopDown(simulation::Node* node, LocalStorage* stack)
{
    if (root == NULL)
    {
        root = node;
    }

    VisitorContext ctx;
    ctx.root = root;
    ctx.node = node;
    ctx.nodeData = rootData;

    const bool writeData = writeNodeData();
    if (writeData)
    {
        // create temporary accumulation buffer for parallel reductions (dot products)
        if (node != root)
        {
            SReal* parentData = stack->empty() ? rootData : (SReal*)stack->top();
            ctx.nodeData = new SReal(0.0);
            setNodeData(node, ctx.nodeData, parentData);
            stack->push(ctx.nodeData);
        }
    }

    return processNodeTopDown(node, &ctx);
}


void BaseMechanicalVisitor::processNodeBottomUp(simulation::Node* node, LocalStorage* stack)
{
    VisitorContext ctx;
    ctx.root = root;
    ctx.node = node;
    ctx.nodeData = rootData;
    SReal* parentData = rootData;

    const bool writeData = writeNodeData();

    if (writeData)
    {
        // use temporary accumulation buffer for parallel reductions (dot products)
        if (node != root)
        {
            ctx.nodeData = (SReal*)stack->pop();
            parentData = stack->empty() ? rootData : (SReal*)stack->top();
        }
    }

    processNodeBottomUp(node, &ctx);

    if (writeData && parentData != ctx.nodeData)
        addNodeData(node, parentData, ctx.nodeData);
}


#ifdef SOFA_DUMP_VISITOR_INFO

void BaseMechanicalVisitor::printReadVectors(core::behavior::BaseMechanicalState* mm)
{
    if (!mm || !readVector.size() || !Visitor::printActivated || !Visitor::outputStateVector) return;

    printNode("Input");
    for (unsigned int i=0; i<readVector.size(); ++i)
    {
        sofa::core::ConstVecId id = readVector[i].getId(mm);
        if( ! id.isNull() ) printVector(mm, id );
    }
    printCloseNode("Input");
}


void BaseMechanicalVisitor::printWriteVectors(core::behavior::BaseMechanicalState* mm)
{
    if (!mm || !writeVector.size() || !Visitor::printActivated || !Visitor::outputStateVector) return;

    printNode("Output");
    for (unsigned int i=0; i<writeVector.size(); ++i)
    {
        sofa::core::VecId id = writeVector[i].getId(mm);
        if( ! id.isNull() ) printVector(mm, id );
    }
    printCloseNode("Output");
}


void BaseMechanicalVisitor::printReadVectors(simulation::Node* node, core::objectmodel::BaseObject* obj)
{
    using sofa::core::behavior::BaseInteractionForceField;
    using sofa::core::behavior::BaseInteractionProjectiveConstraintSet;
    using sofa::core::behavior::BaseInteractionConstraint;

    if (!Visitor::printActivated || !Visitor::outputStateVector) return;

    if (readVector.size())
    {
        core::behavior::BaseMechanicalState *dof1, *dof2;

        if (BaseInteractionForceField* interact = dynamic_cast< BaseInteractionForceField* > (obj))
        {
            dof1 = interact->getMechModel1();
            dof2 = interact->getMechModel2();
        }
        else if (BaseInteractionProjectiveConstraintSet* interact = dynamic_cast< BaseInteractionProjectiveConstraintSet* > (obj))
        {
            dof1 = interact->getMechModel1();
            dof2 = interact->getMechModel2();
        }
        else if (BaseInteractionConstraint* interact = dynamic_cast< BaseInteractionConstraint* > (obj))
        {
            dof1 = interact->getMechModel1();
            dof2 = interact->getMechModel2();
        }/*
        else if (BaseLMConstraint* interact = dynamic_cast< BaseLMConstraint* > (obj))
        {
                dof1 = interact->getConstrainedMechModel1();
                dof2 = interact->getConstrainedMechModel2();
}*/
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


void BaseMechanicalVisitor::printWriteVectors(simulation::Node* node, core::objectmodel::BaseObject* obj)
{
    using sofa::core::behavior::BaseInteractionForceField;
    using sofa::core::behavior::BaseInteractionProjectiveConstraintSet;
    using sofa::core::behavior::BaseInteractionConstraint;
    using sofa::core::behavior::BaseMechanicalState;

    if (!Visitor::printActivated) return;

    if (writeVector.size())
    {
        BaseMechanicalState *dof1, *dof2;

        if (BaseInteractionForceField* interact = dynamic_cast< BaseInteractionForceField* > (obj))
        {
            dof1 = interact->getMechModel1();
            dof2 = interact->getMechModel2();
        }
        else if (BaseInteractionProjectiveConstraintSet* interact = dynamic_cast< BaseInteractionProjectiveConstraintSet* > (obj))
        {
            dof1 = interact->getMechModel1();
            dof2 = interact->getMechModel2();
        }
        else if (BaseInteractionConstraint* interact = dynamic_cast< BaseInteractionConstraint* > (obj))
        {
            dof1 = interact->getMechModel1();
            dof2 = interact->getMechModel2();
        }/*
        else if (BaseLMConstraint* interact = dynamic_cast< BaseLMConstraint* > (obj))
        {
                dof1 = interact->getConstrainedMechModel1();
                dof2 = interact->getConstrainedMechModel2();
}*/
        else
        {
            BaseMechanicalState* dof = node->mechanicalState;
            if (dof == NULL)
                node->getContext()->get(dof);
            printWriteVectors(dof);
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


Visitor::ctime_t BaseMechanicalVisitor::begin(simulation::Node* node, core::objectmodel::BaseObject* obj, const std::string &info)
{
    ctime_t t=Visitor::begin(node, obj, info);
    printReadVectors(node, obj);
    return t;
}


void BaseMechanicalVisitor::end(simulation::Node* node, core::objectmodel::BaseObject* obj, ctime_t t0)
{
    printWriteVectors(node, obj);
    Visitor::end(node, obj, t0);
}
#endif


Visitor::Result MechanicalGetDimensionVisitor::fwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
{
    const unsigned int n = mm->getMatrixSize();
    *ctx->nodeData += (SReal)n;
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalIntegrationVisitor::fwdOdeSolver(simulation::Node* node, core::behavior::OdeSolver* obj)
{
    SReal nextTime = node->getTime() + dt;
    MechanicalBeginIntegrationVisitor beginVisitor( this->params, dt );
    node->execute(&beginVisitor);

    sofa::core::MechanicalParams mparams(*this->params);
    mparams.setDt(dt);

    {
        unsigned int constraintId=0;
        core::ConstraintParams cparams;
        simulation::MechanicalAccumulateConstraint(&cparams, core::MatrixDerivId::holonomicC(), constraintId).execute(node);

    }
    //cerr<<"MechanicalIntegrationVisitor::fwdOdeSolver start solve obj"<<endl;
    obj->solve(params, dt);
    //cerr<<"MechanicalIntegrationVisitor::fwdOdeSolver endVisitor ok"<<endl;

    //cerr<<"MechanicalIntegrationVisitor::fwdOdeSolver end solve obj"<<endl;
    //obj->propagatePositionAndVelocity(nextTime,core::VecCoordId::position(),core::VecDerivId::velocity());

    MechanicalPropagatePositionAndVelocityVisitor(&mparams, nextTime,VecCoordId::position(),VecDerivId::velocity(),
#ifdef SOFA_SUPPORT_MAPPED_MASS
            VecDerivId::dx(),
#endif
            true).execute( node );

    MechanicalEndIntegrationVisitor endVisitor( this->params, dt );
    node->execute(&endVisitor);

    return RESULT_PRUNE;
}


Visitor::Result MechanicalIntegrationVisitor::fwdInteractionForceField(simulation::Node* /*node*/, core::behavior::BaseInteractionForceField* obj)
{
    MultiVecDerivId   ffId      = VecDerivId::externalForce();
    MechanicalParams m_mparams(*this->params);
    m_mparams.setDt(this->dt);

    obj->addForce(&m_mparams, ffId);
    return RESULT_CONTINUE;
}

template< VecType vtype>
Visitor::Result MechanicalVInitVisitor<vtype>::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState *mm)
{
    mm->vInit(this->params, vDest.getId(mm), vSrc.getId(mm));
    return RESULT_CONTINUE;
}

template< VecType vtype>
Visitor::Result MechanicalVInitVisitor<vtype>::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    if (m_propagate)
    {
        mm->vInit(this->params, vDest.getId(mm), vSrc.getId(mm));
    }

    return RESULT_CONTINUE;
}


template< VecType vtype>
std::string  MechanicalVInitVisitor<vtype>::getInfos() const
{
    std::string name = "[" + vDest.getName() + "]";
    return name;
}

template< VecType vtype>
Visitor::Result  MechanicalVAvailVisitor<vtype>::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->vAvail( this->params, v );
    this->states.insert(mm);
    return RESULT_CONTINUE;
}

template< VecType vtype>
std::string  MechanicalVAvailVisitor<vtype>::getInfos() const
{
    std::string name="[" + v.getName() + "]";
    return name;
}



template< VecType vtype>
Visitor::Result MechanicalVAllocVisitor<vtype>::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->vAlloc(this->params, v.getId(mm) );
    return RESULT_CONTINUE;
}


template< VecType vtype>
std::string  MechanicalVAllocVisitor<vtype>::getInfos() const
{
    std::string name="[" + v.getName() + "]";
    return name;
}

template< VecType vtype>
Visitor::Result MechanicalVReallocVisitor<vtype>::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState *mm)
{
    mm->vRealloc( this->params, this->getId(mm) );
    return RESULT_CONTINUE;
}

template< VecType vtype>
Visitor::Result MechanicalVReallocVisitor<vtype>::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    if (m_propagate)
    {
        mm->vRealloc(this->params, this->getId(mm) );
    }

    return RESULT_CONTINUE;
}

template< VecType vtype>
Visitor::Result MechanicalVReallocVisitor<vtype>::fwdInteractionForceField(simulation::Node* /*node*/, core::behavior::BaseInteractionForceField* ff)
{
    if (m_interactionForceField)
    {
        core::behavior::BaseMechanicalState* mm = ff->getMechModel1();
        mm->vRealloc( this->params, this->getId(mm) );
        mm = ff->getMechModel2();
        mm->vRealloc( this->params, this->getId(mm) );
    }

    return RESULT_CONTINUE;
}

template< VecType vtype>
typename MechanicalVReallocVisitor<vtype>::MyVecId MechanicalVReallocVisitor<vtype>::getId( core::behavior::BaseMechanicalState* mm )
{
    MyVecId vid = v->getId(mm);
    if( vid.isNull() ) // not already allocated
    {
        vid = MyVecId(MyVecId::V_FIRST_DYNAMIC_INDEX);
        mm->vAvail( this->params, vid );
        v->setId( mm, vid );
    }
    return vid;
}

template< VecType vtype>
std::string  MechanicalVReallocVisitor<vtype>::getInfos() const
{
    std::string name = "[" + v->getName() + "]";
    return name;
}


template< VecType vtype>
Visitor::Result MechanicalVFreeVisitor<vtype>::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->vFree( this->params, v.getId(mm) );
    return RESULT_CONTINUE;
}

template< VecType vtype>
Visitor::Result MechanicalVFreeVisitor<vtype>::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->vFree( this->params, v.getId(mm) );
    return RESULT_CONTINUE;
}

template< VecType vtype>
Visitor::Result MechanicalVFreeVisitor<vtype>::fwdInteractionForceField(simulation::Node* /*node*/, core::behavior::BaseInteractionForceField* ff)
{
    if( interactionForceField )
    {
        core::behavior::BaseMechanicalState* mm = ff->getMechModel1();
        mm->vFree( this->params, v.getId(mm) );
        mm = ff->getMechModel2();
        mm->vFree( this->params, v.getId(mm) );
    }
    return RESULT_CONTINUE;
}

template< VecType vtype>
std::string  MechanicalVFreeVisitor<vtype>::getInfos() const
{
    std::string name="[" + v.getName() + "]";
    return name;
}

Visitor::Result MechanicalVOpVisitor::fwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMechanicalState, model "<<mm->getName()<<endl;
    if (!only_mapped)
        mm->vOp(this->params, v.getId(mm) ,a.getId(mm),b.getId(mm),((ctx->nodeData && *ctx->nodeData != 1.0) ? *ctx->nodeData * f : f) );
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalVOpVisitor::fwdMappedMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMappedMechanicalState, model "<<mm->getName()<<endl;
    if (mapped || only_mapped)
    {
        mm->vOp(this->params, v.getId(mm) ,a.getId(mm),b.getId(mm),((ctx->nodeData && *ctx->nodeData != 1.0) ? *ctx->nodeData * f : f) );
    }
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalVMultiOpVisitor::fwdMechanicalState(VisitorContext* /*ctx*/, core::behavior::BaseMechanicalState* mm)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMechanicalState, model "<<mm->getName()<<endl;
    mm->vMultiOp(this->params, ops );
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalVMultiOpVisitor::fwdMappedMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
{
    //cerr<<"    MechanicalVOpVisitor::fwdMappedMechanicalState, model "<<mm->getName()<<endl;
    //mm->vMultiOp(ops);
    if (mapped)
    {
        if (ctx->nodeData && *ctx->nodeData != 1.0)
        {
            VMultiOp ops2 = ops;
            const SReal fact = *ctx->nodeData;
            for (VMultiOp::iterator it = ops2.begin(), itend = ops2.end(); it != itend; ++it)
                for (unsigned int i = 1; i < it->second.size(); ++i)
                    it->second[i].second *= fact;
            mm->vMultiOp(this->params, ops2 );
        }
        else
        {
            mm->vMultiOp(this->params, ops );
        }
    }
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalVDotVisitor::fwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
{
    *ctx->nodeData += mm->vDot(this->params, a.getId(mm),b.getId(mm) );
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalVNormVisitor::fwdMechanicalState(VisitorContext* /*ctx*/, core::behavior::BaseMechanicalState* mm)
{
    if( l>0 ) accum += mm->vSum(this->params, a.getId(mm), l );
    else {
        SReal mmax = mm->vMax(this->params, a.getId(mm) );
        if( mmax>accum ) accum=mmax;
    }
    return RESULT_CONTINUE;
}

SReal MechanicalVNormVisitor::getResult() const
{
    if( l>1 )
        return exp( log(accum) / l);
    else return accum;
}

#if 0
/// Parallel code
Visitor::Result MechanicalVDotVisitor::processNodeTopDown(simulation::Node* /*node*/, LocalStorage* stack)
{
    SReal* localTotal = new SReal(0.0);
    stack->push(localTotal);
    if (node->mechanicalState && !node->mechanicalMapping)
    {
        core::behavior::BaseMechanicalState* mm = node->mechanicalState;
        *localTotal += mm->vDot(a,b);
    }
    return RESULT_CONTINUE;
}


/// Parallel code
void MechanicalVDotVisitor::processNodeBottomUp(simulation::Node* /*node*/, LocalStorage* stack)
{
    SReal* localTotal = static_cast<SReal*>(stack->pop());
    SReal* parentTotal = static_cast<SReal*>(stack->top());
    if (!parentTotal)
        *total += *localTotal; // root
    else
        *parentTotal += *localTotal;
    delete localTotal;
}
#endif

Visitor::Result MechanicalPropagateDxVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    //<TO REMOVE>
    //mm->setDx(dx);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPropagateDxVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (!ignoreMask)
    {
        ForceMaskActivate(map->getMechFrom());
        ForceMaskActivate(map->getMechTo());
    }
    //map->propagateDx();
    //std::cout << getClassName() << getInfos() << " : " << map->getName() << "::applyJ()" << std::endl;
    map->applyJ(mparams, dx, dx);

    if (!ignoreMask)
    {
        ForceMaskDeactivate(map->getMechTo());
    }

    return RESULT_CONTINUE;
}


void MechanicalPropagateDxVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    if (!ignoreMask)
    {
        mm->forceMask.activate(false);
    }
}


Visitor::Result MechanicalPropagateVelocityVisitor::fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
{
    if (applyProjections)
    {
        c->projectVelocity(mparams, v);
    }
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPropagateDxAndResetForceVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    //mm->setDx(dx);
    //mm->setF(f);
    mm->resetForce(this->params, f.getId(mm));
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPropagateDxAndResetForceVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (!ignoreMask)
    {
        ForceMaskActivate(map->getMechFrom() );
        ForceMaskActivate(map->getMechTo() );
    }
    //map->propagateDx();
    map->applyJ(mparams, dx, dx);

    if (!ignoreMask)
    {
        ForceMaskDeactivate(map->getMechTo() );
    }

    return RESULT_CONTINUE;
}


void MechanicalPropagateDxAndResetForceVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}


Visitor::Result MechanicalPropagateDxAndResetForceVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->resetForce(this->params, f.getId(mm));
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPropagatePositionAndResetForceVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    //mm->setX(x);
    //mm->setF(f);
    mm->resetForce(this->params, f.getId(mm));
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPropagatePositionAndResetForceVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (!ignoreMask)
    {
        ForceMaskActivate(map->getMechFrom() );
        ForceMaskActivate(map->getMechTo() );
    }
    //map->propagateX();
    map->apply(mparams, x, x);
    if (!ignoreMask)
    {
        ForceMaskDeactivate(map->getMechTo() );
    }


    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPropagatePositionAndResetForceVisitor::fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
{
    if (applyProjections)
    {
        c->projectPosition(mparams, x);
    }
    return RESULT_CONTINUE;
}

void MechanicalPropagatePositionAndResetForceVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}


Visitor::Result MechanicalPropagatePositionAndResetForceVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->resetForce(this->params, f.getId(mm));
    return RESULT_CONTINUE;
}



Visitor::Result MechanicalAddMDxVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    //mm->setF(res);
    //if (!dx.isNull())
    //    mm->setDx(dx);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalAddMDxVisitor::fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* mass)
{
    mass->addMDx(mparams, res, factor);
    return RESULT_PRUNE;
}


Visitor::Result MechanicalAccFromFVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    /////<TO REMOVE>
    //mm->setDx(a);
    //mm->setF(f);
    /// \todo Check presence of Mass
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalAccFromFVisitor::fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* mass)
{
    mass->accFromF(mparams, a);
    return RESULT_CONTINUE;
}



MechanicalPropagatePositionAndVelocityVisitor::MechanicalPropagatePositionAndVelocityVisitor(
    const sofa::core::MechanicalParams* mparams,
    SReal time, MultiVecCoordId x, MultiVecDerivId v,
#ifdef SOFA_SUPPORT_MAPPED_MASS
    MultiVecDerivId a,
#endif
    bool m)
    : MechanicalVisitor(mparams), currentTime(time), x(x), v(v),
#ifdef SOFA_SUPPORT_MAPPED_MASS
      a(a),
#endif
      ignoreMask(m), applyProjections(true)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
    //cerr<<"::MechanicalPropagatePositionAndVelocityVisitor"<<endl;
}

#ifdef SOFA_SUPPORT_MAPPED_MASS

Visitor::Result MechanicalAddMDxVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (!mparams->dx().isNull())
    {
        ForceMaskActivate(map->getMechFrom() );
        ForceMaskActivate(map->getMechTo() );

        //map->propagateDx();
        map->applyJ(mparams, dx, dx); // TODO : check
        ForceMaskDeactivate( map->getMechTo() );

    }
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalAddMDxVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->resetForce(mparams, res.getId(mm));
    return RESULT_CONTINUE;
}


void MechanicalAddMDxVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    ForceMaskActivate(map->getMechFrom() );
    ForceMaskActivate(map->getMechTo() );

    //map->accumulateForce();
    map->applyJT(mparams, res, res);
    ForceMaskDeactivate( map->getMechTo() );
}


void MechanicalAddMDxVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}

#else

Visitor::Result MechanicalAddMDxVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
{
    return RESULT_PRUNE;
}


Visitor::Result MechanicalAddMDxVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    return RESULT_PRUNE;
}

#endif // SOFA_SUPPORT_MAPPED_MASS

Visitor::Result MechanicalProjectJacobianMatrixVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
{
    return RESULT_PRUNE;
}
Visitor::Result MechanicalProjectJacobianMatrixVisitor::fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
{
    c->projectJacobianMatrix(mparams, cId);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalProjectVelocityVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
{
    return RESULT_PRUNE;
}
Visitor::Result MechanicalProjectVelocityVisitor::fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
{
    c->projectVelocity(mparams, vel);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalProjectPositionVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
{
    return RESULT_PRUNE;
}
Visitor::Result MechanicalProjectPositionVisitor::fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
{
    c->projectPosition(mparams, pos);
    return RESULT_CONTINUE;
}





MechanicalPropagatePositionVisitor::MechanicalPropagatePositionVisitor(const sofa::core::MechanicalParams* mparams, SReal t, MultiVecCoordId x, bool m )
    : MechanicalVisitor(mparams) , t(t), x(x), ignoreMask(m), applyProjections(true)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
    //cerr<<"::MechanicalPropagatePositionVisitor"<<endl;
}


Visitor::Result MechanicalPropagatePositionVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    //mm->setX(x);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPropagatePositionVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (!ignoreMask)
    {
        ForceMaskActivate(map->getMechFrom() );
        ForceMaskActivate(map->getMechTo() );
    }
    //map->propagateX();
    map->apply(mparams, x, x);

    if (!ignoreMask)
    {
        ForceMaskDeactivate( map->getMechTo() );
    }

    return RESULT_CONTINUE;
}



void MechanicalPropagatePositionVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}

Visitor::Result MechanicalPropagatePositionVisitor::fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
{
    if (applyProjections)
    {
        c->projectPosition(mparams, x);
    }
    return RESULT_CONTINUE;
}

#ifdef SOFA_SUPPORT_MAPPED_MASS
Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
#else
Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
#endif
{
    //mm->setX(x);
    //mm->setV(v);
#ifdef SOFA_SUPPORT_MAPPED_MASS
    //    mm->setDx(a);
    mm->resetAcc(mparams, a.getId(mm));
#endif
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (!ignoreMask)
    {
        ForceMaskActivate(map->getMechFrom() );
        ForceMaskActivate(map->getMechTo() );
    }
    //map->propagateX();
    //map->propagateV();
    //#ifdef SOFA_SUPPORT_MAPPED_MASS
    //map->propagateA();
    //#endif
    map->apply(mparams, x, x);
    map->applyJ(mparams, v, v);
#ifdef SOFA_SUPPORT_MAPPED_MASS
    map->computeAccFromMapping(mparams, a, v, a);
#endif
    if (!ignoreMask)
    {
        ForceMaskDeactivate( map->getMechTo() );
    }
    return RESULT_CONTINUE;
}

void MechanicalPropagatePositionAndVelocityVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}

Visitor::Result MechanicalPropagatePositionAndVelocityVisitor::fwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
{
    if (applyProjections)
    {
        c->projectPosition(mparams, x);
        c->projectVelocity(mparams, v);
    }
//    cerr<<"MechanicalPropagatePositionAndVelocityVisitor::fwdProjectiveConstraintSet" << endl;
    return RESULT_CONTINUE;
}




MechanicalPropagateVelocityVisitor::MechanicalPropagateVelocityVisitor(
    const sofa::core::MechanicalParams* mparams,
    SReal time, MultiVecDerivId v,
#ifdef SOFA_SUPPORT_MAPPED_MASS
    MultiVecDerivId a,
#endif
    bool m)
    : MechanicalVisitor(mparams), currentTime(time), v(v),
#ifdef SOFA_SUPPORT_MAPPED_MASS
      a(a),
#endif
      ignoreMask(m),
      applyProjections(true)
{
#ifdef SOFA_DUMP_VISITOR_INFO

    setReadWriteVectors();
#endif
    //cerr<<"::MechanicalPropagateVelocityVisitor"<<endl;
}

#ifdef SOFA_SUPPORT_MAPPED_MASS
Visitor::Result MechanicalPropagateVelocityVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
#else
Visitor::Result MechanicalPropagateVelocityVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
#endif
{
    //mm->setV(v);
#ifdef SOFA_SUPPORT_MAPPED_MASS
    //    mm->setDx(a);
    mm->resetAcc(mparams, a.getId(mm));
#endif
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPropagateVelocityVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (!ignoreMask)
    {
        ForceMaskActivate(map->getMechFrom() );
        ForceMaskActivate(map->getMechTo() );
    }
    //map->propagateV();
    //#ifdef SOFA_SUPPORT_MAPPED_MASS
    //map->propagateA();
    //#endif
    map->applyJ(mparams, v, v);
#ifdef SOFA_SUPPORT_MAPPED_MASS
    map->computeAccFromMapping(mparams, a, v, a);
#endif
    if (!ignoreMask)
    {
        ForceMaskDeactivate( map->getMechTo() );
    }
    return RESULT_CONTINUE;
}

void MechanicalPropagateVelocityVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}

#ifdef SOFA_SUPPORT_MAPPED_MASS

MechanicalSetPositionAndVelocityVisitor::MechanicalSetPositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams ,
        SReal time, MultiVecCoordId x, MultiVecDerivId v, MultiVecDerivId a)
    : MechanicalVisitor(mparams) , t(time), x(x), v(v), a(a)
{

#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif // SOFA_DUMP_VISITOR_INFO
    //	cerr<<"::MechanicalSetPositionAndVelocityVisitor"<<endl;
}
#else
MechanicalSetPositionAndVelocityVisitor::MechanicalSetPositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams ,
        SReal time, MultiVecCoordId x, MultiVecDerivId v)
    : MechanicalVisitor(mparams) , t(time), x(x), v(v)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
    //	cerr<<"::MechanicalSetPositionAndVelocityVisitor"<<endl;
}

#endif // SOFA_SUPPORT_MAPPED_MASS

#ifdef SOFA_SUPPORT_MAPPED_MASS
Visitor::Result MechanicalSetPositionAndVelocityVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
#else
Visitor::Result MechanicalSetPositionAndVelocityVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
#endif
{
    //mm->setX(x);
    //mm->setV(v);
#ifdef SOFA_SUPPORT_MAPPED_MASS
    //mm->setDx(a);
    mm->resetAcc(mparams, a.getId(mm));
#endif
    return RESULT_CONTINUE;
}



Visitor::Result MechanicalResetForceVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    //mm->setF(res);
    if (!onlyMapped)
        mm->resetForce(this->params, res.getId(mm));
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalResetForceVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->resetForce(this->params, res.getId(mm));
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalComputeForceVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
//    cerr << "MechanicalComputeForceVisitor::fwdMechanicalState " << node->getName() << endl;
    mm->accumulateForce(this->params, res.getId(mm));
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalComputeForceVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->accumulateForce(this->params, res.getId(mm));
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalComputeForceVisitor::fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
{
    //cerr<<"MechanicalComputeForceVisitor::fwdForceField "<<ff->getName()<<endl;
    if( !neglectingCompliance || !ff->isCompliance.getValue() ) ff->addForce(this->mparams, res);
    else ff->updateForceMask(); // compliances must update the force mask too
    return RESULT_CONTINUE;
}

void MechanicalComputeForceVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    //       cerr<<"MechanicalComputeForceVisitor::bwdMechanicalMapping "<<map->getName()<<endl;
    if (accumulate)
    {
        ForceMaskActivate(map->getMechFrom() );
        ForceMaskActivate(map->getMechTo() );

        //map->accumulateForce();
        map->applyJT(mparams, res, res);

        ForceMaskDeactivate( map->getMechTo() );
    }
}


void MechanicalComputeForceVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}


Visitor::Result MechanicalComputeDfVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /* mm */)
{
    //<TO REMOVE>
    //mm->setF(res);
    //mm->accumulateDf();
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalComputeDfVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    //mm->accumulateDf();
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalComputeDfVisitor::fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
{
    if( !ff->isCompliance.getValue() ) ff->addDForce(this->mparams, res);
    return RESULT_CONTINUE;
}


void MechanicalComputeDfVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (accumulate)
    {
        ForceMaskActivate(map->getMechFrom() );
        ForceMaskActivate(map->getMechTo() );
//        cerr<<"MechanicalComputeDfVisitor::bwdMechanicalMapping"<<endl;
        map->applyJT(mparams, res, res);  // apply material stiffness: variation of force below the mapping
        if( mparams->kFactor() ) map->applyDJT(mparams, res, res); // apply geometric stiffness: variation due to a change of mapping, with a constant force below the mapping
        ForceMaskDeactivate( map->getMechTo() );
    }
}


void MechanicalComputeDfVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}

Visitor::Result MechanicalComputeGeometricStiffness::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    map->updateK( mparams, childForce );
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalAddMBKdxVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    //mm->setF(res);
    //<TO REMOVE>
    //mm->accumulateDf();
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalAddMBKdxVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    //mm->accumulateDf();
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalAddMBKdxVisitor::fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
{
    if( !ff->isCompliance.getValue() )
        ff->addMBKdx( this->mparams, res);
    else
        ff->addMBKdx( &mparamsWithoutStiffness, res);
    return RESULT_CONTINUE;
}


void MechanicalAddMBKdxVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (accumulate)
    {
        ForceMaskActivate(map->getMechFrom() );
        ForceMaskActivate(map->getMechTo() );

        //map->accumulateDf();
        map->applyJT(mparams, res, res);
        if( mparams->kFactor() ) map->applyDJT(mparams, res, res);
        ForceMaskDeactivate( map->getMechTo() );
    }
}


void MechanicalAddMBKdxVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    mm->forceMask.activate(false);
}


Visitor::Result MechanicalResetConstraintVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    // mm->setC(res);
    mm->resetConstraint(this->params);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalResetConstraintVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->resetConstraint(this->params);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalResetConstraintVisitor::fwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* c)
{
    // mm->setC(res);
    c->resetConstraint();
    return RESULT_CONTINUE;
}



Visitor::Result MechanicalWriteLMConstraint::fwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* c)
{
    if (core::behavior::BaseLMConstraint* LMc=c->toBaseLMConstraint())
    {
        LMc->writeConstraintEquations(offset, id, order);
        datasC.push_back(LMc);
    }
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalAccumulateConstraint::fwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* c)
{
    c->setConstraintId(contactId);
    c->buildConstraintMatrix(cparams, res, contactId);
    return RESULT_CONTINUE;
}


void MechanicalAccumulateConstraint::bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    map->applyJT(cparams, res, res);
}


Visitor::Result MechanicalApplyConstraintsVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    //mm->setDx(res);
    //mm->projectResponse();
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalApplyConstraintsVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    //mm->projectResponse();
    return RESULT_CONTINUE;
}


void MechanicalApplyConstraintsVisitor::bwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
{
    c->projectResponse(mparams, res);
    if (W != NULL)
    {
        c->projectResponse(mparams, W);
    }
}


Visitor::Result MechanicalBeginIntegrationVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->beginIntegration(dt);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalBeginIntegrationVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->beginIntegration(dt);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalEndIntegrationVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->endIntegration(params, dt);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalEndIntegrationVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->endIntegration(params, dt);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalComputeContactForceVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    //mm->setF(res);
    mm->accumulateForce(this->params, res.getId(mm));
    return RESULT_PRUNE;
}


Visitor::Result MechanicalComputeContactForceVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    mm->accumulateForce(this->params, res.getId(mm));
    return RESULT_CONTINUE;
}


void MechanicalComputeContactForceVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    ForceMaskActivate(map->getMechFrom() );
    ForceMaskActivate(map->getMechTo() );
    //map->accumulateForce();
    map->applyJT(mparams, res, res);
    ForceMaskDeactivate(map->getMechTo() );
}


Visitor::Result MechanicalAddSeparateGravityVisitor::fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* mass)
{
    if( mass->m_separateGravity.getValue() )
    {
        //<TO REMOVE>
        //if (! (res == VecId::velocity())) dynamic_cast<core::behavior::BaseMechanicalState*>(node->getMechanicalState())->setV(res);
        mass->addGravityToV(this->mparams, res);
        //if (! (res == VecId::velocity())) dynamic_cast<core::behavior::BaseMechanicalState*>(node->getMechanicalState())->setV(VecId::velocity());
    }
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPickParticlesVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
//    std::cerr << "MechanicalPickParticlesVisitor::fwdMechanicalState, Picking particles on state " << mm->getName() << " within radius " << radius0 << " + dist * " << dRadius << std::endl;

    if (mm->hasTag(tagNoPicking)) // picking disabled for this model
        return RESULT_CONTINUE;

    //We deactivate the Picking with static objects (not simulated)
    core::CollisionModel *c;
    mm->getContext()->get(c, core::objectmodel::BaseContext::Local);
    if (c && !c->isSimulated()) //If it is an obstacle, we don't try to pick
    {
        return RESULT_CONTINUE;
    }
    mm->pickParticles(this->params, rayOrigin[0], rayOrigin[1], rayOrigin[2], rayDirection[0], rayDirection[1], rayDirection[2], radius0, dRadius, particles);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPickParticlesVisitor::fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
{
    if (mm->hasTag(tagNoPicking)) // picking disabled for this model
        return RESULT_CONTINUE;

    if (node->mechanicalMapping  && !node->mechanicalMapping->isMechanical())
        return RESULT_PRUNE;
    mm->pickParticles(this->params, rayOrigin[0], rayOrigin[1], rayOrigin[2], rayDirection[0], rayDirection[1], rayDirection[2], radius0, dRadius, particles);
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPickParticlesVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (!map->isMechanical())
        return RESULT_PRUNE;
    return RESULT_CONTINUE;
}



/// get the closest pickable particle
void MechanicalPickParticlesVisitor::getClosestParticle( core::behavior::BaseMechanicalState*& mstate, unsigned int& indexCollisionElement, defaulttype::Vector3& point, SReal& rayLength )
{
    mstate = NULL;

    if( particles.empty() ) return;

    rayLength = std::numeric_limits<SReal>::max();

    core::behavior::BaseMechanicalState* mstatei;
    unsigned int indexCollisionElementi;
    defaulttype::Vector3 pointi;
    SReal rayLengthi;

    // particles are sorted from their distance to the ray
    // threshold for valid particles is the shortest distance + small tolerance relative to ray length
    SReal dmax = particles.begin()->first + radius0*1e-10;

    for( Particles::const_iterator it=particles.begin(), itend=particles.end() ; it!=itend ; ++it )
    {
        if( it->first > dmax ) break; // from now on, particles are too far from the ray

        // get current valid particle
        mstatei = it->second.first;
        indexCollisionElementi = it->second.second;
        pointi[0] = mstatei->getPX(indexCollisionElementi);
        pointi[1] = mstatei->getPY(indexCollisionElementi);
        pointi[2] = mstatei->getPZ(indexCollisionElementi);
        rayLengthi = (pointi-rayOrigin)*rayDirection;

        if( rayLengthi < rayLength ) // keep the closest particle to the ray origin
        {
            mstate = mstatei;
            indexCollisionElement = indexCollisionElementi;
            point = pointi;
            rayLength = rayLengthi;
        }
    }
}

Visitor::Result MechanicalPickParticlesWithTagsVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
	//    std::cerr << "MechanicalPickParticlesVisitor::fwdMechanicalState, Picking particles on state " << mm->getName() << " within radius " << radius0 << " + dist * " << dRadius << std::endl;


    if(!isComponentTagIncluded(mm))
		return RESULT_CONTINUE;

	//We deactivate the Picking with static objects (not simulated)
	core::CollisionModel *c;
	mm->getContext()->get(c, core::objectmodel::BaseContext::Local);
	if (c && !c->isSimulated()) //If it is an obstacle, we don't try to pick
	{
		return RESULT_CONTINUE;
	}
	mm->pickParticles(this->params, rayOrigin[0], rayOrigin[1], rayOrigin[2], rayDirection[0], rayDirection[1], rayDirection[2], radius0, dRadius, particles);

	return RESULT_CONTINUE;
}


Visitor::Result MechanicalPickParticlesWithTagsVisitor::fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
{
	if (node->mechanicalMapping  && !node->mechanicalMapping->isMechanical())
		return RESULT_PRUNE;

    if(!isComponentTagIncluded(mm))
        return RESULT_CONTINUE;

	mm->pickParticles(this->params, rayOrigin[0], rayOrigin[1], rayOrigin[2], rayDirection[0], rayDirection[1], rayDirection[2], radius0, dRadius, particles);
	return RESULT_CONTINUE;
}


Visitor::Result MechanicalPickParticlesWithTagsVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
	if (!map->isMechanical())
		return RESULT_PRUNE;
	return RESULT_CONTINUE;
}

bool MechanicalPickParticlesWithTagsVisitor::isComponentTagIncluded(const behavior::BaseMechanicalState *mm)
{
    bool tagOk = mustContainAllTags || tags.empty();
    for(std::list<sofa::core::objectmodel::Tag>::const_iterator tagIt = tags.begin(); tags.end() != tagIt; ++tagIt)
    {
        if (!mm->hasTag(*tagIt)) // picking disabled for this model
        {
            if(mustContainAllTags)
            {
                tagOk = false;
                break;
            }
        }
        else if (mm->hasTag(*tagIt)) // picking disabled for this model
        {
            tagOk = true;

            if(!mustContainAllTags)
                break;
        }
    }
    return tagOk;
}

/// get the closest pickable particle
void MechanicalPickParticlesWithTagsVisitor::getClosestParticle( core::behavior::BaseMechanicalState*& mstate, unsigned int& indexCollisionElement, defaulttype::Vector3& point, SReal& rayLength )
{
	mstate = NULL;

	if( particles.empty() ) return;

	rayLength = std::numeric_limits<SReal>::max();

	core::behavior::BaseMechanicalState* mstatei;
	unsigned int indexCollisionElementi;
	defaulttype::Vector3 pointi;
	SReal rayLengthi;

	// particles are sorted from their distance to the ray
	// threshold for valid particles is the shortest distance + small tolerance relative to ray length
	SReal dmax = particles.begin()->first + radius0*1e-10;

	for( Particles::const_iterator it=particles.begin(), itend=particles.end() ; it!=itend ; ++it )
	{
		if( it->first > dmax ) break; // from now on, particles are too far from the ray

		// get current valid particle
		mstatei = it->second.first;
		indexCollisionElementi = it->second.second;
		pointi[0] = mstatei->getPX(indexCollisionElementi);
		pointi[1] = mstatei->getPY(indexCollisionElementi);
		pointi[2] = mstatei->getPZ(indexCollisionElementi);
		rayLengthi = (pointi-rayOrigin)*rayDirection;

		if( rayLengthi < rayLength ) // keep the closest particle to the ray origin
		{
			mstate = mstatei;
			indexCollisionElement = indexCollisionElementi;
			point = pointi;
			rayLength = rayLengthi;
		}
	}
}



Visitor::Result MechanicalVSizeVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    ConstVecId id = v.getId(mm);
    if( !id.isNull() )
        *result += mm->vSize(this->params, id );
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalVSizeVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    ConstVecId id = v.getId(mm);
    if( !id.isNull() )
        *result += mm->vSize(this->params, id );
    return RESULT_CONTINUE;
}


template class SOFA_SIMULATION_CORE_API MechanicalVAvailVisitor<V_COORD>;
template class SOFA_SIMULATION_CORE_API MechanicalVAvailVisitor<V_DERIV>;
template class SOFA_SIMULATION_CORE_API MechanicalVAllocVisitor<V_COORD>;
template class SOFA_SIMULATION_CORE_API MechanicalVAllocVisitor<V_DERIV>;
template class SOFA_SIMULATION_CORE_API MechanicalVReallocVisitor<V_COORD>;
template class SOFA_SIMULATION_CORE_API MechanicalVReallocVisitor<V_DERIV>;
template class SOFA_SIMULATION_CORE_API MechanicalVFreeVisitor<V_COORD>;
template class SOFA_SIMULATION_CORE_API MechanicalVFreeVisitor<V_DERIV>;
template class SOFA_SIMULATION_CORE_API MechanicalVInitVisitor<V_COORD>;
template class SOFA_SIMULATION_CORE_API MechanicalVInitVisitor<V_DERIV>;


} // namespace simulation

} // namespace sofa

