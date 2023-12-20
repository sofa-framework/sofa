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
#include <sofa/simulation/BaseMechanicalVisitor.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/LocalStorage.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/BaseInteractionConstraint.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseInteractionProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/behavior/OdeSolver.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/BaseMapping.h>

#include <sofa/core/ConstraintParams.h>
#include <sofa/core/CollisionModel.h>

namespace sofa::simulation
{

using namespace sofa::core;

const std::string BaseMechanicalVisitor::fwdVisitorType = "fwd";
const std::string BaseMechanicalVisitor::bwdVisitorType = "bwd";

BaseMechanicalVisitor::BaseMechanicalVisitor(const sofa::core::ExecParams *params)
        : Visitor(params), root(nullptr), rootData(nullptr)
{
    // mechanical visitors shouldn't be able to acess a sleeping node, only visual visitor should
    canAccessSleepingNode = false;
}

Visitor::Result BaseMechanicalVisitor::processNodeTopDown(simulation::Node *node, VisitorContext *ctx)
{
    for (auto *solver : node->solver)
    {
        if (runVisitorTask(this, ctx, &BaseMechanicalVisitor::fwdOdeSolver, solver, fwdVisitorType) == RESULT_PRUNE)
        {
            return RESULT_PRUNE;
        }
    }

    Result res = RESULT_CONTINUE;
    if (node->mechanicalMapping != nullptr)
    {
        if (stopAtMechanicalMapping(node, node->mechanicalMapping))
        {
            // stop all mechanical computations
            return RESULT_PRUNE;
        }
        res = runVisitorTask(this, ctx, &BaseMechanicalVisitor::fwdMechanicalMapping, &*node->mechanicalMapping, fwdVisitorType);
    }

    if (node->mechanicalState != nullptr)
    {
        if (node->mechanicalMapping != nullptr)
        {
            res = runVisitorTask(this, ctx, &BaseMechanicalVisitor::fwdMappedMechanicalState, &*node->mechanicalState, fwdVisitorType);
        }
        else
        {
            res = runVisitorTask(this, ctx, &BaseMechanicalVisitor::fwdMechanicalState, &*node->mechanicalState, fwdVisitorType);
        }
    }

    if (res == RESULT_PRUNE)
    {
        return RESULT_PRUNE;
    }

    if (node->mass != nullptr &&
        runVisitorTask(this, ctx, &BaseMechanicalVisitor::fwdMass, &*node->mass, fwdVisitorType) == RESULT_PRUNE)
    {
        return RESULT_PRUNE;
    }

    if (for_each(this, ctx, node->constraintSolver, &BaseMechanicalVisitor::fwdConstraintSolver, fwdVisitorType) == RESULT_PRUNE)
    {
        return RESULT_PRUNE;
    }

    if (for_each(this, ctx, node->forceField, &BaseMechanicalVisitor::fwdForceField, fwdVisitorType) == RESULT_PRUNE)
    {
        return RESULT_PRUNE;
    }

    if (for_each(this, ctx, node->interactionForceField, &BaseMechanicalVisitor::fwdInteractionForceField, fwdVisitorType) == RESULT_PRUNE)
    {
        return RESULT_PRUNE;
    }

    if (for_each(this, ctx, node->projectiveConstraintSet, &BaseMechanicalVisitor::fwdProjectiveConstraintSet, fwdVisitorType) == RESULT_PRUNE)
    {
        return RESULT_PRUNE;
    }

    if (for_each(this, ctx, node->constraintSet, &BaseMechanicalVisitor::fwdConstraintSet, fwdVisitorType) == RESULT_PRUNE)
    {
        return RESULT_PRUNE;
    }

    return RESULT_CONTINUE;
}


void BaseMechanicalVisitor::processNodeBottomUp(simulation::Node *node, VisitorContext *ctx)
{
    for_each(this, ctx, node->projectiveConstraintSet, &BaseMechanicalVisitor::bwdProjectiveConstraintSet, bwdVisitorType);
    for_each(this, ctx, node->constraintSet, &BaseMechanicalVisitor::bwdConstraintSet, bwdVisitorType);
    for_each(this, ctx, node->constraintSolver, &BaseMechanicalVisitor::bwdConstraintSolver, bwdVisitorType);

    if (node->mechanicalState != nullptr)
    {
        if (node->mechanicalMapping != nullptr)
        {
            if (!stopAtMechanicalMapping(node, node->mechanicalMapping))
            {
                if (testTags(node->mechanicalState))
                {
                    runVisitorTask(this, ctx, &BaseMechanicalVisitor::bwdMappedMechanicalState, &*node->mechanicalState, bwdVisitorType);
                    runVisitorTask(this, ctx, &BaseMechanicalVisitor::bwdMechanicalMapping, &*node->mechanicalMapping, bwdVisitorType);
                }
            }
        }
        else
        {
            runVisitorTask(this, ctx, &BaseMechanicalVisitor::bwdMechanicalState, &*node->mechanicalState, bwdVisitorType);
        }

    }

    for_each(this, ctx, node->solver, &BaseMechanicalVisitor::bwdOdeSolver, bwdVisitorType);

    if (node == root)
    {
        root = nullptr;
    }
}

Visitor::Result BaseMechanicalVisitor::processNodeTopDown(simulation::Node* node)
{
    if (root == nullptr)
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

/// Process all the InteractionConstraint
Visitor::Result BaseMechanicalVisitor::fwdInteractionProjectiveConstraintSet(VisitorContext* ctx, core::behavior::BaseInteractionProjectiveConstraintSet* c)
{
    return fwdProjectiveConstraintSet(ctx->node, c);
}

/// Process all the InteractionConstraint
Visitor::Result BaseMechanicalVisitor::fwdInteractionConstraint(VisitorContext* ctx, core::behavior::BaseInteractionConstraint* c)
{
    return fwdConstraintSet(ctx->node, c);
}


Visitor::Result BaseMechanicalVisitor::processNodeTopDown(simulation::Node* node, LocalStorage* stack)
{
    SOFA_UNUSED(stack);
    return processNodeTopDown(node);
}


void BaseMechanicalVisitor::processNodeBottomUp(simulation::Node* node, LocalStorage* stack)
{
    SOFA_UNUSED(stack);
    return processNodeBottomUp(node);
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
        }else
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
        }else
        {
            BaseMechanicalState* dof = node->mechanicalState;
            if (dof == nullptr)
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

//const sofa::core::MechanicalParams* BaseMechanicalVisitor::getMechanicalParams() const
//{
//    return dynamic_cast<const sofa::core::MechanicalParams*>(params);
//}

//const sofa::core::ConstraintParams* BaseMechanicalVisitor::getConstraintParams() const
//{
//    return dynamic_cast<const sofa::core::ConstraintParams*>(params);
//}


/// Return a class name for this visitor
/// Only used for debugging / profiling purposes
const char* BaseMechanicalVisitor::getClassName() const { return "MechanicalVisitor"; }

/// Process the OdeSolver
auto BaseMechanicalVisitor::fwdOdeSolver(simulation::Node* /*node*/, sofa::core::behavior::OdeSolver* /*solver*/) -> Result
{
    return RESULT_CONTINUE;
}

/// Process the OdeSolver
auto BaseMechanicalVisitor::fwdOdeSolver(VisitorContext* ctx, sofa::core::behavior::OdeSolver* solver) -> Result
{
    return fwdOdeSolver(ctx->node, solver);
}

/// Process the ConstraintSolver
auto BaseMechanicalVisitor::fwdConstraintSolver(simulation::Node* /*node*/, sofa::core::behavior::ConstraintSolver* /*solver*/) -> Result
{
    return RESULT_CONTINUE;
}

/// Process the ConstraintSolver
auto BaseMechanicalVisitor::fwdConstraintSolver(VisitorContext* ctx, sofa::core::behavior::ConstraintSolver* solver)-> Result
{
    return fwdConstraintSolver(ctx->node, solver);
}

/// Process the BaseMechanicalMapping
auto BaseMechanicalVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/)-> Result
{
    return RESULT_CONTINUE;
}

/// Process the BaseMechanicalMapping
auto BaseMechanicalVisitor::fwdMechanicalMapping(VisitorContext* ctx, sofa::core::BaseMapping* map)-> Result
{
    return fwdMechanicalMapping(ctx->node, map);
}

/// Process the BaseMechanicalState if it is mapped from the parent level
auto BaseMechanicalVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, sofa::core::behavior::BaseMechanicalState* /*mm*/)-> Result
{
    return RESULT_CONTINUE;
}

/// Process the BaseMechanicalState if it is mapped from the parent level
auto BaseMechanicalVisitor::fwdMappedMechanicalState(VisitorContext* ctx, sofa::core::behavior::BaseMechanicalState* mm)-> Result
{
    return fwdMappedMechanicalState(ctx->node, mm);
}

/// Process the BaseMechanicalState if it is not mapped from the parent level
auto BaseMechanicalVisitor::fwdMechanicalState(simulation::Node* /*node*/, sofa::core::behavior::BaseMechanicalState* /*mm*/)-> Result
{
    return RESULT_CONTINUE;
}

/// Process the BaseMechanicalState if it is not mapped from the parent level
auto BaseMechanicalVisitor::fwdMechanicalState(VisitorContext* ctx, sofa::core::behavior::BaseMechanicalState* mm)-> Result
{
    return fwdMechanicalState(ctx->node, mm);
}

/// Process the BaseMass
auto BaseMechanicalVisitor::fwdMass(simulation::Node* /*node*/, sofa::core::behavior::BaseMass* /*mass*/)-> Result
{
    return RESULT_CONTINUE;
}

/// Process the BaseMass
auto BaseMechanicalVisitor::fwdMass(VisitorContext* ctx, sofa::core::behavior::BaseMass* mass)-> Result
{
    return fwdMass(ctx->node, mass);
}

/// Process all the BaseForceField
auto BaseMechanicalVisitor::fwdForceField(simulation::Node* /*node*/, sofa::core::behavior::BaseForceField* /*ff*/)-> Result
{
    return RESULT_CONTINUE;
}


/// Process all the BaseForceField
auto BaseMechanicalVisitor::fwdForceField(VisitorContext* ctx, sofa::core::behavior::BaseForceField* ff)-> Result
{
    return fwdForceField(ctx->node, ff);
}

/// Process all the InteractionForceField
auto BaseMechanicalVisitor::fwdInteractionForceField(simulation::Node* node, sofa::core::behavior::BaseInteractionForceField* ff)-> Result
{
    return fwdForceField(node, ff);
}

/// Process all the InteractionForceField
auto BaseMechanicalVisitor::fwdInteractionForceField(VisitorContext* ctx, sofa::core::behavior::BaseInteractionForceField* ff)-> Result
{
    return fwdInteractionForceField(ctx->node, ff);
}

/// Process all the BaseProjectiveConstraintSet
auto BaseMechanicalVisitor::fwdProjectiveConstraintSet(simulation::Node* /*node*/, sofa::core::behavior::BaseProjectiveConstraintSet* /*c*/)-> Result
{
    return RESULT_CONTINUE;
}

/// Process all the BaseConstraintSet
auto BaseMechanicalVisitor::fwdConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseConstraintSet* /*c*/)-> Result
{
    return RESULT_CONTINUE;
}

/// Process all the BaseProjectiveConstraintSet
auto BaseMechanicalVisitor::fwdProjectiveConstraintSet(VisitorContext* ctx,sofa::core::behavior::BaseProjectiveConstraintSet* c)-> Result
{
    return fwdProjectiveConstraintSet(ctx->node, c);
}

/// Process all the BaseConstraintSet
auto BaseMechanicalVisitor::fwdConstraintSet(VisitorContext* ctx,sofa::core::behavior::BaseConstraintSet* c)-> Result
{
    return fwdConstraintSet(ctx->node, c);
}

/// Process all the InteractionConstraint
auto BaseMechanicalVisitor::fwdInteractionProjectiveConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseInteractionProjectiveConstraintSet* /*c*/)-> Result
{
    return RESULT_CONTINUE;
}

/// Process all the InteractionConstraint
auto BaseMechanicalVisitor::fwdInteractionConstraint(simulation::Node* /*node*/,sofa::core::behavior::BaseInteractionConstraint* /*c*/)-> Result
{
    return RESULT_CONTINUE;
}

/// Process the BaseMechanicalState when it is not mapped from parent level
void BaseMechanicalVisitor::bwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* /*mm*/)
{}

/// Process the BaseMechanicalState when it is not mapped from parent level
void BaseMechanicalVisitor::bwdMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm)
{ bwdMechanicalState(ctx->node, mm); }

/// Process the BaseMechanicalState when it is mapped from parent level
void BaseMechanicalVisitor::bwdMappedMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* /*mm*/)
{}

/// Process the BaseMechanicalState when it is mapped from parent level
void BaseMechanicalVisitor::bwdMappedMechanicalState(VisitorContext* ctx,sofa::core::behavior::BaseMechanicalState* mm)
{ bwdMappedMechanicalState(ctx->node, mm); }

/// Process the BaseMechanicalMapping
void BaseMechanicalVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/)
{}

/// Process the BaseMechanicalMapping
void BaseMechanicalVisitor::bwdMechanicalMapping(VisitorContext* ctx, sofa::core::BaseMapping* map)
{ bwdMechanicalMapping(ctx->node, map); }

/// Process the OdeSolver
void BaseMechanicalVisitor::bwdOdeSolver(simulation::Node* /*node*/,sofa::core::behavior::OdeSolver* /*solver*/)
{}

/// Process the OdeSolver
void BaseMechanicalVisitor::bwdOdeSolver(VisitorContext* ctx,sofa::core::behavior::OdeSolver* solver)
{ bwdOdeSolver(ctx->node, solver); }

/// Process the ConstraintSolver
void BaseMechanicalVisitor::bwdConstraintSolver(simulation::Node* /*node*/,sofa::core::behavior::ConstraintSolver* /*solver*/)
{}

/// Process the ConstraintSolver
void BaseMechanicalVisitor::bwdConstraintSolver(VisitorContext* ctx,sofa::core::behavior::ConstraintSolver* solver)
{ bwdConstraintSolver(ctx->node, solver); }

/// Process all the BaseProjectiveConstraintSet
void BaseMechanicalVisitor::bwdProjectiveConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseProjectiveConstraintSet* /*c*/)
{}

/// Process all the BaseConstraintSet
void BaseMechanicalVisitor::bwdConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseConstraintSet* /*c*/)
{}

/// Process all the BaseProjectiveConstraintSet
void BaseMechanicalVisitor::bwdProjectiveConstraintSet(VisitorContext* ctx,sofa::core::behavior::BaseProjectiveConstraintSet* c)
{ bwdProjectiveConstraintSet(ctx->node, c); }

/// Process all the BaseConstraintSet
void BaseMechanicalVisitor::bwdConstraintSet(VisitorContext* ctx,sofa::core::behavior::BaseConstraintSet* c)
{ bwdConstraintSet(ctx->node, c); }

///@}


/// Return a category name for this action.
/// Only used for debugging / profiling purposes
const char* BaseMechanicalVisitor::getCategoryName() const
{
    return "animate";
}

bool BaseMechanicalVisitor::stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map)
{
    return !map->areForcesMapped();
}



} // namespace sofa::simulation
