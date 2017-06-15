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
#ifndef SOFA_SIMULATION_TREE_VISUALACTION_H
#define SOFA_SIMULATION_TREE_VISUALACTION_H

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ExecParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/helper/system/gl.h>
#include <iostream>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace core
{
namespace visual
{
class VisualParams;
} // namespace visual
} // namespace core

namespace simulation
{


class SOFA_SIMULATION_CORE_API VisualVisitor : public Visitor
{
public:
    VisualVisitor(core::visual::VisualParams* params)
        : Visitor(params)
        ,vparams(params)
    {}

    virtual void processVisualModel(simulation::Node* node, core::visual::VisualModel* vm) = 0;
    virtual void processObject(simulation::Node* /*node*/, core::objectmodel::BaseObject* /*o*/) {}

    virtual Result processNodeTopDown(simulation::Node* node)
    {
        for_each(this, node, node->object, &VisualVisitor::processObject);
        for_each(this, node, node->visualModel, &VisualVisitor::processVisualModel);
        return RESULT_CONTINUE;
    }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "visual"; }
    virtual const char* getClassName() const { return "VisualVisitor"; }

    /// visual visitor must be executed as a tree, such as forward and backward orders are coherent
    virtual bool treeTraversal(TreeTraversalRepetition& repeat) { repeat=NO_REPETITION; return true; }

protected:
    core::visual::VisualParams* vparams;
};

class SOFA_SIMULATION_CORE_API VisualDrawVisitor : public VisualVisitor
{
public:
    bool hasShader;
    VisualDrawVisitor(core::visual::VisualParams* params)
        : VisualVisitor(params)
    {
    }
    virtual Result processNodeTopDown(simulation::Node* node);
    virtual void processNodeBottomUp(simulation::Node* node);
    virtual void fwdVisualModel(simulation::Node* node, core::visual::VisualModel* vm);
    virtual void processVisualModel(simulation::Node* node, core::visual::VisualModel* vm);
    virtual void processObject(simulation::Node* node, core::objectmodel::BaseObject* o);
    virtual void bwdVisualModel(simulation::Node* node, core::visual::VisualModel* vm);
    virtual const char* getClassName() const { return "VisualDrawVisitor"; }
#ifdef SOFA_DUMP_VISITOR_INFO
    virtual void printInfo(const core::objectmodel::BaseContext*,bool )  {return;}
#endif
};

class SOFA_SIMULATION_CORE_API VisualUpdateVisitor : public Visitor
{
public:
    VisualUpdateVisitor(const core::ExecParams* params) : Visitor(params) {}

    virtual void processVisualModel(simulation::Node*, core::visual::VisualModel* vm);
    virtual Result processNodeTopDown(simulation::Node* node);

    virtual const char* getClassName() const { return "VisualUpdateVisitor"; }
};

class SOFA_SIMULATION_CORE_API VisualInitVisitor : public Visitor
{
public:
    VisualInitVisitor(const core::ExecParams* params):Visitor(params) {}

    virtual void processVisualModel(simulation::Node*, core::visual::VisualModel* vm);
    virtual Result processNodeTopDown(simulation::Node* node);
    virtual const char* getClassName() const { return "VisualInitVisitor"; }
};
#ifdef SOFA_SMP
class SOFA_SIMULATION_CORE_API ParallelVisualUpdateVisitor : public Visitor
{
public:
    ParallelVisualUpdateVisitor(const core::ExecParams* params) : Visitor(params) {}

    virtual void processVisualModel(simulation::Node*, core::visual::VisualModel* vm);
    virtual const char* getClassName() const { return "ParallelVisualUpdateVisitor"; }
};
#endif


class SOFA_SIMULATION_CORE_API VisualComputeBBoxVisitor : public Visitor
{
public:
    SReal minBBox[3];
    SReal maxBBox[3];
    VisualComputeBBoxVisitor(const core::ExecParams* params);

    virtual void processBehaviorModel(simulation::Node*, core::BehaviorModel* vm);
    virtual void processMechanicalState(simulation::Node*, core::behavior::BaseMechanicalState* vm);
    virtual void processVisualModel(simulation::Node*, core::visual::VisualModel* vm);

    virtual Result processNodeTopDown(simulation::Node* node)
    {
        for_each(this, node, node->behaviorModel,  &VisualComputeBBoxVisitor::processBehaviorModel);
        for_each(this, node, node->mechanicalState, &VisualComputeBBoxVisitor::processMechanicalState);
        for_each(this, node, node->visualModel,     &VisualComputeBBoxVisitor::processVisualModel);

        return RESULT_CONTINUE;
    }
    virtual const char* getClassName() const { return "VisualComputeBBoxVisitor"; }
};


class SOFA_SIMULATION_CORE_API VisualClearVisitor : public VisualVisitor
{
public:
    VisualClearVisitor(core::visual::VisualParams* params) : VisualVisitor(params)
    {}

    virtual void processVisualModel(simulation::Node*, core::visual::VisualModel* vm)
    {
        vm->clearVisual();
    }
    virtual const char* getClassName() const { return "VisualClearVisitor"; }
};


} // namespace simulation

} // namespace sofa

#endif
