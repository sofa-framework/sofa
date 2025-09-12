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
#ifndef SOFA_SIMULATION_TREE_VISUALACTION_H
#define SOFA_SIMULATION_TREE_VISUALACTION_H

#include <sofa/simulation/Visitor.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/BaseVisualStyle.h>


namespace sofa::simulation
{

class SOFA_SIMULATION_CORE_API VisualVisitor : public Visitor
{
public:
    VisualVisitor(core::visual::VisualParams* visuparams)
        : Visitor(sofa::core::visual::visualparams::castToExecParams(visuparams))
        ,vparams(visuparams)
    {}

    virtual void processVisualModel(simulation::Node* node, core::visual::VisualModel* vm) = 0;
    virtual void fwdProcessVisualStyle(simulation::Node* node, core::visual::BaseVisualStyle* vm);
    virtual void bwdProcessVisualStyle(simulation::Node* node, core::visual::BaseVisualStyle* vm);
    virtual void processObject(simulation::Node* /*node*/, core::objectmodel::BaseObject* /*o*/) {}

    Result processNodeTopDown(simulation::Node* node) override;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "visual"; }
    const char* getClassName() const override { return "VisualVisitor"; }

    /// visual visitor must be executed as a tree, such as forward and backward orders are coherent
    bool treeTraversal(TreeTraversalRepetition& repeat) override { repeat=NO_REPETITION; return true; }

protected:
    core::visual::VisualParams* vparams;
};


class SOFA_SIMULATION_CORE_API VisualDrawVisitor : public VisualVisitor
{
public:
    bool hasShader;
    VisualDrawVisitor(core::visual::VisualParams* visuparams)
    : VisualVisitor(visuparams)
    {};
    Result processNodeTopDown(simulation::Node* node) override;
    void processNodeBottomUp(simulation::Node* node) override;
    virtual void fwdVisualModel(simulation::Node* node, core::visual::VisualModel* vm);
    void processVisualModel(simulation::Node* node, core::visual::VisualModel* vm) override;
    void processObject(simulation::Node* node, core::objectmodel::BaseObject* o) override;
    virtual void bwdVisualModel(simulation::Node* node, core::visual::VisualModel* vm);
    const char* getClassName() const override { return "VisualDrawVisitor"; }
#ifdef SOFA_DUMP_VISITOR_INFO
    virtual void printInfo(const core::objectmodel::BaseContext*,bool ) override {return;}
#endif
};

class SOFA_SIMULATION_CORE_API VisualUpdateVisitor : public VisualVisitor
{
public:
    VisualUpdateVisitor(core::visual::VisualParams* visuparams)
        : VisualVisitor(visuparams)
    {}

    virtual void processVisualModel(simulation::Node*, core::visual::VisualModel* vm) override;
    Result processNodeTopDown(simulation::Node* node) override;
    void processNodeBottomUp(simulation::Node* node) override;

    const char* getClassName() const override { return "VisualUpdateVisitor"; }

};

class SOFA_SIMULATION_CORE_API VisualInitVisitor : public VisualVisitor
{
public:
    VisualInitVisitor(core::visual::VisualParams* visuparams)
        : VisualVisitor(visuparams)
    {}
    
    virtual void processVisualModel(simulation::Node*, core::visual::VisualModel* vm) override;
    Result processNodeTopDown(simulation::Node* node) override;
    const char* getClassName() const override { return "VisualInitVisitor"; }
};



class SOFA_SIMULATION_CORE_API VisualComputeBBoxVisitor : public Visitor
{
public:
    SReal minBBox[3];
    SReal maxBBox[3];
    VisualComputeBBoxVisitor(const core::ExecParams* eparams);

    virtual void processBehaviorModel(simulation::Node*, core::BehaviorModel* vm);
    virtual void processMechanicalState(simulation::Node*, core::behavior::BaseMechanicalState* vm);
    virtual void processVisualModel(simulation::Node*, core::visual::VisualModel* vm);

    Result processNodeTopDown(simulation::Node* node) override;
    const char* getClassName() const override { return "VisualComputeBBoxVisitor"; }
};


class SOFA_SIMULATION_CORE_API VisualClearVisitor : public VisualVisitor
{
public:
    VisualClearVisitor(core::visual::VisualParams* visuparams) : VisualVisitor(visuparams)
    {}

    void processVisualModel(simulation::Node*, core::visual::VisualModel* vm) override
    {
        vm->clearVisual();
    }
    const char* getClassName() const override { return "VisualClearVisitor"; }
};


} // namespace sofa::simulation


#endif
