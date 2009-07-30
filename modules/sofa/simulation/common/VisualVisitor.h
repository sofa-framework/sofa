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
#ifndef SOFA_SIMULATION_TREE_VISUALACTION_H
#define SOFA_SIMULATION_TREE_VISUALACTION_H

#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/core/VisualModel.h>
#include <sofa/helper/system/gl.h>
#include <iostream>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace simulation
{

using std::cerr;
using std::endl;
class SOFA_SIMULATION_COMMON_API VisualVisitor : public Visitor
{
public:
    virtual void processVisualModel(simulation::Node* node, core::VisualModel* vm) = 0;
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
};

class SOFA_SIMULATION_COMMON_API VisualDrawVisitor : public VisualVisitor
{
public:
    typedef core::VisualModel::Pass Pass;
    Pass pass;
    bool hasShader;
    VisualDrawVisitor(Pass pass = core::VisualModel::Std)
        : pass(pass)
    {
    }
    virtual Result processNodeTopDown(simulation::Node* node);
    virtual void processNodeBottomUp(simulation::Node* node);
    virtual void fwdVisualModel(simulation::Node* node, core::VisualModel* vm);
    virtual void processVisualModel(simulation::Node* node, core::VisualModel* vm);
    virtual void processObject(simulation::Node* node, core::objectmodel::BaseObject* o);
    virtual void bwdVisualModel(simulation::Node* node, core::VisualModel* vm);
    virtual const char* getClassName() const { return "VisualDrawVisitor"; }
#ifdef SOFA_DUMP_VISITOR_INFO
    virtual void printInfo(const core::objectmodel::BaseContext*,bool )  {return;}
#endif
};

class SOFA_SIMULATION_COMMON_API VisualUpdateVisitor : public VisualVisitor
{
public:
    virtual void processVisualModel(simulation::Node*, core::VisualModel* vm);
    virtual const char* getClassName() const { return "VisualUpdateVisitor"; }
};

class SOFA_SIMULATION_COMMON_API VisualInitVisitor : public VisualVisitor
{
public:
    virtual void processVisualModel(simulation::Node*, core::VisualModel* vm);
    virtual const char* getClassName() const { return "VisualInitVisitor"; }
};

class SOFA_SIMULATION_COMMON_API VisualComputeBBoxVisitor : public Visitor
{
public:
    double minBBox[3];
    double maxBBox[3];
    VisualComputeBBoxVisitor();

    virtual void processMechanicalState(simulation::Node*, core::componentmodel::behavior::BaseMechanicalState* vm);
    virtual void processVisualModel(simulation::Node*, core::VisualModel* vm);

    virtual Result processNodeTopDown(simulation::Node* node)
    {
        for_each(this, node, node->mechanicalState, &VisualComputeBBoxVisitor::processMechanicalState);
        for_each(this, node, node->visualModel,     &VisualComputeBBoxVisitor::processVisualModel);

        return RESULT_CONTINUE;
    }
    virtual const char* getClassName() const { return "VisualComputeBBoxVisitor"; }


};

} // namespace simulation

} // namespace sofa

#endif
