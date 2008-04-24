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
#ifndef SOFA_SIMULATION_TREE_VISUALACTION_H
#define SOFA_SIMULATION_TREE_VISUALACTION_H

#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/Visitor.h>
#include <sofa/core/VisualModel.h>
#include <sofa/helper/system/gl.h>
#include <iostream>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

using std::cerr;
using std::endl;

namespace sofa
{

namespace simulation
{

namespace tree
{

class VisualVisitor : public Visitor
{
public:
    virtual void processVisualModel(GNode* node, core::VisualModel* vm) = 0;
    virtual void processObject(GNode* /*node*/, core::objectmodel::BaseObject* /*o*/) {}

    virtual Result processNodeTopDown(GNode* node)
    {
        for_each(this, node, node->object, &VisualVisitor::processObject);
        for_each(this, node, node->visualModel, &VisualVisitor::processVisualModel);
        return RESULT_CONTINUE;
    }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "visual"; }
};

class VisualDrawVisitor : public VisualVisitor
{
public:
    typedef core::VisualModel::Pass Pass;
    Pass pass;
    bool hasShader;
    VisualDrawVisitor(Pass pass = core::VisualModel::Std)
        : pass(pass)
    {
    }
    virtual Result processNodeTopDown(GNode* node);
    virtual void processNodeBottomUp(GNode* node);
    virtual void fwdVisualModel(GNode* node, core::VisualModel* vm);
    virtual void processVisualModel(GNode* node, core::VisualModel* vm);
    virtual void processObject(GNode* node, core::objectmodel::BaseObject* o);
    virtual void bwdVisualModel(GNode* node, core::VisualModel* vm);
};

class VisualUpdateVisitor : public VisualVisitor
{
public:
    virtual void processVisualModel(GNode*, core::VisualModel* vm);
};

class VisualInitVisitor : public VisualVisitor
{
public:
    virtual void processVisualModel(GNode*, core::VisualModel* vm);
};

class VisualComputeBBoxVisitor : public Visitor
{
public:
    typedef sofa::defaulttype::Vector3::value_type Real_Sofa;
    Real_Sofa minBBox[3];
    Real_Sofa maxBBox[3];
    VisualComputeBBoxVisitor();

    virtual void processMechanicalState(GNode*, core::componentmodel::behavior::BaseMechanicalState* vm);
    virtual void processVisualModel(GNode*, core::VisualModel* vm);

    virtual Result processNodeTopDown(GNode* node)
    {
        for_each(this, node, node->mechanicalState, &VisualComputeBBoxVisitor::processMechanicalState);
        for_each(this, node, node->visualModel,     &VisualComputeBBoxVisitor::processVisualModel);

        return RESULT_CONTINUE;
    }


};

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
