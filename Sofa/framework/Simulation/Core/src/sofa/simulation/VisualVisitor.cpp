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
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/Shader.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#ifdef DEBUG_DRAW
#define DO_DEBUG_DRAW true
#else
#define DO_DEBUG_DRAW false
#endif // DEBUG_DRAW

namespace sofa
{

namespace simulation
{

Visitor::Result VisualVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each(this, node, node->object, &VisualVisitor::processObject);
    for_each(this, node, node->visualModel, &VisualVisitor::processVisualModel);
    return RESULT_CONTINUE;
}


Visitor::Result VisualDrawVisitor::processNodeTopDown(simulation::Node* node)
{

    // NB: hasShader is only used when there are visual models and getShader does a graph search when there is no shader,
    // which will most probably be the case when there are no visual models, so we skip the search unless we have visual models.
    hasShader = !node->visualModel.empty() && (node->getShader()!=nullptr);

    for_each(this, node, node->visualModel,     &VisualDrawVisitor::fwdVisualModel);
    this->VisualVisitor::processNodeTopDown(node);

    return RESULT_CONTINUE;
}

void VisualDrawVisitor::processNodeBottomUp(simulation::Node* node)
{
    // don't draw if specified not to do so in the user interface
    if (!vparams->displayFlags().getShowVisualModels())
        return;

    for_each(this, node, node->visualModel,     &VisualDrawVisitor::bwdVisualModel);
}

void VisualDrawVisitor::processObject(simulation::Node* /*node*/, core::objectmodel::BaseObject* o)
{
    if (vparams->pass() == core::visual::VisualParams::Transparent || vparams->pass() == core::visual::VisualParams::Shadow)
    {
        msg_info_when(DO_DEBUG_DRAW, o) << " entering VisualVisitor::draw()" ;

        o->draw(vparams);

        msg_info_when(DO_DEBUG_DRAW, o) << " leaving VisualVisitor::draw()" ;
    }
}

void VisualDrawVisitor::fwdVisualModel(simulation::Node* /*node*/, core::visual::VisualModel* vm)
{
    msg_info_when(DO_DEBUG_DRAW, vm) << " entering VisualVisitor::fwdVisualModel()" ;

    vm->fwdDraw(vparams);

    msg_info_when(DO_DEBUG_DRAW, vm) << " leaving VisualVisitor::fwdVisualModel()" ;
}

void VisualDrawVisitor::bwdVisualModel(simulation::Node* /*node*/,core::visual::VisualModel* vm)
{
    msg_info_when(DO_DEBUG_DRAW, vm) << " entering VisualVisitor::bwdVisualModel()" ;

    vm->bwdDraw(vparams);

    msg_info_when(DO_DEBUG_DRAW, vm) << " leaving VisualVisitor::bwdVisualModel()" ;
}

void VisualDrawVisitor::processVisualModel(simulation::Node* node, core::visual::VisualModel* vm)
{
    // don't draw if specified not to do so in the user interface
    if (!vparams->displayFlags().getShowVisualModels())
        return;

    // don't draw if this component is specifically configured to be disabled
    if (!vm->d_enable.getValue())
        return;

    // don't draw if the component is not in valid state
    if( vm->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid )
        return;

    if(vparams->pass() == core::visual::VisualParams::Shadow)
    {
        msg_info_when(DO_DEBUG_DRAW, vm) << " before calling drawShadow" ;
        vm->drawShadow(vparams);
        msg_info_when(DO_DEBUG_DRAW, vm) << " after calling drawVisual" ;
        return;
    }

    sofa::core::visual::Shader* shader = nullptr;
    if (hasShader)
        shader = node->getShader(subsetsToManage);

    if (shader && shader->isActive())
        shader->start();

    switch(vparams->pass())
    {
        case core::visual::VisualParams::Std:
            msg_info_when(DO_DEBUG_DRAW, vm) << " before calling drawVisual" ;
            vm->drawVisual(vparams);
            msg_info_when(DO_DEBUG_DRAW, vm) << " after calling drawVisual" ;
            break;
        case core::visual::VisualParams::Transparent:
            msg_info_when(DO_DEBUG_DRAW, vm) << " before calling drawTransparent" ;
            vm->drawTransparent(vparams);
            msg_info_when(DO_DEBUG_DRAW, vm) << " after calling drawTransparent" ;
            break;
        default:
            return;
    }

    if (shader && shader->isActive())
        shader->stop();
}

Visitor::Result VisualUpdateVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each(this, node, node->visualModel,              &VisualUpdateVisitor::processVisualModel);

    return RESULT_CONTINUE;
}

void VisualUpdateVisitor::processVisualModel(simulation::Node*, core::visual::VisualModel* vm)
{
    helper::ScopedAdvancedTimer timer("VisualUpdateVisitor process: " + vm->getName());
    vm->updateVisual();
}


Visitor::Result VisualInitVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each(this, node, node->visualModel,              &VisualInitVisitor::processVisualModel);

    return RESULT_CONTINUE;
}
void VisualInitVisitor::processVisualModel(simulation::Node*, core::visual::VisualModel* vm)
{
    vm->initVisual();
}

VisualComputeBBoxVisitor::VisualComputeBBoxVisitor(const core::ExecParams* params)
    : Visitor(params)
{
    minBBox[0] = minBBox[1] = minBBox[2] = 1e10;
    maxBBox[0] = maxBBox[1] = maxBBox[2] = -1e10;
}

Visitor::Result VisualComputeBBoxVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each(this, node, node->behaviorModel,  &VisualComputeBBoxVisitor::processBehaviorModel);
    for_each(this, node, node->mechanicalState, &VisualComputeBBoxVisitor::processMechanicalState);
    for_each(this, node, node->visualModel,     &VisualComputeBBoxVisitor::processVisualModel);

    return RESULT_CONTINUE;
}


void VisualComputeBBoxVisitor::processMechanicalState(simulation::Node*, core::behavior::BaseMechanicalState* vm)
{
    vm->addBBox(minBBox, maxBBox);
}

void VisualComputeBBoxVisitor::processVisualModel(simulation::Node*, core::visual::VisualModel* vm)
{
    vm->addBBox(minBBox, maxBBox);
}

void VisualComputeBBoxVisitor::processBehaviorModel(simulation::Node*, core::BehaviorModel* bm)
{
    bm->addBBox(minBBox, maxBBox);
}

} // namespace simulation

} // namespace sofa

