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
#include <sofa/simulation/common/VisualVisitor.h>

//#define DEBUG_DRAW

namespace sofa
{

namespace simulation
{


Visitor::Result VisualDrawVisitor::processNodeTopDown(simulation::Node* node)
{
    glPushMatrix();
#ifdef SOFA_SUPPORT_MOVING_FRAMES
    double glMatrix[16];
    node->getPositionInWorld().writeOpenGlMatrix(glMatrix);
    glMultMatrixd( glMatrix );
#endif
    hasShader = (node->getShader()!=NULL);

    for_each(this, node, node->visualModel,     &VisualDrawVisitor::fwdVisualModel);
    this->VisualVisitor::processNodeTopDown(node);

    glPopMatrix();
    return RESULT_CONTINUE;
}

void VisualDrawVisitor::processNodeBottomUp(simulation::Node* node)
{
    for_each(this, node, node->visualModel,     &VisualDrawVisitor::bwdVisualModel);
}

void VisualDrawVisitor::processObject(simulation::Node* /*node*/, core::objectmodel::BaseObject* o)
{
    if (pass == core::VisualModel::Std || pass == core::VisualModel::Shadow)
    {
#ifdef DEBUG_DRAW
        std::cerr << ">" << o->getClassName() << "::draw() of " << o->getName() << std::endl;
#endif
        o->draw();
#ifdef DEBUG_DRAW
        std::cerr << "<" << o->getClassName() << "::draw() of " << o->getName() << std::endl;
#endif
    }
}

void VisualDrawVisitor::fwdVisualModel(simulation::Node* /*node*/, core::VisualModel* vm)
{
#ifdef DEBUG_DRAW
    std::cerr << ">" << vm->getClassName() << "::fwdDraw() of " << vm->getName() << std::endl;
#endif
    vm->fwdDraw(pass);
#ifdef DEBUG_DRAW
    std::cerr << "<" << vm->getClassName() << "::fwdDraw() of " << vm->getName() << std::endl;
#endif
}

void VisualDrawVisitor::bwdVisualModel(simulation::Node* /*node*/, core::VisualModel* vm)
{
#ifdef DEBUG_DRAW
    std::cerr << ">" << vm->getClassName() << "::bwdDraw() of " << vm->getName() << std::endl;
#endif
    vm->bwdDraw(pass);
#ifdef DEBUG_DRAW
    std::cerr << "<" << vm->getClassName() << "::bwdDraw() of " << vm->getName() << std::endl;
#endif
}

void VisualDrawVisitor::processVisualModel(simulation::Node* node, core::VisualModel* vm)
{
    //cerr<<"VisualDrawVisitor::processVisualModel "<<vm->getName()<<endl;
    sofa::core::Shader* shader = NULL;
    if (hasShader)
        shader = dynamic_cast<sofa::core::Shader*>(node->getShader());

    switch(pass)
    {
    case core::VisualModel::Std:
    {
        if (shader && shader->isActive())
            shader->start();
#ifdef DEBUG_DRAW
        std::cerr << ">" << vm->getClassName() << "::drawVisual() of " << vm->getName() << std::endl;
#endif
        vm->drawVisual();
#ifdef DEBUG_DRAW
        std::cerr << "<" << vm->getClassName() << "::drawVisual() of " << vm->getName() << std::endl;
#endif
        if (shader && shader->isActive())
            shader->stop();
        break;
    }
    case core::VisualModel::Transparent:
    {
        if (shader && shader->isActive())
            shader->start();
#ifdef DEBUG_DRAW
        std::cerr << ">" << vm->getClassName() << "::drawTransparent() of " << vm->getName() << std::endl;
#endif
        vm->drawTransparent();
#ifdef DEBUG_DRAW
        std::cerr << "<" << vm->getClassName() << "::drawTransparent() of " << vm->getName() << std::endl;
#endif
        if (shader && shader->isActive())
            shader->stop();
        break;
    }
    case core::VisualModel::Shadow:
#ifdef DEBUG_DRAW
        std::cerr << ">" << vm->getClassName() << "::drawShadow() of " << vm->getName() << std::endl;
#endif
        vm->drawShadow();
#ifdef DEBUG_DRAW
        std::cerr << "<" << vm->getClassName() << "::drawShadow() of " << vm->getName() << std::endl;
#endif
        break;
    }
}


Visitor::Result VisualUpdateVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each(this, node, node->visualModel,              &VisualUpdateVisitor::processVisualModel);
    for_each(this, node, node->visualModelInVisualGraph, &VisualUpdateVisitor::processVisualModel);

    {
        for (simulation::Node::ChildIterator itChild = node->childInVisualGraph.begin(); itChild != node->childInVisualGraph.end(); ++itChild)
        {
            simulation::Node *child=*itChild;
            child->execute<VisualUpdateVisitor>(params);
        }
    }
    return RESULT_CONTINUE;
}

void VisualUpdateVisitor::processVisualModel(simulation::Node*, core::VisualModel* vm)
{
    vm->updateVisual();
}
#ifdef SOFA_SMP
void ParallelVisualUpdateVisitor::processVisualModel(simulation::Node*, core::VisualModel* vm)
{
    vm->parallelUpdateVisual();
}
#endif

Visitor::Result VisualInitVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each(this, node, node->visualModel,              &VisualInitVisitor::processVisualModel);
    for_each(this, node, node->visualModelInVisualGraph, &VisualInitVisitor::processVisualModel);

    {
        for (simulation::Node::ChildIterator itChild = node->childInVisualGraph.begin(); itChild != node->childInVisualGraph.end(); ++itChild)
        {
            simulation::Node *child=*itChild;
            child->execute<VisualInitVisitor>(params);
        }
    }
    return RESULT_CONTINUE;
}
void VisualInitVisitor::processVisualModel(simulation::Node*, core::VisualModel* vm)
{
    vm->initVisual();
}

VisualComputeBBoxVisitor::VisualComputeBBoxVisitor(const core::ExecParams* params)
    : Visitor(params)
{
    minBBox[0] = minBBox[1] = minBBox[2] = 1e10;
    maxBBox[0] = maxBBox[1] = maxBBox[2] = -1e10;
}

void VisualComputeBBoxVisitor::processMechanicalState(simulation::Node*, core::behavior::BaseMechanicalState* vm)
{
    vm->addBBox(minBBox, maxBBox);
}
void VisualComputeBBoxVisitor::processVisualModel(simulation::Node*, core::VisualModel* vm)
{
    vm->addBBox(minBBox, maxBBox);
}
void VisualComputeBBoxVisitor::processBehaviorModel(simulation::Node*, core::BehaviorModel* bm)
{
    bm->addBBox(minBBox, maxBBox);
}

} // namespace simulation

} // namespace sofa

