#include "VisualPickVisitor.h"
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
//#define DEBUG_DRAW
namespace sofa
{

namespace simplegui
{

VisualPickVisitor::VisualPickVisitor(core::visual::VisualParams* params)
    : VisualVisitor(params)
{
    pickedId = 0;
}


simulation::Visitor::Result VisualPickVisitor::processNodeTopDown(simulation::Node* node)
{
#ifdef SOFA_SUPPORT_MOVING_FRAMES
    glPushMatrix();
    double glMatrix[16];
    node->getPositionInWorld().writeOpenGlMatrix(glMatrix);
    glMultMatrixd( glMatrix );
#endif
//    cerr <<"VisualPickVisitor::processNodeTopDown" << endl;
    hasShader = (node->getShader()!=NULL);

    for_each(this, node, node->visualModel,     &VisualPickVisitor::fwdVisualModel);
    this->VisualVisitor::processNodeTopDown(node);

#ifdef SOFA_SUPPORT_MOVING_FRAMES
    glPopMatrix();
#endif
    return RESULT_CONTINUE;
}

void VisualPickVisitor::processNodeBottomUp(simulation::Node* node)
{
//    cerr <<"VisualPickVisitor::processNodeBottomUp" << endl;
    for_each(this, node, node->visualModel,     &VisualPickVisitor::bwdVisualModel);
}

void VisualPickVisitor::processObject(simulation::Node* /*node*/, core::objectmodel::BaseObject* o)
{
#ifdef DEBUG_DRAW
        std::cerr << ">" << o->getClassName() << "::draw() of " << o->getName() << std::endl;
#endif
//        cout<<"VisualPickVisitor::processObject push name of "<< o->getName() << " = " << pickedId << endl;
        names.push_back(o->getName());
        glPushName(pickedId++);
        o->draw(vparams);
        glPopName();
//        cout<<"VisualPickVisitor::processObject end " << endl;

#ifdef DEBUG_DRAW
        std::cerr << "<" << o->getClassName() << "::draw() of " << o->getName() << std::endl;
#endif
}

void VisualPickVisitor::fwdVisualModel(simulation::Node* /*node*/, core::visual::VisualModel* vm)
{
#ifdef DEBUG_DRAW
    std::cerr << ">" << vm->getClassName() << "::fwdDraw() of " << vm->getName() << std::endl;
#endif
    vm->fwdDraw(vparams);
#ifdef DEBUG_DRAW
    std::cerr << "<" << vm->getClassName() << "::fwdDraw() of " << vm->getName() << std::endl;
#endif
}

void VisualPickVisitor::bwdVisualModel(simulation::Node* /*node*/,core::visual::VisualModel* vm)
{
#ifdef DEBUG_DRAW
    std::cerr << ">" << vm->getClassName() << "::bwdDraw() of " << vm->getName() << std::endl;
#endif
    vm->bwdDraw(vparams);
#ifdef DEBUG_DRAW
    std::cerr << "<" << vm->getClassName() << "::bwdDraw() of " << vm->getName() << std::endl;
#endif
}

void VisualPickVisitor::processVisualModel(simulation::Node* node, core::visual::VisualModel* vm)
{
    //cerr<<"VisualPickVisitor::processVisualModel "<<vm->getName()<<endl;
    sofa::core::visual::Shader* shader = NULL;
    if (hasShader)
        shader = dynamic_cast<sofa::core::visual::Shader*>(node->getShader(subsetsToManage));


    switch(vparams->pass())
    {
    case core::visual::VisualParams::Std:
    {
        if (shader && shader->isActive())
            shader->start();
#ifdef DEBUG_DRAW
        std::cerr << ">" << vm->getClassName() << "::drawVisual() of " << vm->getName() << std::endl;
#endif
        names.push_back(vm->getName());
        glPushName(pickedId++);
        vm->drawVisual(vparams);
        glPopName();
#ifdef DEBUG_DRAW
        std::cerr << "<" << vm->getClassName() << "::drawVisual() of " << vm->getName() << std::endl;
#endif
        if (shader && shader->isActive())
            shader->stop();
        break;
    }
    case core::visual::VisualParams::Transparent:
    {
        if (shader && shader->isActive())
            shader->start();
#ifdef DEBUG_DRAW
        std::cerr << ">" << vm->getClassName() << "::drawTransparent() of " << vm->getName() << std::endl;
#endif
        vm->drawTransparent(vparams);
#ifdef DEBUG_DRAW
        std::cerr << "<" << vm->getClassName() << "::drawTransparent() of " << vm->getName() << std::endl;
#endif
        if (shader && shader->isActive())
            shader->stop();
        break;
    }
    case core::visual::VisualParams::Shadow:
#ifdef DEBUG_DRAW
        std::cerr << ">" << vm->getClassName() << "::drawShadow() of " << vm->getName() << std::endl;
#endif
        vm->drawShadow(vparams);
#ifdef DEBUG_DRAW
        std::cerr << "<" << vm->getClassName() << "::drawShadow() of " << vm->getName() << std::endl;
#endif
        break;
    }

}


}}


