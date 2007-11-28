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
#include <sofa/simulation/tree/VisualVisitor.h>
#include <sofa/component/visualmodel/VisualModelImpl.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

Visitor::Result VisualDrawVisitor::processNodeTopDown(GNode* node)
{
    glPushMatrix();
    double glMatrix[16];
    node->getPositionInWorld().writeOpenGlMatrix(glMatrix);
    glMultMatrixd( glMatrix );

    //cerr<<"VisualDrawVisitor::processNodeTopDown "<<node->getName()<<endl;
    this->VisualVisitor::processNodeTopDown(node);

    glPopMatrix();
    return RESULT_CONTINUE;
}
void VisualDrawVisitor::processVisualModel(GNode* node, core::VisualModel* vm)
{
    //cerr<<"VisualDrawVisitor::processVisualModel "<<vm->getName()<<endl;
    switch(pass)
    {
    case Std:
        vm->draw();
        break;
    case StdWithShader:
    {
        //we assume that the resulting shader can't be null.
        core::objectmodel::BaseObject* obj = node->getShader();
        sofa::core::Shader* shader = dynamic_cast<sofa::core::Shader*>(obj);

        sofa::component::visualmodel::VisualModelImpl* vmi =  dynamic_cast<sofa::component::visualmodel::VisualModelImpl*> (vm);
        if (vmi)
            shader->start();

        vm->draw();

        if (vmi)
            shader->stop();
        break;
    }
    case Transparent:
        vm->drawTransparent();
        break;
    case Shadow:
        vm->drawShadow();
        break;
    }
}

void VisualUpdateVisitor::processVisualModel(GNode*, core::VisualModel* vm)
{
    vm->update();
}

void VisualInitTexturesVisitor::processVisualModel(GNode*, core::VisualModel* vm)
{
    vm->initTextures();
}

VisualComputeBBoxVisitor::VisualComputeBBoxVisitor()
{
    minBBox[0] = minBBox[1] = minBBox[2] = 1e10;
    maxBBox[0] = maxBBox[1] = maxBBox[2] = -1e10;
}
void VisualComputeBBoxVisitor::processVisualModel(GNode*, core::VisualModel* vm)
{
    vm->addBBox(minBBox, maxBBox);
}

} // namespace tree

} // namespace simulation

} // namespace sofa

