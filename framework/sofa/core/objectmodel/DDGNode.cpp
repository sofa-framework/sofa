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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/DDGNode.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/// Constructor
DDGNode::DDGNode()
    : dirtyValue(false)
    , dirtyOutputs(false)
{
}

DDGNode::~DDGNode()
{
    for(std::list< DDGNode* >::iterator it=inputs.begin(); it!=inputs.end(); ++it)
        (*it)->doDelOutput(this);
    for(std::list< DDGNode* >::iterator it=outputs.begin(); it!=outputs.end(); ++it)
        (*it)->doDelInput(this);
}

void DDGNode::setDirtyValue()
{
    if (!dirtyValue)
    {
        dirtyValue = true;
        setDirtyOutputs();
    }
}

void DDGNode::setDirtyOutputs()
{
    if (!dirtyOutputs)
    {
        dirtyOutputs = true;
        for(std::list<DDGNode*>::iterator it=outputs.begin(); it!=outputs.end(); ++it)
        {
            (*it)->setDirtyValue();
        }
    }
}

void DDGNode::cleanDirty()
{
    if (dirtyValue)
    {
        dirtyValue = false;
        for(std::list< DDGNode* >::iterator it=inputs.begin(); it!=inputs.end(); ++it)
            (*it)->dirtyOutputs = false;
    }
}

void DDGNode::addInput(DDGNode* n)
{
    doAddInput(n);
    n->doAddOutput(this);
    setDirtyValue();
}

void DDGNode::delInput(DDGNode* n)
{
    doDelInput(n);
    n->doDelOutput(this);
}

void DDGNode::addOutput(DDGNode* n)
{
    doAddOutput(n);
    n->doAddInput(this);
    n->setDirtyValue();
}

void DDGNode::delOutput(DDGNode* n)
{
    doDelOutput(n);
    n->doDelInput(this);
}

std::list<DDGNode*> DDGNode::getInputs()
{
    return inputs;
}

std::list<DDGNode*> DDGNode::getOutputs()
{
    return outputs;
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
