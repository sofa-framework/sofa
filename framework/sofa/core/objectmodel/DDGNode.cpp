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
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Base.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/// Constructor
DDGNode::DDGNode()
    : inputs(initLink("inputs", "Links to inputs Data"))
    , outputs(initLink("outputs", "Links to outputs Data"))
{
}

DDGNode::~DDGNode()
{
    for(DDGLinkIterator it=inputs.begin(); it!=inputs.end(); ++it)
        (*it)->doDelOutput(this);
    for(DDGLinkIterator it=outputs.begin(); it!=outputs.end(); ++it)
        (*it)->doDelInput(this);
}

template<>
TClass<DDGNode,void>::TClass()
{
    DDGNode* ptr = NULL;
    namespaceName = Base::namespaceName(ptr);
    className = Base::className(ptr);
    templateName = Base::templateName(ptr);
    shortName = Base::shortName(ptr);
}

void DDGNode::setDirtyValue(const core::ExecParams* params)
{
    bool& dirtyValue = dirtyFlags[currentAspect(params)].dirtyValue;
    if (!dirtyValue)
    {
        dirtyValue = true;

        // TRACE LOG (HACK...)
        BaseData* d = dynamic_cast<BaseData*>(this);
        if (d && d->getOwner())
            d->getOwner()->sout << "Data " << d->getName() << " is now dirty." << d->getOwner()->sendl;

        setDirtyOutputs(params);
    }
}

void DDGNode::setDirtyOutputs(const core::ExecParams* params)
{
    bool& dirtyOutputs = dirtyFlags[currentAspect(params)].dirtyOutputs;
    if (!dirtyOutputs)
    {
        dirtyOutputs = true;
        for(DDGLinkIterator it=outputs.begin(params), itend=outputs.end(params); it != itend; ++it)
        {
            (*it)->setDirtyValue(params);
        }
    }
}

void DDGNode::cleanDirty(const core::ExecParams* params)
{
    bool& dirtyValue = dirtyFlags[currentAspect(params)].dirtyValue;
    if (dirtyValue)
    {
        dirtyValue = false;

        BaseData* d = dynamic_cast<BaseData*>(this);
        if (d && d->getOwner())
            d->getOwner()->sout << "Data " << d->getName() << " has been updated." << d->getOwner()->sendl;

        for(DDGLinkIterator it=inputs.begin(params), itend=inputs.end(params); it != itend; ++it)
            (*it)->dirtyFlags[currentAspect(params)].dirtyOutputs = false;
    }
}

void DDGNode::copyAspect(int destAspect, int srcAspect)
{
    dirtyFlags[destAspect] = dirtyFlags[srcAspect];
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

const DDGNode::DDGLinkContainer& DDGNode::getInputs()
{
    return inputs.getValue();
}

const DDGNode::DDGLinkContainer& DDGNode::getOutputs()
{
    return outputs.getValue();
}

sofa::core::objectmodel::Base* LinkTraitsPtrCasts<DDGNode>::getBase(sofa::core::objectmodel::DDGNode* n)
{
    if (!n) return NULL;
    return n->getOwner();
    //sofa::core::objectmodel::BaseData* d = dynamic_cast<sofa::core::objectmodel::BaseData*>(n);
    //if (d) return d->getOwner();
    //return dynamic_cast<sofa::core::objectmodel::Base*>(n);
}

sofa::core::objectmodel::BaseData* LinkTraitsPtrCasts<DDGNode>::getData(sofa::core::objectmodel::DDGNode* n)
{
    if (!n) return NULL;
    return n->getData();
    //return dynamic_cast<sofa::core::objectmodel::BaseData*>(n);
}

bool DDGNode::findDataLinkDest(DDGNode*& ptr, const std::string& path, const BaseLink* link)
{
    return false; // TODO
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
