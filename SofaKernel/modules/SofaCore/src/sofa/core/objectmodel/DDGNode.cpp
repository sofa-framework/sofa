/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/DataEngine.h>

//#define SOFA_DDG_TRACE

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
    DDGNode* ptr = nullptr;
    namespaceName = Base::namespaceName(ptr);
    className = Base::className(ptr);
    templateName = Base::templateName(ptr);
    shortName = Base::shortName(ptr);
}

void DDGNode::setDirtyValue()
{
    bool& dirtyValue = dirtyFlags.dirtyValue;
    if (!dirtyValue)
    {
        dirtyValue = true;
        setDirtyOutputs();
    }
}

void DDGNode::setDirtyOutputs()
{
    bool& dirtyOutputs = dirtyFlags.dirtyOutputs;
    if (!dirtyOutputs)
    {
        dirtyOutputs = true;
        for(DDGLinkIterator it=outputs.begin(), itend=outputs.end(); it != itend; ++it)
        {
            (*it)->setDirtyValue();
        }
    }
}

void DDGNode::cleanDirty()
{
    bool& dirtyValue = dirtyFlags.dirtyValue;
    if (dirtyValue)
    {
        dirtyValue = false;
        cleanDirtyOutputsOfInputs();
    }
}

void DDGNode::notifyEndEdit()
{
    for(DDGLinkIterator it=outputs.begin(), itend=outputs.end(); it != itend; ++it)
        (*it)->notifyEndEdit();
}

void DDGNode::cleanDirtyOutputsOfInputs()
{
    for(DDGLinkIterator it=inputs.begin(), itend=inputs.end(); it != itend; ++it)
        (*it)->dirtyFlags.dirtyOutputs = false;
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
    if (!n) return nullptr;
    return n->getOwner();
    //sofa::core::objectmodel::BaseData* d = dynamic_cast<sofa::core::objectmodel::BaseData*>(n);
    //if (d) return d->getOwner();
    //return dynamic_cast<sofa::core::objectmodel::Base*>(n);
}

sofa::core::objectmodel::BaseData* LinkTraitsPtrCasts<DDGNode>::getData(sofa::core::objectmodel::DDGNode* n)
{
    if (!n) return nullptr;
    return n->getData();
    //return dynamic_cast<sofa::core::objectmodel::BaseData*>(n);
}

bool DDGNode::findDataLinkDest(DDGNode*& ptr, const std::string& path, const BaseLink* link)
{
    std::string pathStr, dataStr(" "); // non-empty data to specify that a data name is optionnal for DDGNode (as it can be a DataEngine)
    if (link)
    {
        if (!link->parseString(path, &pathStr, &dataStr))
            return false;
    }
    else
    {
        if (!BaseLink::ParseString(path, &pathStr, &dataStr, this->getOwner()))
            return false;
    }
    bool self = (pathStr.empty() || pathStr == "[]");
    if (dataStr == "") // no Data -> we look for a DataEngine
    {
        if (self)
        {
            ptr = this;
            return true;
        }
        else
        {
            Base* owner = this->getOwner();
            DataEngine* obj = nullptr;
            if (!owner)
                return false;
            if (!owner->findLinkDest(obj, path, link))
                return false;
            ptr = obj;
            return true;
        }
    }
    Base* owner = this->getOwner();
    if (!owner)
        return false;
    if (self)
    {
        ptr = owner->findData(dataStr);
        return (ptr != nullptr);
    }
    else
    {
        Base* obj = nullptr;
        if (!owner->findLinkDest(obj, BaseLink::CreateString(pathStr), link))
            return false;
        if (!obj)
            return false;
        ptr = obj->findData(dataStr);
        return (ptr != nullptr);
    }
}

void DDGNode::addLink(BaseLink* /*l*/)
{
    // the inputs and outputs links in DDGNode is manually added
    // once the link vectors are constructed in Base or BaseData
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
