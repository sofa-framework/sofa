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
#define SOFA_SIMULATION_COMMON_XML_BASEELEMENT_CPP
#include <sofa/simulation/common/xml/BaseElement.h>
#include <sofa/helper/Factory.inl>
#include <sofa/helper/system/SetDirectory.h>
#include <cstring>

template class SOFA_SIMULATION_COMMON_API sofa::helper::Factory< std::string, sofa::simulation::xml::BaseElement, std::pair<std::string, std::string> >;

namespace sofa::simulation::xml
{

BaseElement::BaseElement(const std::string& name, const std::string& type, BaseElement* newParent)
    : BaseObjectDescription(name.c_str(), type.c_str()), parent(nullptr), includeNodeType(INCLUDE_NODE_CHILD)
{
    if (newParent!=nullptr) newParent->addChild(this);
}

BaseElement::~BaseElement()
{
    for (ChildList::iterator it = children.begin();
            it != children.end(); ++it)
    {
        delete *it;
    }
    children.clear();
}

/// Get the file where this description was read from. Useful to resolve relative file paths.
std::string BaseElement::getBaseFile()
{
    if (isFileRoot()) return basefile;
    else if (getParentElement()!=nullptr) return getParentElement()->getBaseFile();
    else return "";
}

void BaseElement::setBaseFile(const std::string& newBaseFile)
{
    basefile = newBaseFile;
}

/// Return true if this element was the root of the file
bool BaseElement::isFileRoot()
{
    return !basefile.empty();
}

const std::string& BaseElement::getSrcFile() const {
    return m_srcfile ;
}

void BaseElement::setSrcFile(const std::string& filename) {
    m_srcfile = filename ;
}

int BaseElement::getSrcLine() const {
    return m_srcline ;
}

void BaseElement::setSrcLine(const int l)
{
    m_srcline = l ;
}

bool BaseElement::presenceAttribute(const std::string& s)
{
    return attributes.contains(s);
}
/// Remove an attribute. Fails if this attribute is "name" or "type"
bool BaseElement::removeAttribute(const std::string& attr)
{
    const AttributeMap::iterator it = attributes.find(attr);
    if (it == attributes.end())
        return false;
    attributes.erase(it);
    return true;
}

void BaseElement::addReplaceAttribute(const std::string &attr, const char* val)
{
    replaceAttribute[attr]=val;
}

bool BaseElement::addChild(BaseElement* child)
{
    if (child->getParent()==this) return false;
    BaseElement* oldParent =  child->getParentElement();
    if (!child->setParent(this)) return false;
    if (oldParent != nullptr)
    {
        oldParent->removeChild(child);
    }
    children.push_back(child);
    return true;
}

bool BaseElement::removeChild(BaseElement* child)
{
    if (child->getParent()!=this) return false;
    const ChildList::iterator it = children.begin();
    while (it!=children.end())
    {
        if (*it == child)
        {
            child->setParent(nullptr);
            children.erase(it);
            return true;
        }
    }
    return false;
}

bool BaseElement::init()
{
    sofa::helper::system::SetDirectory cwd(basefile);
    bool res = initNode();
    for (child_iterator<> it = begin(); it != end(); ++it)
    {
        res &= it->init();
    }
    return res;
}

BaseElement* BaseElement::Create(const std::string& nodeClass, const std::string& name, const std::string& type)
{
    return NodeFactory::CreateObject(nodeClass, std::pair<std::string,std::string>(name, type));
}

/// Find a node given its name
BaseElement* BaseElement::findNode(const char* nodeName, bool absolute)
{
    if (nodeName == nullptr) return nullptr;
    if (nodeName[0]=='\\' || nodeName[0]=='/')
    {
        if (!absolute && getParentElement()!=nullptr)
            return getParentElement()->findNode(nodeName);
        else
        { ++nodeName; absolute = true; }
    }
    if (nodeName[0]=='\0')
    {
        if (absolute) return this;
        else return nullptr;
    }
    const char* sep = nodeName;
    while (*sep!='\0' && *sep!='\\' && *sep!='/')
        ++sep;
    if (!strncmp(nodeName,".",sep-nodeName))
        return findNode(sep, true);
    if (!strncmp(nodeName,"..",sep-nodeName))
    {
        if (getParentElement()==nullptr) return nullptr;
        else return getParentElement()->findNode(sep,true);
    }
    for (child_iterator<> it = begin(); it != end(); ++it)
    {
        if (it->getName().length() == (unsigned)(sep-nodeName) && !strncmp(it->getName().c_str(), nodeName, sep-nodeName))
        {
            BaseElement* res = it->findNode(sep,true);
            if (res!=nullptr) return res;
        }
    }
    if (!absolute && getParentElement()!=nullptr)
        return getParentElement()->findNode(nodeName);
    else
        return nullptr;
}

} // namespace sofa::simulation::xml
