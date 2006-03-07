#include "BaseNode.h"
#include "../Common/Factory.inl"

namespace Sofa
{

namespace Components
{

namespace XML
{

using namespace Common;

BaseNode::BaseNode(const std::string& name, const std::string& type, BaseNode* newParent)
    : name(name), type(type), parent(NULL)
{
    if (newParent!=NULL) newParent->addChild(this);
    attributes["name"]=&this->name;
    attributes["type"]=&this->type;
}

BaseNode::~BaseNode()
{
    attributes.erase("name");
    attributes.erase("type");
    for (std::map<std::string,std::string*>::iterator it = attributes.begin();
            it != attributes.end(); ++it)
    {
        delete it->second;
    }
    attributes.clear();
    for (ChildList::iterator it = children.begin();
            it != children.end(); ++it)
    {
        delete *it;
    }
    children.clear();
}

/// Get an attribute given its name (return defaultVal if not present)
const char* BaseNode::getAttribute(const std::string& attr, const char* defaultVal)
{
    std::map<std::string,std::string*>::iterator it = attributes.find(attr);
    if (it == attributes.end())
        return defaultVal;
    else
        return it->second->c_str();
}

/// Set an attribute. Override any existing value
void BaseNode::setAttribute(const std::string& attr, const char* val)
{
    std::map<std::string,std::string*>::iterator it = attributes.find(attr);
    if (it != attributes.end())
        *(it->second) = val;
    else
        attributes[attr] = new std::string(val);
}

/// Remove an attribute. Fails if this attribute is "name" or "type"
bool BaseNode::removeAttribute(const std::string& attr)
{
    std::map<std::string,std::string*>::iterator it = attributes.find(attr);
    if (it == attributes.end())
        return false;
    if (it->second == &name)
        return false;
    if (it->second == &type)
        return false;
    delete it->second;
    attributes.erase(it);
    return true;
}

bool BaseNode::addChild(BaseNode* child)
{
    if (child->getParent()==this) return false;
    BaseNode* oldParent =  child->getParent();
    if (!child->setParent(this)) return false;
    if (oldParent != NULL)
    {
        oldParent->removeChild(child);
    }
    children.push_back(child);
    return true;
}

bool BaseNode::removeChild(BaseNode* child)
{
    if (child->getParent()!=this) return false;
    ChildList::iterator it = children.begin();
    while (it!=children.end())
    {
        if (*it == child)
        {
            child->setParent(NULL);
            children.erase(it);
            return true;
        }
    }
    return false;
}

bool BaseNode::init()
{
    bool res = initNode();
    for (child_iterator<> it = begin();
            it != end(); ++it)
    {
        res &= it->init();
    }
    return res;
}

// commented by Sylvere F.
// template<> class Factory< std::string, BaseNode, std::pair<std::string, std::string> > NodeFactory;

BaseNode* BaseNode::Create(const std::string& nodeClass, const std::string& name, const std::string& type)
{
    return NodeFactory::CreateObject(nodeClass, std::pair<std::string,std::string>(name, type));
}

/// Find a node given its name
BaseNode* BaseNode::findNode(const char* nodeName, bool absolute)
{
    if (nodeName == NULL) return NULL;
    if (nodeName[0]=='\\' || nodeName[0]=='/')
    {
        if (!absolute && getParent()!=NULL)
            return getParent()->findNode(nodeName);
        else
        { ++nodeName; absolute = true; }
    }
    if (nodeName[0]=='\0')
    {
        if (absolute) return this;
        else return NULL;
    }
    const char* sep = nodeName;
    while (*sep!='\0' && *sep!='\\' && *sep!='/')
        ++sep;
    if (!strncmp(nodeName,".",sep-nodeName))
        return findNode(sep, true);
    if (!strncmp(nodeName,"..",sep-nodeName))
    {
        if (getParent()==NULL) return NULL;
        else return getParent()->findNode(sep,true);
    }
    for (child_iterator<> it = begin(); it != end(); ++it)
    {
        if (!strncmp(it->getName().c_str(), nodeName, sep-nodeName))
        {
            BaseNode* res = it->findNode(sep,true);
            if (res!=NULL) return res;
        }
    }
    if (!absolute && getParent()!=NULL)
        return getParent()->findNode(nodeName);
    else
        return NULL;
}

} // namespace XML

} // namespace Components

} // namespace Sofa
