#ifndef SOFA_COMPONENTS_XML_BASENODE_H
#define SOFA_COMPONENTS_XML_BASENODE_H

#include "../Common/Factory.h"
#include "Sofa/Abstract/Base.h"
#include "Sofa/Abstract/BaseContext.h"
#include <string>
#include <list>
#include <map>

namespace Sofa
{

namespace Components
{

namespace XML
{

using namespace Common;

template<class Node>
void create(Node*& obj, std::pair<std::string,std::string> arg)
{
    obj = new Node(arg.first,arg.second);
}

class BaseNode
{
private:
    std::string name;
    std::string type;
    BaseNode* parent;
    typedef std::list<BaseNode*> ChildList;
    ChildList children;
    std::map<std::string,std::string*> attributes;
public:
    BaseNode(const std::string& name, const std::string& type, BaseNode* newParent=NULL);

    virtual ~BaseNode();

    /// Get the node class (Scene, Mapping, ...)
    virtual const char* getClass() const = 0;

    /// Get the associated object
    virtual Abstract::Base* getBaseObject() = 0;

    /// Get the node instance name
    const std::string& getName() const
    { return name; }

    std::string getFullName() const
    {
        if (parent==NULL) return "/";
        std::string pname = parent->getFullName();
        pname += "/";
        pname += getName();
        return pname;
    }

    virtual void setName(const std::string& newName)
    { name = newName; }

    /// Get the node instance type (MassObject, IdentityMapping, ...)
    const std::string& getType() const
    { return type; }

    virtual void setType(const std::string& newType)
    { type = newType; }

    /// Get the parent node
    BaseNode* getParent() const
    { return parent; }

    /// Get all attribute data, read-only
    const std::map<std::string,std::string*>& getAttributeMap() const;

    /// Get all attribute data
    std::map<std::string,std::string*>& getAttributeMap();

    /// Get an attribute given its name (return defaultVal if not present)
    const char* getAttribute(const std::string& attr, const char* defaultVal=NULL);

    /// Set an attribute. Override any existing value
    virtual void setAttribute(const std::string& attr, const char* val);

    /// Remove an attribute. Fails if this attribute is "name" or "type"
    virtual bool removeAttribute(const std::string& attr);

    /// Find a node given its name
    virtual BaseNode* findNode(const char* nodeName, bool absolute=false);

    /// Find an object given its name
    virtual Abstract::Base* findObject(const char* nodeName)
    {
        BaseNode* node = findNode(nodeName);
        if (node!=NULL)
        {
            //std::cout << "Found node "<<nodeName<<": "<<node->getName()<<std::endl;
            return node->getBaseObject();
        }
        else return NULL;
    }

    /// Get all objects of a given type
    template<class Sequence>
    void pushObjects(Sequence& result)
    {
        typename Sequence::value_type obj = dynamic_cast<typename Sequence::value_type>(getBaseObject());
        if (obj!=NULL) result.push_back(obj);

        for (child_iterator<> it = begin(); it != end(); ++it)
            it->pushObjects<Sequence>(result);
    }

    /// Get all objects of a given type
    template<class Map>
    void pushNamedObjects(Map& result)
    {
        typedef typename Map::value_type V;
        typedef typename V::second_type OPtr;
        OPtr obj = dynamic_cast<OPtr>(getBaseObject());
        if (obj!=NULL) result.insert(std::make_pair(getFullName(),obj));

        for (child_iterator<> it = begin(); it != end(); ++it)
            it->pushNamedObjects<Map>(result);
    }

protected:
    /// Change this node's parent. Note that this method is protected as it should be called by the parent's addChild/removeChild methods
    virtual bool setParent(BaseNode* newParent)
    { parent = newParent; return true; }

public:
    virtual bool addChild(BaseNode* child);

    virtual bool removeChild(BaseNode* child);

    virtual bool initNode() = 0;

    virtual bool init();

    template<class Node=BaseNode>
    class child_iterator
    {
    protected:
        BaseNode* parent;
        ChildList::iterator it;
        Node* current;
        child_iterator(BaseNode* parent, ChildList::iterator it)
            : parent(parent), it(it), current(NULL)
        {
            checkIt();
        }
        void checkIt()
        {
            while (it != parent->children.end())
            {
                current=dynamic_cast<Node*>(*it);
                if (current!=NULL) return;
                ++it;
            }
            current = NULL;
        }
    public:
        operator Node*() { return current; }
        Node* operator->() { return current; }
        void operator ++() { ++it; checkIt(); }
        bool operator==(const child_iterator<Node>& i) const
        {
            return it == i.it;
        }
        friend class BaseNode;
    };

    template<class Node>
    child_iterator<Node> begin()
    {
        return child_iterator<Node>(this, children.begin());
    }

    child_iterator<BaseNode> begin()
    {
        return begin<BaseNode>();
    }

    template<class Node>
    child_iterator<Node> end()
    {
        return child_iterator<Node>(this, children.end());
    }

    child_iterator<BaseNode> end()
    {
        return end<BaseNode>();
    }

    typedef Factory< std::string, BaseNode, std::pair<std::string, std::string> > NodeFactory;

    static BaseNode* Create(const std::string& nodeClass, const std::string& name, const std::string& type);

};

} // namespace XML

} // namespace Components

} // namespace Sofa

#endif
