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
#pragma once
#include <sofa/helper/Factory.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/simulation/common/config.h>
#include <string>
#include <list>
#include <map>

namespace sofa::simulation::xml
{

/// Flags indicating when an element is from an included file that should be treated specifically
enum IncludeNodeType
{
    INCLUDE_NODE_CHILD, ///< indicating a standard node that should be used as child
    INCLUDE_NODE_GROUP, ///< indicating a node that should be removed, and its content added within the parent node
    INCLUDE_NODE_MERGE, ///< indicating a node that should be merged with its parent, and any child node with the same name as an existing child should be recursively merged
};

class SOFA_SIMULATION_COMMON_API BaseElement : public sofa::core::objectmodel::BaseObjectDescription
{
private:
    std::string basefile;
    std::string m_srcfile;
    int m_srcline;

    BaseElement* parent;
    typedef std::list<BaseElement*> ChildList;
    ChildList children;
    IncludeNodeType includeNodeType;
protected:
    std::map< std::string, std::string > replaceAttribute;
public:
    BaseElement(const std::string& name, const std::string& type, BaseElement* newParent=nullptr);

    ~BaseElement() override;

    /// Get the node class (Scene, Mapping, ...)
    virtual const char* getClass() const = 0;

    /// Get the associated object
    sofa::core::objectmodel::Base* getObject() override = 0;

    /// Get the node instance name
    std::string getName() override
    { return attributes["name"]; }

    virtual void setName(const std::string& newName) override
    { attributes["name"] = newName; }

    /// Get the node instance type (MassObject, IdentityMapping, ...)
    std::string getType()
    { return attributes["type"]; }

    virtual void setType(const std::string& newType)
    { attributes["type"] = newType; }

    /// Get the parent node
    sofa::core::objectmodel::BaseObjectDescription* getParent() const override
    { return parent; }

    /// Get the parent node
    BaseElement* getParentElement() const
    { return parent; }


    /// Get the file where this description was read from. Useful to resolve relative file paths.
    std::string getBaseFile() override;
    virtual void setBaseFile(const std::string& newBaseFile);

    const std::string& getSrcFile() const ;
    virtual void setSrcFile(const std::string& newSrcFile);

    int getSrcLine() const ;
    virtual void setSrcLine(const int l);

    /// Return true if this element was the root of the file
    bool isFileRoot();

    /// Return if the current element ifsa special group node from an included file
    IncludeNodeType getIncludeNodeType() const { return includeNodeType; }

    /// Specify that the current element is a special group node from an included file
    void setIncludeNodeType(IncludeNodeType t) { includeNodeType=t; }

    ///// Get all attribute data, read-only
    //const std::map<std::string,std::string*>& getAttributeMap() const;

    ///// Get all attribute data
    //std::map<std::string,std::string*>& getAttributeMap();

    ///// Get an attribute given its name (return defaultVal if not present)
    //const char* getAttribute(const std::string& attr, const char* defaultVal=nullptr);


    /// Verify the presence of an attribute
    virtual bool presenceAttribute(const std::string& s);

    /// Remove an attribute. Fails if this attribute is "name" or "type"
    virtual bool removeAttribute(const std::string& attr) override;

    /// List of parameters to be replaced
    virtual void addReplaceAttribute(const std::string &attr, const char* val);
    /// Find a node given its name
    virtual BaseElement* findNode(const char* nodeName, bool absolute=false);

    /// Find a node given its name
    BaseObjectDescription* find(const char* nodeName, bool absolute=false) override
    {
        return findNode(nodeName, absolute);
    }

    /// Get all objects of a given type
    template<class Sequence>
    void pushObjects(Sequence& result)
    {
        typename Sequence::value_type obj = dynamic_cast<typename Sequence::value_type>(getObject());
        if (obj!=nullptr) result.push_back(obj);

        for (child_iterator<> it = begin(); it != end(); ++it)
            it->pushObjects<Sequence>(result);
    }

    /// Get all objects of a given type
    template<class Map>
    void pushNamedObjects(Map& result)
    {
        typedef typename Map::value_type V;
        typedef typename V::second_type OPtr;
        OPtr obj = dynamic_cast<OPtr>(getObject());
        if (obj!=nullptr) result.insert(std::make_pair(getFullName(),obj));

        for (child_iterator<> it = begin(); it != end(); ++it)
            it->pushNamedObjects<Map>(result);
    }

protected:
    /// Change this node's parent. Note that this method is protected as it should be called by the parent's addChild/removeChild methods
    virtual bool setParent(BaseElement* newParent)
    { parent = newParent; return true; }

public:
    virtual bool addChild(BaseElement* child);

    virtual bool removeChild(BaseElement* child);

    virtual bool initNode() = 0;

    virtual bool init();

    template<class Node=BaseElement>
    class child_iterator
    {
    protected:
        BaseElement* parent;
        ChildList::iterator it;
        Node* current;
        child_iterator(BaseElement* parent, ChildList::iterator it)
            : parent(parent), it(it), current(nullptr)
        {
            checkIt();
        }
        void checkIt()
        {
            while (it != parent->children.end())
            {
                current=dynamic_cast<Node*>(*it);
                if (current!=nullptr) return;
                ++it;
            }
            current = nullptr;
        }
    public:
        operator Node*() { return current; }
        Node* operator->() { return current; }
        void operator ++() { ++it; checkIt(); }
        bool operator==(const child_iterator<Node>& i) const
        {
            return it == i.it;
        }
        friend class BaseElement;
    };

    template<class Node>
    child_iterator<Node> begin()
    {
        return child_iterator<Node>(this, children.begin());
    }

    child_iterator<BaseElement> begin()
    {
        return begin<BaseElement>();
    }

    template<class Node>
    child_iterator<Node> end()
    {
        return child_iterator<Node>(this, children.end());
    }

    child_iterator<BaseElement> end()
    {
        return end<BaseElement>();
    }

    typedef helper::Factory< std::string, BaseElement, std::pair<std::string, std::string> > NodeFactory;

    static BaseElement* Create(const std::string& nodeClass, const std::string& name, const std::string& type);

    template<class Node>
    static Node* create(Node*, std::pair<std::string,std::string> arg)
    {
        return new Node(arg.first,arg.second);
    }

};

} // namespace sofa::simulation::xml

namespace sofa::helper
{
#if !defined(SOFA_SIMULATION_COMMON_XML_BASEELEMENT_CPP)
    extern template class SOFA_SIMULATION_COMMON_API Factory< std::string, sofa::simulation::xml::BaseElement, std::pair<std::string, std::string> >;
#endif
} // namespace sofa::helper
