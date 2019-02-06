#ifndef FAILNODE_H
#define FAILNODE_H

#include <sofa/simulation/Node.h>

namespace sofa {
namespace simulation {

// throws on every method. for testing purpose.
class FailNode : public Node {
public:
	
    void doExecuteVisitor(simulation::Visitor* action, bool precomputedOrder=false);

	static void fail();

	void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, 
					const sofa::core::objectmodel::TagSet& tags, 
					SearchDirection dir = SearchUp) const;

    virtual Node* findCommonParent( simulation::Node* node2 );
	
	virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, 
							const std::string& path) const;

	virtual void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, 
							GetObjectsCallBack& container, 
							const sofa::core::objectmodel::TagSet& tags, 
							SearchDirection dir = SearchUp) const;

	virtual Node::SPtr createChild(const std::string& nodeName);

	virtual Parents getParents() const;
	virtual Children getChildren() const;

    /// returns number of parents
    virtual size_t getNbParents() const;

    /// return the first parent (returns NULL if no parent)
    virtual BaseNode* getFirstParent() const;
	
	  /// Add a child node
    virtual void doAddChild(BaseNode::SPtr node);

    /// Remove a child node
    virtual void doRemoveChild(BaseNode::SPtr node);

    /// Move a node from another node
    virtual void doMoveChild(BaseNode::SPtr node);

    /// Add a generic object
    virtual void doAddObject(core::objectmodel::BaseObject::SPtr obj);

    /// Remove a generic object
    virtual void doRemoveObject(core::objectmodel::BaseObject::SPtr obj);

    /// Move an object from a node to another node
    virtual void doMoveObject(core::objectmodel::BaseObject::SPtr obj);

    /// Test if the given node is a parent of this node.
    virtual bool hasParent(const BaseNode* node) const; 

    /// Test if the given node is an ancestor of this node.
    /// An ancestor is a parent or (recursively) the parent of an ancestor.
    virtual bool hasAncestor(const BaseNode* node) const;
    virtual bool hasAncestor(const BaseContext* context) const;

    /// Remove the current node from the graph: depending on the type of Node, it can have one or several parents.
    virtual void detachFromGraph();

    /// Get this node context
    virtual BaseContext* getContext();

    /// Get this node context
    virtual const BaseContext* getContext() const;

    /// Return the full path name of this node
    virtual std::string getPathName() const;

    /// Return the path from this node to the root node
    virtual std::string getRootPath() const;

    virtual void* findLinkDestClass(const core::objectmodel::BaseClass* destType, 
									const std::string& path, 
									const BaseLink* link);

	

}; 

}
}

#endif

