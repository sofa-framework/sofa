#ifndef FAILNODE_H
#define FAILNODE_H

#include <sofa/simulation/Node.h>

namespace sofa {
namespace simulation {

// throws on every method. for testing purpose.
class FailNode : public Node {
public:
	
    void doExecuteVisitor(simulation::Visitor* action, bool precomputedOrder=false) override;

	static void fail();

	void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, 
					const sofa::core::objectmodel::TagSet& tags, 
					SearchDirection dir = SearchUp) const override;

    Node* findCommonParent( simulation::Node* node2 ) override;
	
	virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, 
							const std::string& path) const;

	void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, 
							GetObjectsCallBack& container, 
							const sofa::core::objectmodel::TagSet& tags, 
							SearchDirection dir = SearchUp) const override;

	virtual Node::SPtr createChild(const std::string& nodeName);

	Parents getParents() const override;
	Children getChildren() const override;

    /// returns number of parents
    size_t getNbParents() const override;

    /// return the first parent (returns NULL if no parent)
    BaseNode* getFirstParent() const override;
	
	  /// Add a child node
    virtual void doAddChild(BaseNode::SPtr node);

    /// Remove a child node
    virtual void doRemoveChild(BaseNode::SPtr node);

    /// Move a node from another node
    virtual void doMoveChild(BaseNode::SPtr node);

    /// Add a generic object
    virtual bool doAddObject(core::objectmodel::BaseObject::SPtr obj);

    /// Remove a generic object
    virtual bool doRemoveObject(core::objectmodel::BaseObject::SPtr obj);

    /// Move an object from a node to another node
    virtual void doMoveObject(core::objectmodel::BaseObject::SPtr obj);

    /// Test if the given node is a parent of this node.
    bool hasParent(const BaseNode* node) const override; 

    /// Test if the given node is an ancestor of this node.
    /// An ancestor is a parent or (recursively) the parent of an ancestor.
    bool hasAncestor(const BaseNode* node) const override;
    bool hasAncestor(const BaseContext* context) const override;

    /// Remove the current node from the graph: depending on the type of Node, it can have one or several parents.
    void detachFromGraph() override;

    /// Get this node context
    BaseContext* getContext() override;

    /// Get this node context
    const BaseContext* getContext() const override;

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

