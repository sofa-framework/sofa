#include "FailNode.h"

namespace sofa {
namespace simulation {


using namespace core::objectmodel;

void FailNode::doExecuteVisitor(simulation::Visitor* action) {
	fail();
}

void FailNode::fail() { 
	throw std::logic_error("not implemented");
}

void* FailNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, 
				const sofa::core::objectmodel::TagSet& tags, 
				SearchDirection dir ) const { 
	fail(); 
}

Node* FailNode::findCommonParent( simulation::Node* node2 ) {fail();}
	
void* FailNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, 
				const std::string& path) const {
	fail();
}

void FailNode::getObjects(const sofa::core::objectmodel::ClassInfo& class_info, 
				GetObjectsCallBack& container, 
				const sofa::core::objectmodel::TagSet& tags, 
				SearchDirection dir ) const {
	fail();
}

Node::SPtr FailNode::createChild(const std::string& nodeName) { fail(); }

FailNode::Parents FailNode::getParents() const { fail(); }
FailNode::Children FailNode::getChildren() const { fail(); }
	
/// Add a child node
void FailNode::addChild(BaseNode::SPtr node){ fail(); };

/// Remove a child node
void FailNode::removeChild(BaseNode::SPtr node){ fail(); };

/// Move a node from another node
void FailNode::moveChild(BaseNode::SPtr node){ fail(); };

/// Add a generic object
bool FailNode::addObject(BaseObject::SPtr obj){ fail(); };

/// Remove a generic object
bool FailNode::removeObject(BaseObject::SPtr obj){ fail(); };

/// Move an object from a node to another node
void FailNode::moveObject(BaseObject::SPtr obj){ fail(); };

/// Test if the given node is a parent of this node.
bool FailNode::hasParent(const BaseNode* node) const{ fail(); };

/// Test if the given node is an ancestor of this node.
/// An ancestor is a parent or (recursively) the parent of an ancestor.
bool FailNode::hasAncestor(const BaseNode* node) const{ fail(); };

/// Remove the current node from the graph: depending on the type of Node, it can have one or several parents.
void FailNode::detachFromGraph(){ fail(); };

/// Get this node context
BaseContext* FailNode::getContext(){ fail(); };

/// Get this node context
const BaseContext* FailNode::getContext() const{ fail(); };

/// Return the full path name of this node
std::string FailNode::getPathName() const {fail();}

void* FailNode::findLinkDestClass(const BaseClass* destType, const std::string& path, const BaseLink* link){ fail(); };



}
}
