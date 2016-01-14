#include "FailNode.h"

namespace sofa {
namespace simulation {


using namespace core::objectmodel;

void FailNode::doExecuteVisitor(simulation::Visitor* /*action*/, bool /*precomputedOrder*/) {
	fail();
}

void FailNode::fail() { 
	throw std::logic_error("not implemented");
}

void* FailNode::getObject(const sofa::core::objectmodel::ClassInfo& /*class_info*/,
                const sofa::core::objectmodel::TagSet& /*tags*/,
                SearchDirection /*dir*/ ) const {
	fail();
	return 0;
}

Node* FailNode::findCommonParent( simulation::Node* /*node2*/ ) {fail(); return 0; }
	
void* FailNode::getObject(const sofa::core::objectmodel::ClassInfo& /*class_info*/,
                const std::string& /*path*/) const {
	fail();
    return 0;
}

void FailNode::getObjects(const sofa::core::objectmodel::ClassInfo& /*class_info*/,
                GetObjectsCallBack& /*container*/,
                const sofa::core::objectmodel::TagSet& /*tags*/,
                SearchDirection /*dir*/ ) const {
	fail();
}

Node::SPtr FailNode::createChild(const std::string& /*nodeName*/) { fail(); return 0;}

FailNode::Parents FailNode::getParents() const { fail();  FailNode::Parents oNull; return oNull; }
FailNode::Children FailNode::getChildren() const { fail();  FailNode::Parents oNull; return oNull; }

/// returns number of parents
size_t FailNode::getNbParents() const { fail(); return 0;}

/// return the first parent (returns NULL if no parent)
BaseNode* FailNode::getFirstParent() const { fail(); return NULL; }
	
/// Add a child node
void FailNode::addChild(BaseNode::SPtr /*node*/){ fail(); }

/// Remove a child node
void FailNode::removeChild(BaseNode::SPtr /*node*/){ fail(); }

/// Move a node from another node
void FailNode::moveChild(BaseNode::SPtr /*node*/){ fail(); }

/// Add a generic object
bool FailNode::addObject(BaseObject::SPtr /*obj*/){ fail(); return 0; }

/// Remove a generic object
bool FailNode::removeObject(BaseObject::SPtr /*obj*/){ fail(); return 0; }

/// Move an object from a node to another node
void FailNode::moveObject(BaseObject::SPtr /*obj*/){ fail(); }

/// Test if the given node is a parent of this node.
bool FailNode::hasParent(const BaseNode* /*node*/) const{ fail(); return 0; }

/// Test if the given node is an ancestor of this node.
/// An ancestor is a parent or (recursively) the parent of an ancestor.
bool FailNode::hasAncestor(const BaseNode * /*node*/) const{ fail(); return 0; }
bool FailNode::hasAncestor(const BaseContext * /*context*/) const{ fail(); return 0; }

/// Remove the current node from the graph: depending on the type of Node, it can have one or several parents.
void FailNode::detachFromGraph(){ fail(); }

/// Get this node context
BaseContext* FailNode::getContext(){ fail(); return 0; }

/// Get this node context
const BaseContext* FailNode::getContext() const{ fail(); return 0; }

/// Return the full path name of this node
std::string FailNode::getPathName() const {fail(); return 0; }

/// Return the path from this node to the root node
std::string FailNode::getRootPath() const {fail(); return 0; }

void* FailNode::findLinkDestClass(const BaseClass* /*destType*/, const std::string& /*path*/, const BaseLink* /*link*/){ fail(); return 0;}



}
}
