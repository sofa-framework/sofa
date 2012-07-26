#include "DAGSubGraphNode.h"
#include "DAGNode.h"

#include <sofa/simulation/common/Visitor.h>

namespace sofa
{
namespace simulation
{
namespace graph
{


DAGSubGraphNode::DAGSubGraphNode(DAGNode *node) :
    _node(node),
    visitedStatus(NOT_VISITED)
{

}

DAGSubGraphNode::~DAGSubGraphNode()
{
//    std::cout << "DAGSubGraphNode destructor begin " << _node->getName() << std::endl;

    // first, unlink from parents
    while (!parents.empty())
    {
        DAGSubGraphNode* parent = parents.front();
        parents.pop_front();
        parent->children.remove(this);
    }

    // destructor propagates downward....
    while (!children.empty())
        delete children.front();

//    std::cout << "DAGSubGraphNode destructor DONE " << _node->getName() << std::endl;

}

/// a subgraph has only ONE root
DAGSubGraphNode* DAGSubGraphNode::getRoot()
{
    if (parents.empty()) return this;

    DAGSubGraphNode *root = NULL;
    for (Nodes::iterator it=parents.begin(); it!=parents.end() && !root; it++)
        root = (*it)->getRoot();

    // TODO check if root is NULL (impossible in theory)
    return root;
}

/// is this node in the sub-graph?
DAGSubGraphNode* DAGSubGraphNode::findNode(DAGNode *node,Direction direction)
{
    if (_node == node) return this;

    DAGSubGraphNode *found = NULL;
    if (direction==downward)
        for (Nodes::iterator it=children.begin(); it!=children.end() && !found; it++)
            found = (*it)->findNode(node,downward);
    else if (direction==upward)
        for (Nodes::iterator it=parents.begin(); it!=parents.end() && !found; it++)
            found = (*it)->findNode(node,upward);

    return found;
}

/// adds a child
void DAGSubGraphNode::addChild(DAGSubGraphNode* node)
{
    children.push_back(node);
    node->parents.push_back(this);
}

/// Execute a recursive action starting from this node
void DAGSubGraphNode::executeVisitorTopDown(simulation::Visitor* action,Nodes* executedNodes)
{
//    std::cout << "DAGSubGraphNode::executeVisitor " << _node->getName() << std::endl;

    if (visitedStatus!=NOT_VISITED)
    {
//        std::cout << "...skipped (already visited)" << std::endl;
        return; // skipped (already visited)
    }

    // pour chaque noeud "prune" on continue à parcourir quand même juste pour marquer le noeud comme parcouru

    // check du "visitedStatus" des parents:
    // un enfant n'est pruné que si tous ses parents le sont
    // on ne passe à un enfant que si tous ses parents ont été visités
    bool allParentsPruned = true;
    for (Nodes::iterator it=parents.begin(); it!=parents.end(); it++)
    {
        if ((*it)->visitedStatus == NOT_VISITED)
        {
//            std::cout << "...skipped (not all parents visited)" << std::endl;
            return; // skipped for now...
        }
        allParentsPruned &= ((*it)->visitedStatus == PRUNED);
    }

    // all parents have been visited, let's go with the visitor
    if (allParentsPruned && !parents.empty())
    {
        // do not execute the visitor on this node
        visitedStatus = PRUNED;
//        std::cout << "...pruned (all parents pruned)" << std::endl;
        // ... but continue the recursion anyway!
        for (Nodes::iterator it=children.begin(); it!=children.end(); it++)
            (*it)->executeVisitorTopDown(action,executedNodes);
    }
    else
    {
        // execute the visitor on this node!
        visitedStatus = VISITED;
        executedNodes->push_back(this);
        if(action->processNodeTopDown(_node) == simulation::Visitor::RESULT_PRUNE)
        {
//            std::cout << "...pruned (on its own)" << std::endl;
            visitedStatus = PRUNED;
        }
//        else std::cout << "...visitor executed" << std::endl;


        // ... and continue the recursion !
        for (Nodes::iterator it=children.begin(); it!=children.end(); it++)
            (*it)->executeVisitorTopDown(action,executedNodes);

    }

}

void DAGSubGraphNode::executeVisitorBottomUp(simulation::Visitor* action)
{
    action->processNodeBottomUp(_node);
}

} // namespace graph

} // namespace simulation

} // namespace sofa
