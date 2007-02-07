#ifndef SOFA_SIMULATION_TREE_LOCALSTORAGE_H
#define SOFA_SIMULATION_TREE_LOCALSTORAGE_H

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

class GNode;
class Action;

/// Abstract class allowing actions to store local data as a stack while traversing the graph.
class LocalStorage
{
protected:
    virtual ~LocalStorage() {}

public:
    virtual void push(void* data) = 0;
    virtual void* pop() = 0;
    virtual void* top() const = 0;
    virtual bool empty() const = 0;
};

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
