#ifndef SOFA_COMPONENTS_GRAPH_LOCALSTORAGE_H
#define SOFA_COMPONENTS_GRAPH_LOCALSTORAGE_H

#include "Sofa/Abstract/BaseObject.h"

namespace Sofa
{

namespace Components
{

namespace Graph
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

} // namespace Graph

} // namespace Components

} // namespace Sofa

#endif
