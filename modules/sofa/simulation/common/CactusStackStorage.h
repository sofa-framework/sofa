#ifndef SOFA_SIMULATION_TREE_CACTUSSTACKSTORAGE_H
#define SOFA_SIMULATION_TREE_CACTUSSTACKSTORAGE_H

#include <sofa/simulation/common/LocalStorage.h>
#include <stack>


namespace sofa
{

namespace simulation
{


/// Cactus Stack implementation of LocalStorage.
/// See http://www.nist.gov/dads/HTML/cactusstack.html
class CactusStackStorage : public simulation::LocalStorage
{
protected:
    CactusStackStorage* up; ///< This point to the parent stack
    CactusStackStorage* down; ///< This point to the *array* of child stacks
    std::stack<void*> stack;
public:
    CactusStackStorage()
        : up(NULL), down(NULL)
    {
    }
    void setParent(CactusStackStorage* parent)
    {
        up = parent;
    }
    void setChildren(CactusStackStorage* children)
    {
        down = children;
    }
    CactusStackStorage* getParent()
    {
        return up;
    }
    CactusStackStorage* getChildren()
    {
        return down;
    }

    void push(void* data);
    void* pop();
    void* top() const;
    bool empty() const;
};

} // namespace simulation

} // namespace sofa

#endif
