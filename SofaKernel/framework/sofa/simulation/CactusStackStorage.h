/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_TREE_CACTUSSTACKSTORAGE_H
#define SOFA_SIMULATION_TREE_CACTUSSTACKSTORAGE_H

#include <sofa/simulation/simulationcore.h>
#include <sofa/simulation/LocalStorage.h>
#include <stack>


namespace sofa
{

namespace simulation
{


/// Cactus Stack implementation of LocalStorage.
/// See http://www.nist.gov/dads/HTML/cactusstack.html
class SOFA_SIMULATION_CORE_API CactusStackStorage : public simulation::LocalStorage
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
    virtual const char* getClassName() const { return "CactusStackStorage"; }
};

} // namespace simulation

} // namespace sofa

#endif
