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
#ifndef SOFA_SIMULATION_TREE_CACTUSSTACKSTORAGE_H
#define SOFA_SIMULATION_TREE_CACTUSSTACKSTORAGE_H

#include <sofa/simulation/config.h>
#include <sofa/simulation/LocalStorage.h>
#include <stack>

SOFA_HEADER_DEPRECATED_LOCALSTORAGE()


namespace sofa
{

namespace simulation
{


/// Cactus Stack implementation of LocalStorage.
/// See http://www.nist.gov/dads/HTML/cactusstack.html
SOFA_ATTRIBUTE_DEPRECATED_LOCALSTORAGE()
class SOFA_SIMULATION_CORE_API CactusStackStorage : public simulation::LocalStorage
{
protected:
    CactusStackStorage* up; ///< This point to the parent stack
    CactusStackStorage* down; ///< This point to the *array* of child stacks
    std::stack<void*> stack;
public:
    CactusStackStorage()
        : up(nullptr), down(nullptr)
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

    void push(void* data) override;
    void* pop() override;
    void* top() const override;
    bool empty() const override;
    virtual const char* getClassName() const { return "CactusStackStorage"; }
};

} // namespace simulation

} // namespace sofa

#endif
