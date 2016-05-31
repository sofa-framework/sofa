/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SIMULATION_TREE_LOCALSTORAGE_H
#define SOFA_SIMULATION_TREE_LOCALSTORAGE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include "ClassSystem.h"

namespace sofa
{

namespace simulation
{


class Visitor;

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

} // namespace simulation

} // namespace sofa

#endif
