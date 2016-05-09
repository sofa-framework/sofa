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

#ifndef SOFA_SIMULATION_TREE_CHANGELISTENER_H
#define SOFA_SIMULATION_TREE_CHANGELISTENER_H
#include <set>





#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/MutationListener.h>



namespace sofa
{

namespace simulation
{
namespace common
{


class ChangeListener: public MutationListener
{
public:
    std::set<core::objectmodel::BaseObject *> added;
    std::set<core::objectmodel::BaseObject*> removed;
    ChangeListener()
    {
    }


    /*****************************************************************************************************************/
    void addObject(Node* parent, core::objectmodel::BaseObject* object);
    void removeObject(Node* /*parent*/, core::objectmodel::BaseObject* object);
    bool changed();
    void reset();
};
}
}
}
#endif
