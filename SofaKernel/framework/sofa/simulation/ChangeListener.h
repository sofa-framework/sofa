/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_SIMULATION_TREE_CHANGELISTENER_H
#define SOFA_SIMULATION_TREE_CHANGELISTENER_H
#include <set>





#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/MutationListener.h>



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
