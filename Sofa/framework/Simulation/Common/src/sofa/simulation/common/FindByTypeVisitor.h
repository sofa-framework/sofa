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
#pragma once
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Visitor.h>
#include <iostream>
#include <typeinfo>
#include <sofa/helper/Factory.h>

namespace sofa::simulation
{



/** Find all components of a given type and store pointers in a list.
*/
template<class T>
class FindByTypeVisitor : public Visitor
{
public:
    std::vector<T*> found; ///< The result of the search: contains pointers to all components of the given type found.

    FindByTypeVisitor(const core::ExecParams* params) : Visitor(params) {}

    /// For each component, if it is of the given type, then put it in the list
    Result processNodeTopDown(simulation::Node* node) override
    {
        for( simulation::Node::ObjectIterator i=node->object.begin(), iend=node->object.end(); i!=iend; i++ )
        {
            if( T* obj= dynamic_cast<T*>(i->get()) )
                found.push_back(obj);
        }
        return RESULT_CONTINUE;
    }
    const char* getClassName() const override { return "FindByTypeVisitor"; }
    virtual std::string getInfos() const override { std::string name="["+sofa::helper::gettypename(typeid(T))+"]"; return name; }

};


} // namespace sofa::simulation
