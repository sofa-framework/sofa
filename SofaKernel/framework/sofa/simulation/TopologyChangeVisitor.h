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
#ifndef SOFA_SIMULATION_TREE_TOPOLOGYCHANGEACTION_H
#define SOFA_SIMULATION_TREE_TOPOLOGYCHANGEACTION_H

#include <sofa/simulation/Visitor.h>
#include <sofa/core/topology/Topology.h>

namespace sofa
{

namespace simulation
{


class SOFA_SIMULATION_CORE_API TopologyChangeVisitor : public Visitor
{

public:
    TopologyChangeVisitor(const sofa::core::ExecParams* params, core::topology::Topology* source)
        : Visitor(params), /*root(NULL),*/ source(source)
    {}

    virtual ~TopologyChangeVisitor() {}

    virtual void processTopologyChangeNoCheck(simulation::Node* node, core::objectmodel::BaseObject* obj);
    virtual void processTopologyChange(simulation::Node* node, core::objectmodel::BaseObject* obj);

    virtual Result processNodeTopDown(simulation::Node* node);
    virtual void processNodeBottomUp(simulation::Node* node);

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "topologyChange"; }
    virtual const char* getClassName() const { return "TopologyChangeVisitor"; }
    virtual std::string getInfos() const { return "Topology:" + source->getName(); }

protected:
    /// Flag to know the number of iterations of the overloaded method processNodeTopDown
    //simulation::Node* root;

    core::topology::Topology* source;
};

//Only triggers the handleTopologyChange method
class SOFA_SIMULATION_CORE_API HandleTopologyChangeVisitor: public Visitor
{
public:
    HandleTopologyChangeVisitor(const sofa::core::ExecParams* params)
        : Visitor(params)
    {}

    virtual Result processNodeTopDown(simulation::Node* node);

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "topologyChange"; }
    virtual const char* getClassName() const { return "HandleTopologyChangeVisitor"; }
};



} // namespace simulation

} // namespace sofa

#endif
