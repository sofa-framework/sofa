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
#ifndef SOFA_SIMULATION_TREE_UPDATEMAPPINGACTION_H
#define SOFA_SIMULATION_TREE_UPDATEMAPPINGACTION_H

#include <sofa/simulation/Visitor.h>
#include <sofa/core/BaseMapping.h>

namespace sofa
{

namespace simulation
{

/// propagating position and velocity through non-mechanical mappings
/// (while MechanicalPropagateOnlyPositionAndVelocityVisitor is propagating mechanical mappings)
class SOFA_SIMULATION_CORE_API UpdateMappingVisitor : public Visitor
{
public:
    UpdateMappingVisitor(const sofa::core::ExecParams* params) : Visitor(params) {}
    void processMapping(simulation::Node* node, core::BaseMapping* obj);
    void processMechanicalMapping(simulation::Node*, core::BaseMapping* obj);

    virtual Result processNodeTopDown(simulation::Node* node);

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "mapping"; }
    virtual const char* getClassName() const { return "UpdateMappingVisitor"; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const { return true; }
};

} // namespace simulation

} // namespace sofa

#endif
