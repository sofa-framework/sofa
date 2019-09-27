/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_INTERNALUPDATEDATAVISITOR_H
#define SOFA_SIMULATION_INTERNALUPDATEDATAVISITOR_H

#include <sofa/core/ExecParams.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace simulation
{

/** Triggers the internalUpdate() function to update method called
 * when Data (used to compute other internal variables) are modified
 */
class SOFA_SIMULATION_CORE_API InternalUpdateDataVisitor : public Visitor
{

public:
    InternalUpdateDataVisitor(const core::ExecParams* params): Visitor(params) {}

    void processInternalUpdateData(simulation::Node* node, sofa::core::objectmodel::BaseObject* baseObj);
    Result processNodeTopDown(simulation::Node* node) override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "internal update of data"; }
    const char* getClassName() const override { return "InternalUpdateDataVisitor"; }
};

} // namespace simulation

} // namespace sofa

#endif
