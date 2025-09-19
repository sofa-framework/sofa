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
#ifndef SOFA_SIMULATION_UPDATEINTERNALDATAVISITOR_H
#define SOFA_SIMULATION_UPDATEINTERNALDATAVISITOR_H


#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/fwd.h>


namespace sofa::simulation
{

/** Triggers the updateInternal() function to update method called
 * when variables (used to compute other internal variables) are modified
 */
class SOFA_SIMULATION_CORE_API UpdateInternalDataVisitor : public Visitor
{

public:
    UpdateInternalDataVisitor(const core::ExecParams* eparams): Visitor(eparams) {}

    void processUpdateInternalData(simulation::Node* node, sofa::core::objectmodel::BaseObject* baseObj);
    Result processNodeTopDown(simulation::Node* node) override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "internal data update"; }
    const char* getClassName() const override { return "UpdateInternalDataVisitor"; }
};

} // namespace sofa::simulation


#endif
