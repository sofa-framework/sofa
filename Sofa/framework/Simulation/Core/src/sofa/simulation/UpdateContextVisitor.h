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
#ifndef SOFA_SIMULATION_UPDATECONTEXTACTION_H
#define SOFA_SIMULATION_UPDATECONTEXTACTION_H

#include <sofa/simulation/Visitor.h>

namespace sofa
{


namespace core::visual
{
class VisualParams;
} // namespace core::visual


namespace simulation
{

class SOFA_SIMULATION_CORE_API UpdateContextVisitor : public Visitor
{
public:
    UpdateContextVisitor(const core::ExecParams* eparams)
        : Visitor(eparams), startingNode(nullptr)
    {
    }

    Result processNodeTopDown(simulation::Node* node) override;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "context"; }
    const char* getClassName() const override { return "UpdateContextVisitor"; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return true; }
protected:
    Node* startingNode;
};

class SOFA_SIMULATION_CORE_API UpdateSimulationContextVisitor : public UpdateContextVisitor
{
public:
    UpdateSimulationContextVisitor(const core::ExecParams* eparams)
        : UpdateContextVisitor(eparams)
    {
    }

    Result processNodeTopDown(simulation::Node* node) override;
    const char* getClassName() const override { return "UpdateSimulationContextVisitor"; }
};

class SOFA_SIMULATION_CORE_API UpdateVisualContextVisitor : public UpdateContextVisitor
{
public:
    UpdateVisualContextVisitor(const sofa::core::visual::VisualParams* visuparams);

    Result processNodeTopDown(simulation::Node* node) override;
    const char* getClassName() const override { return "UpdateVisualContextVisitor"; }
protected:
};

} // namespace simulation

} // namespace sofa

#endif
