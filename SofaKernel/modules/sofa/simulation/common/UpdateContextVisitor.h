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
#ifndef SOFA_SIMULATION_UPDATECONTEXTACTION_H
#define SOFA_SIMULATION_UPDATECONTEXTACTION_H

#include <sofa/simulation/common/Visitor.h>

namespace sofa
{

namespace core
{
namespace visual
{
class VisualParams; 
} // namespace visual
} // namespace core

namespace simulation
{

class SOFA_SIMULATION_COMMON_API UpdateContextVisitor : public Visitor
{
public:
    UpdateContextVisitor(const core::ExecParams* params)
        : Visitor(params), startingNode(NULL)
    {
    }

    virtual Result processNodeTopDown(simulation::Node* node);

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "context"; }
    virtual const char* getClassName() const { return "UpdateContextVisitor"; }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const { return true; }
protected:
    Node* startingNode;
};

class SOFA_SIMULATION_COMMON_API UpdateSimulationContextVisitor : public UpdateContextVisitor
{
public:
    UpdateSimulationContextVisitor(const core::ExecParams* params)
        : UpdateContextVisitor(params)
    {
    }

    virtual Result processNodeTopDown(simulation::Node* node);
    virtual const char* getClassName() const { return "UpdateSimulationContextVisitor"; }
};

class SOFA_SIMULATION_COMMON_API UpdateVisualContextVisitor : public UpdateContextVisitor
{
public:
    UpdateVisualContextVisitor(const sofa::core::visual::VisualParams* vparams);

    virtual Result processNodeTopDown(simulation::Node* node);
    virtual const char* getClassName() const { return "UpdateVisualContextVisitor"; }
protected:
};

} // namespace simulation

} // namespace sofa

#endif
