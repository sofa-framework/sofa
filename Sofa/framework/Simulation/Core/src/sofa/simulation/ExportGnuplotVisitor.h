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
#ifndef SOFA_SIMULATION_TREE_EXPORTGNUPLOTACTION_H
#define SOFA_SIMULATION_TREE_EXPORTGNUPLOTACTION_H

#include <sofa/simulation/Visitor.h>


namespace sofa::simulation
{


class SOFA_SIMULATION_CORE_API InitGnuplotVisitor : public simulation::Visitor
{
public:
    std::string gnuplotDirectory;

    InitGnuplotVisitor(const core::ExecParams* eparams, std::string dir) : Visitor(eparams),gnuplotDirectory(dir) {}

    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    Result processNodeTopDown(simulation::Node* node) override;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override
    {
        return "initGnuplot";
    }
    const char* getClassName() const override { return "InitGnuplotVisitor"; }
};

class SOFA_SIMULATION_CORE_API ExportGnuplotVisitor : public simulation::Visitor
{
public:
    ExportGnuplotVisitor(const core::ExecParams* eparams, SReal time);
    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    Result processNodeTopDown(simulation::Node* node) override;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override
    {
        return "exportGnuplot";
    }
    const char* getClassName() const override { return "ExportGnuplotVisitor"; }
protected:
    SReal m_time;
};

} // namespace sofa::simulation


#endif
