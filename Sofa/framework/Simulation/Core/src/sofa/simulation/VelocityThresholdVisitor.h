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
#ifndef SOFA_SIMULATION_VelocityThresholdVisitor_H
#define SOFA_SIMULATION_VelocityThresholdVisitor_H

#include <sofa/simulation/Visitor.h>
#include <sofa/core/MultiVecId.h>


namespace sofa::simulation
{

class SOFA_SIMULATION_CORE_API VelocityThresholdVisitor : public Visitor
{
public:
    Visitor::Result processNodeTopDown(simulation::Node* node) override;

    VelocityThresholdVisitor(const core::ExecParams* params, core::MultiVecId v, SReal threshold);



    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override
    {
        return "threshold";
    }
    const char* getClassName() const override { return "VelocityThresholdVisitor"; }

protected:
    core::MultiVecId vid; ///< Id of the vector to process
    SReal threshold; ///< All the entries below this threshold will be set to 0.
};

} // namespace sofa::simulation


#endif
