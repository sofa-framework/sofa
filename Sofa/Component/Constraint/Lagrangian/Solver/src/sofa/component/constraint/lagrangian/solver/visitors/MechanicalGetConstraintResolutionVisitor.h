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
#include <sofa/component/constraint/lagrangian/solver/config.h>
#include <sofa/simulation/BaseMechanicalVisitor.h>

namespace sofa::component::constraint::lagrangian::solver
{

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API MechanicalGetConstraintResolutionVisitor : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalGetConstraintResolutionVisitor(const core::ConstraintParams* params, std::vector<core::behavior::ConstraintResolution*>& res, unsigned int offset = 0);

    Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override;

    bool isThreadSafe() const override;
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* node, core::BaseMapping* map) override;

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override { }
#endif
private:
    /// Constraint parameters
    const sofa::core::ConstraintParams *cparams;

    std::vector<core::behavior::ConstraintResolution*>& _res;
    unsigned int _offset;
};

}
