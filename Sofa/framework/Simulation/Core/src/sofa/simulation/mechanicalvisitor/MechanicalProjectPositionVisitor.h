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

#include <sofa/simulation/MechanicalVisitor.h>

namespace sofa::simulation::mechanicalvisitor
{

class SOFA_SIMULATION_CORE_API MechanicalProjectPositionVisitor : public MechanicalVisitor
{
public:
    SReal t;
    sofa::core::MultiVecCoordId pos;
    MechanicalProjectPositionVisitor(const sofa::core::MechanicalParams* mechaparams , SReal time=0,
                                     sofa::core::MultiVecCoordId x = sofa::core::vec_id::write_access::position)
            : MechanicalVisitor(mechaparams) , t(time), pos(x)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override;
    Result fwdProjectiveConstraintSet(simulation::Node* /*node*/,sofa::core::behavior::BaseProjectiveConstraintSet* c) override;


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalProjectPositionVisitor"; }
    std::string getInfos() const override;
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(pos);
    }
#endif
};
}
