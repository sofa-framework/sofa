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

/**
* @brief Visitor class used to set positions and velocities of the top level MechanicalStates of the hierarchy.
*/
class SOFA_SIMULATION_CORE_API MechanicalSetPositionAndVelocityVisitor : public MechanicalVisitor
{
public:
    SReal t;
    sofa::core::MultiVecCoordId x;
    sofa::core::MultiVecDerivId v;

    MechanicalSetPositionAndVelocityVisitor(const sofa::core::MechanicalParams* mparams ,SReal time=0,
                                            sofa::core::MultiVecCoordId x = sofa::core::VecCoordId::position(),
                                            sofa::core::MultiVecDerivId v = sofa::core::VecDerivId::velocity());

    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalSetPositionAndVelocityVisitor";}
    virtual std::string getInfos() const override { std::string name="x["+x.getName()+"] v["+v.getName()+"]"; return name; }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadVector(x);
        addReadVector(v);
    }
#endif
};

}