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

/** Add dt*mass*Gravity to the velocity
This is called if the mass wants to be added separately to the mm from the other forces
*/
class SOFA_SIMULATION_CORE_API MechanicalAddSeparateGravityVisitor : public MechanicalVisitor
{
public:
    sofa::core::MultiVecDerivId res;
    MechanicalAddSeparateGravityVisitor(const sofa::core::MechanicalParams* m_mparams,
                                        sofa::core::MultiVecDerivId resvecid )
            : MechanicalVisitor(m_mparams) , res(resvecid)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    /// Process the BaseMass
    Result fwdMass(simulation::Node* /*node*/,sofa::core::behavior::BaseMass* mass) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAddSeparateGravityVisitor"; }
    virtual std::string getInfos() const override { std::string name= "["+res.getName()+"]"; return name; }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadWriteVector(res);
    }
#endif
};
}
