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

#include <sofa/simulation/BaseMechanicalVisitor.h>

namespace sofa::simulation::mechanicalvisitor
{

/** Find the first available index for a VecId
*/
template <sofa::core::VecType vtype>
class SOFA_SIMULATION_CORE_API MechanicalVAvailVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TVecId<vtype,sofa::core::V_WRITE> MyVecId;
    typedef sofa::core::TMultiVecId<vtype,sofa::core::V_WRITE> MyMultiVecId;
    typedef std::set<sofa::core::BaseState*> StateSet;
    MyVecId& v;
    StateSet states;
    MechanicalVAvailVisitor( const sofa::core::ExecParams* eparams, MyVecId& vecid)
            : BaseMechanicalVisitor(eparams), v(vecid)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    Result fwdMechanicalState(simulation::Node* /*node*/,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalVAvailVisitor"; }
    virtual std::string getInfos() const override;
    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return false;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        MyMultiVecId mv(v);
        addReadWriteVector( mv );
    }
#endif
};

#if !defined(SOFA_SIMULATION_MECHANICALVISITOR_MECHANICALVAVAILVISITOR_CPP)
extern template class MechanicalVAvailVisitor<sofa::core::V_COORD>;
extern template class MechanicalVAvailVisitor<sofa::core::V_DERIV>;
#endif

}
