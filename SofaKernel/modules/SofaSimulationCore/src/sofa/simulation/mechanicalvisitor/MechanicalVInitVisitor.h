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

/**
 * Initialize unset MState destVecId vectors with srcVecId vectors value.
 *
 */
template< sofa::core::VecType vtype >
class SOFA_SIMULATION_CORE_API MechanicalVInitVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TMultiVecId<vtype,sofa::core::V_WRITE> DestMultiVecId;
    typedef sofa::core::TMultiVecId<vtype,sofa::core::V_READ> SrcMultiVecId;

    DestMultiVecId vDest;
    SrcMultiVecId vSrc;
    bool m_propagate;

    /// Default constructor
    /// \param _vDest output vector
    /// \param _vSrc input vector
    /// \param propagate sets to true propagates vector initialization to mapped mechanical states
    MechanicalVInitVisitor(const sofa::core::ExecParams* params, DestMultiVecId _vDest, SrcMultiVecId _vSrc = SrcMultiVecId::null(), bool propagate=false)
            : BaseMechanicalVisitor(params)
            , vDest(_vDest)
            , vSrc(_vSrc)
            , m_propagate(propagate)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    bool stopAtMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* /*map*/) override
    {
        return false;
    }

    Result fwdMechanicalState(simulation::Node* node,sofa::core::behavior::BaseMechanicalState* mm) override;

    Result fwdMappedMechanicalState(simulation::Node* node,sofa::core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override
    {
        return "MechanicalVInitVisitor";
    }

    std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return false;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addReadVector(vSrc);
        addWriteVector(vDest);
    }
#endif
};

#if !defined(SOFA_SIMULATION_MECHANICALVISITOR_MECHANICALVINITVISITOR_CPP)
extern template class MechanicalVInitVisitor<sofa::core::V_COORD>;
extern template class MechanicalVInitVisitor<sofa::core::V_DERIV>;
#endif
}