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
 * Reserve an auxiliary vector identified by a symbolic constant.
 *
 */
template< sofa::core::VecType vtype >
class SOFA_SIMULATION_CORE_API MechanicalVReallocVisitor : public BaseMechanicalVisitor
{
public:
    typedef sofa::core::TMultiVecId<vtype,sofa::core::V_WRITE> DestMultiVecId;
    typedef sofa::core::TVecId<vtype,sofa::core::V_WRITE> MyVecId;


    DestMultiVecId *v;
    bool m_propagate;
    bool m_interactionForceField;
    const core::VecIdProperties& m_properties;

    /// Default constructor
    /// \param v output vector
    /// \param propagate sets to true propagates vector initialization to mapped mechanical states
    /// \param interactionForceField sets to true also initializes external mechanical states linked by an interaction force field
    MechanicalVReallocVisitor(const sofa::core::ExecParams* params, DestMultiVecId *v, bool interactionForceField=false, bool propagate=false, const core::VecIdProperties& properties = {})
            : BaseMechanicalVisitor(params)
            , v(v)
            , m_propagate(propagate)
            , m_interactionForceField(interactionForceField)
            , m_properties(properties)
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

    Result fwdInteractionForceField(simulation::Node* node,sofa::core::behavior::BaseInteractionForceField* ff) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override
    {
        return "MechanicalVReallocVisitor";
    }

    std::string getInfos() const override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override
    {
        addWriteVector(*v);
    }
#endif
protected:


    MyVecId getId(sofa::core::behavior::BaseMechanicalState* mm );
};

#if !defined(SOFA_SIMULATION_MECHANICALVISITOR_MECHANICALVREALLOCVISITOR_CPP)
extern template class MechanicalVReallocVisitor<sofa::core::V_COORD>;
extern template class MechanicalVReallocVisitor<sofa::core::V_DERIV>;
#endif
}
