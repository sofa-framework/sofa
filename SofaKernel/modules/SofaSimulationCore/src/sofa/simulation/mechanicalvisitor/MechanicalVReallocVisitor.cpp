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
#define SOFA_SIMULATION_MECHANICALVISITOR_MECHANICALVREALLOCVISITOR_CPP
#include <sofa/simulation/mechanicalvisitor/MechanicalVReallocVisitor.h>

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>

namespace sofa::simulation::mechanicalvisitor
{


template< sofa::core::VecType vtype>
Visitor::Result MechanicalVReallocVisitor<vtype>::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState *mm)
{
    mm->vRealloc( this->params, this->getId(mm), m_properties);
    return RESULT_CONTINUE;
}

template< sofa::core::VecType vtype>
Visitor::Result MechanicalVReallocVisitor<vtype>::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    if (m_propagate)
    {
        mm->vRealloc(this->params, this->getId(mm), m_properties);
    }

    return RESULT_CONTINUE;
}

template< sofa::core::VecType vtype>
Visitor::Result MechanicalVReallocVisitor<vtype>::fwdInteractionForceField(simulation::Node* /*node*/, core::behavior::BaseInteractionForceField* ff)
{
    if (m_interactionForceField)
    {
        for (auto* mm : ff->getMechanicalStates())
        {
            mm->vRealloc( this->params, this->getId(mm), m_properties);
        }
    }

    return RESULT_CONTINUE;
}

template< sofa::core::VecType vtype>
typename MechanicalVReallocVisitor<vtype>::MyVecId MechanicalVReallocVisitor<vtype>::getId( core::behavior::BaseMechanicalState* mm )
{
    MyVecId vid = v->getId(mm);
    if( vid.isNull() ) // not already allocated
    {
        vid = MyVecId(MyVecId::V_FIRST_DYNAMIC_INDEX);
        mm->vAvail( this->params, vid );
        v->setId( mm, vid );
    }
    return vid;
}

template< sofa::core::VecType vtype>
std::string  MechanicalVReallocVisitor<vtype>::getInfos() const
{
    std::string name = "[" + v->getName() + "]";
    return name;
}


template class SOFA_SIMULATION_CORE_API MechanicalVReallocVisitor<sofa::core::V_COORD>;
template class SOFA_SIMULATION_CORE_API MechanicalVReallocVisitor<sofa::core::V_DERIV>;
}
