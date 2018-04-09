/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/objectmodel/BaseNode.h>


namespace sofa
{

namespace core
{

namespace behavior
{

BaseMechanicalState::BaseMechanicalState()
{
}

BaseMechanicalState::~BaseMechanicalState()
{
}

/// Perform a sequence of linear vector accumulation operation $r_i = sum_j (v_j*f_{ij})$
///
/// This is used to compute in on steps operations such as $v = v + a*dt, x = x + v*dt$.
/// Note that if the result vector appears inside the expression, it must be the first operand.
/// By default this method decompose the computation into multiple vOp calls.
void BaseMechanicalState::vMultiOp(const ExecParams* params, const VMultiOp& ops)
{
    for(VMultiOp::const_iterator it = ops.begin(), itend = ops.end(); it != itend; ++it)
    {
        VecId r = it->first.getId(this);
        const helper::vector< std::pair< ConstMultiVecId, SReal > >& operands = it->second;
        size_t nop = operands.size();
        if (nop==0)
        {
            vOp(params, r);
        }
        else if (nop==1)
        {
            if (operands[0].second == 1.0)
                vOp( params, r, operands[0].first.getId(this));
            else
                vOp( params, r, ConstVecId::null(), operands[0].first.getId(this), operands[0].second);
        }
        else
        {
            size_t i;
            if (operands[0].second == 1.0)
            {
                vOp( params, r, operands[0].first.getId(this), operands[1].first.getId(this), operands[1].second);
                i = 2;
            }
            else
            {
                vOp( params, r, ConstVecId::null(), operands[0].first.getId(this), operands[0].second);
                i = 1;
            }
            for (; i<nop; ++i)
                vOp( params, r, r, operands[i].first.getId(this), operands[i].second);
        }
    }
}

/// Handle state Changes from a given Topology
void BaseMechanicalState::handleStateChange(core::topology::Topology* /*t*/)
{
//    if (t == this->getContext()->getTopology())
//    {
        handleStateChange();
//    }
}

void BaseMechanicalState::writeState( std::ostream& )
{ }

bool BaseMechanicalState::insertInNode( objectmodel::BaseNode* node )
{
    node->addMechanicalState(this);
    Inherit1::insertInNode(node);
    return true;
}

bool BaseMechanicalState::removeInNode( objectmodel::BaseNode* node )
{
    node->removeMechanicalState(this);
    Inherit1::removeInNode(node);
    return true;
}


} // namespace behavior

} // namespace core

} // namespace sofa
