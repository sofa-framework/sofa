/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/behavior/BaseMechanicalState.h>


namespace sofa
{

namespace core
{

namespace behavior
{

BaseMechanicalState::BaseMechanicalState()
    : useMask(initData(&useMask, true, "useMask", "Usage of a mask to optimize the computation of the system, highly reducing the passage through the mappings"))
    , forceMask(&useMask)
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
void BaseMechanicalState::vMultiOp(const ExecParams* params /* PARAMS FIRST */, const VMultiOp& ops)
{
    for(VMultiOp::const_iterator it = ops.begin(), itend = ops.end(); it != itend; ++it)
    {
        VecId r = it->first.getId(this);
        const helper::vector< std::pair< ConstMultiVecId, double > >& operands = it->second;
        int nop = operands.size();
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
            int i;
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
void BaseMechanicalState::handleStateChange(core::topology::Topology* t)
{
    if (t == this->getContext()->getTopology())
        handleStateChange();
}

void BaseMechanicalState::writeState( std::ostream& )
{ }

} // namespace behavior

} // namespace core

} // namespace sofa
