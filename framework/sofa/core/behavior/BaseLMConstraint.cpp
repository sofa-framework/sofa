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
#include <sofa/core/behavior/BaseLMConstraint.h>
namespace sofa
{

namespace core
{

namespace behavior
{
//------------------------------------------------------------------------
//ConstraintGroup

ConstraintGroup::ConstraintGroup(ConstraintParams::ConstOrder idConstraint)
    : Order(idConstraint)
    , active(true)
{
}

void ConstraintGroup::addConstraint( unsigned int &constraintId, unsigned int idx, SReal c)
{
    equations.resize(equations.size()+1);
    ConstraintEquation &eq=equations.back();

    eq.idx = idx;
    eq.correction=c;

    eq.constraintId = constraintId;
    constraintId++;
}
//------------------------------------------------------------------------
BaseLMConstraint::BaseLMConstraint()
    : pathObject1( initData(&pathObject1,  "object1","First Object to constrain") ),
      pathObject2( initData(&pathObject2,  "object2","Second Object to constrain") )
{
}

unsigned int BaseLMConstraint::getNumConstraint(ConstraintParams::ConstOrder Order)
{
    unsigned int result=0;
    const helper::vector< ConstraintGroup* > &vec = constraintOrder[Order];
    for (unsigned int i=0; i<vec.size(); ++i) result+=vec[i]->getNumConstraint();
    return result;
}

ConstraintGroup* BaseLMConstraint::addGroupConstraint(ConstraintParams::ConstOrder id)
{
    ConstraintGroup *c = new ConstraintGroup(id);
    constraintOrder[id].push_back(c);
    return c;
}

void BaseLMConstraint::getConstraintViolation(defaulttype::BaseVector *v, ConstMultiVecId /*vId*/, ConstraintParams::ConstOrder order)
{
    const helper::vector< ConstraintGroup* > &constraints = constraintOrder[order];
    for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
    {
        ConstraintGroup *group=constraints[idxGroupConstraint];
        std::pair< ConstraintGroup::EquationIterator, ConstraintGroup::EquationIterator > range=group->data();

        for (ConstraintGroup::EquationIterator equation = range.first; equation != range.second; ++equation)
        {
            v->set(equation->constraintId, equation->correction);
        }
    }
}


void BaseLMConstraint::resetConstraint()
{
    std::map< ConstraintParams::ConstOrder, helper::vector< ConstraintGroup*> >::iterator it;
    for (it=constraintOrder.begin(); it!=constraintOrder.end(); it++)
    {
        helper::vector< ConstraintGroup* > &v=it->second;
        for (unsigned int i=0; i<v.size(); ++i)
        {
            delete v[i];
        }
    }
    constraintOrder.clear();
}

} // namespace behavior

} // namespace core

} // namespace sofa
