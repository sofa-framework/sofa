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
    size_t result=0;
    const helper::vector< ConstraintGroup* > &vec = constraintOrder[Order];
    for (size_t i=0; i<vec.size(); ++i) result+=vec[i]->getNumConstraint();
    return static_cast<unsigned int>(result);
}

ConstraintGroup* BaseLMConstraint::addGroupConstraint(ConstraintParams::ConstOrder id)
{
    ConstraintGroup *c = new ConstraintGroup(id);
    constraintOrder[id].push_back(c);
    return c;
}

void BaseLMConstraint::getConstraintViolation(const core::ConstraintParams* cparams, defaulttype::BaseVector *v)
{
    getConstraintViolation(v,cparams->constOrder());
}

void BaseLMConstraint::getConstraintViolation(defaulttype::BaseVector * v, const sofa::core::ConstraintParams::ConstOrder Order)
{
    const helper::vector< ConstraintGroup* > &constraints = constraintOrder[Order];
    for (size_t idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
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
    for (it=constraintOrder.begin(); it!=constraintOrder.end(); ++it)
    {
        helper::vector< ConstraintGroup* > &v=it->second;
        for (size_t i=0; i<v.size(); ++i)
        {
            delete v[i];
        }
    }
    constraintOrder.clear();
}

} // namespace behavior

} // namespace core

} // namespace sofa
