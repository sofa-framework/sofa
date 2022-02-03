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

#include <sofa/simulation/mechanicalvisitor/MechanicalGetConstraintInfoVisitor.h>
#include <sofa/core/ConstraintParams.h>

namespace sofa::simulation::mechanicalvisitor
{

MechanicalGetConstraintInfoVisitor::MechanicalGetConstraintInfoVisitor(const core::ConstraintParams* params,
    VecConstraintBlockInfo& blocks, VecPersistentID& ids, VecConstCoord& positions, VecConstDeriv& directions,
    VecConstArea& areas)
    : simulation::BaseMechanicalVisitor(params)
    , _blocks(blocks)
    , _ids(ids)
    , _positions(positions)
    , _directions(directions)
    , _areas(areas)
    , _cparams(params)
{}

Visitor::Result MechanicalGetConstraintInfoVisitor::fwdConstraintSet(simulation::Node* node,
    core::behavior::BaseConstraintSet* cSet)
{
    if (core::behavior::BaseConstraint *c=cSet->toBaseConstraint())
    {
        const ctime_t t0 = begin(node, c);
        c->getConstraintInfo(_cparams, _blocks, _ids, _positions, _directions, _areas);
        end(node, c, t0);
    }
    return RESULT_CONTINUE;
}

bool MechanicalGetConstraintInfoVisitor::stopAtMechanicalMapping(simulation::Node*,
    core::BaseMapping*)
{
    return false;
}

} //namespace sofa::simulation::mechanicalvisitor
