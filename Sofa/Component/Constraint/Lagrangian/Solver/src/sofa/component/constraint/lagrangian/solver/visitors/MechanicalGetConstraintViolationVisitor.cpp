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
#include <sofa/component/constraint/lagrangian/solver/visitors/MechanicalGetConstraintViolationVisitor.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/behavior/BaseConstraintSet.h>

namespace sofa::component::constraint::lagrangian::solver
{

MechanicalGetConstraintViolationVisitor::MechanicalGetConstraintViolationVisitor(
    const core::ConstraintParams* params, sofa::linearalgebra::BaseVector* v): simulation::BaseMechanicalVisitor(params)
    , cparams(params)
    , m_v(v)
{}

simulation::Visitor::Result MechanicalGetConstraintViolationVisitor::fwdConstraintSet(
    simulation::Node* node, core::behavior::BaseConstraintSet* c)
{
    const ctime_t t0 = begin(node, c);
    c->getConstraintViolation(cparams, m_v);
    end(node, c, t0);
    return RESULT_CONTINUE;
}

bool MechanicalGetConstraintViolationVisitor::stopAtMechanicalMapping(simulation::Node* node,
    core::BaseMapping* base_mapping)
{
    SOFA_UNUSED(node);
    SOFA_UNUSED(base_mapping);
    return false; // !map->isMechanical();
}
}

