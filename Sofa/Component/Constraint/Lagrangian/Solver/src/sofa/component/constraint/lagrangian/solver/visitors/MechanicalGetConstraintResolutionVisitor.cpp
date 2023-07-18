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
#include <sofa/component/constraint/lagrangian/solver/visitors/MechanicalGetConstraintResolutionVisitor.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/behavior/BaseConstraint.h>

namespace sofa::component::constraint::lagrangian::solver
{
MechanicalGetConstraintResolutionVisitor::MechanicalGetConstraintResolutionVisitor(const core::ConstraintParams* params, std::vector<core::behavior::ConstraintResolution*>& res)
: simulation::BaseMechanicalVisitor(params)
, cparams(params)
, _res(res)
, _offset(0)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
}

MechanicalGetConstraintResolutionVisitor::Result MechanicalGetConstraintResolutionVisitor::fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet)
{
    if (core::behavior::BaseConstraint *c=cSet->toBaseConstraint())
    {
        const ctime_t t0 = begin(node, c);
        c->getConstraintResolution(cparams, _res, _offset);
        end(node, c, t0);
    }
    return RESULT_CONTINUE;
}

/// Return a class name for this visitor
/// Only used for debugging / profiling purposes
const char* MechanicalGetConstraintResolutionVisitor::getClassName() const
{
    return "MechanicalGetConstraintResolutionVisitor";
}

bool MechanicalGetConstraintResolutionVisitor::isThreadSafe() const
{
    return false;
}

bool MechanicalGetConstraintResolutionVisitor::stopAtMechanicalMapping(simulation::Node* node, core::BaseMapping* map)
{
    SOFA_UNUSED(node);
    SOFA_UNUSED(map);
    return false;
}
}
