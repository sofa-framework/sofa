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
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/objectmodel/BaseNode.h>

#include <sofa/core/behavior/MatrixAPICompatibility.h>
#include <sofa/core/MechanicalParams.h>

namespace sofa::core::behavior
{
void BaseProjectiveConstraintSet::applyConstraint(sofa::core::behavior::ZeroDirichletCondition* zeroDirichletCondition)
{
    static std::set<BaseProjectiveConstraintSet*> hasEmittedWarning;
    if (hasEmittedWarning.insert(this).second)
    {
        dmsg_warning() << "applyConstraint(ZeroDirichletCondition*) not implemented: for compatibility reason, the "
            "deprecated API (applyConstraint(MultiMatrixAccessor*)) will be used (without guarantee). This compatibility will disapear in the "
            "future, and will cause issues in simulations. Please update the code of " <<
            this->getClassName() << " to ensure right behavior: the function applyConstraint(MultiMatrixAccessor*) "
            "has been replaced by applyConstraint(ZeroDirichletCondition*)";
    }

    MatrixAccessorCompat accessor;

    const auto& mstates = this->getMechanicalStates();
    std::set<BaseMechanicalState*> uniqueMstates(mstates.begin(), mstates.end());

    for(const auto& mstate1 : uniqueMstates)
    {
        if (mstate1)
        {
            for(const auto& mstate2 : uniqueMstates)
            {
                if (mstate2)
                {
                    const auto mat = std::make_shared<ApplyConstraintCompat>();
                    mat->component = this;
                    mat->zeroDirichletCondition = zeroDirichletCondition;
                    accessor.setMatrix(mstate1, mstate2, mat);
                }
            }
        }
    }

    MechanicalParams params;
    params.setKFactor(1.);
    params.setMFactor(1.);
    params.setBFactor(1.);

    applyConstraint(&params, &accessor);
}

bool BaseProjectiveConstraintSet::insertInNode( objectmodel::BaseNode* node )
{
    node->addProjectiveConstraintSet(this);
    Inherit1::insertInNode(node);
    return true;
}

bool BaseProjectiveConstraintSet::removeInNode( objectmodel::BaseNode* node )
{
    node->removeProjectiveConstraintSet(this);
    Inherit1::removeInNode(node);
    return true;
}

} // namespace sofa::core::behavior
