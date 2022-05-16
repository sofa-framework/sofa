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
#include <sofa/core/behavior/BaseConstraint.h>

namespace sofa::core::behavior
{
int BaseConstraint::getGroup() const
{
    return group.getValue();
}

void BaseConstraint::setGroup(int g)
{
    group.setValue(g);
}

void BaseConstraint::getConstraintInfo(const ConstraintParams* cParams, VecConstraintBlockInfo& blocks,
    VecPersistentID& ids, VecConstCoord& positions, VecConstDeriv& directions, VecConstArea& areas)
{
    SOFA_UNUSED(cParams);
    SOFA_UNUSED(blocks);
    SOFA_UNUSED(ids);
    SOFA_UNUSED(positions);
    SOFA_UNUSED(directions);
    SOFA_UNUSED(areas);

}

void BaseConstraint::getConstraintResolution(const ConstraintParams* cParams,
    std::vector<ConstraintResolution*>& resTab, unsigned& offset)
{
    getConstraintResolution(resTab, offset);
    SOFA_UNUSED(cParams);
}

void BaseConstraint::getConstraintResolution(std::vector<ConstraintResolution*>& resTab, unsigned& offset)
{
    SOFA_UNUSED(resTab);
    SOFA_UNUSED(offset);
}

} // namespace sofa::core::behavior
