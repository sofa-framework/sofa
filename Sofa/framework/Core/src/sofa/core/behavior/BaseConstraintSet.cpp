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
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa::core::behavior
{

BaseConstraintSet::BaseConstraintSet()
    : group(initData(&group, 0, "group", "ID of the group containing this constraint. This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle."))
    , d_constraintIndex(initData(&d_constraintIndex, 0u, "constraintIndex", "Constraint index (first index in the right hand term resolution vector)"))
{
    m_constraintIndex.setParent(&d_constraintIndex);
}


BaseConstraintSet::~BaseConstraintSet()
{

}

bool BaseConstraintSet::insertInNode( objectmodel::BaseNode* node )
{
    node->addConstraintSet(this);
    Inherit1::insertInNode(node);
    return true;
}

bool BaseConstraintSet::removeInNode( objectmodel::BaseNode* node )
{
    node->removeConstraintSet(this);
    Inherit1::removeInNode(node);
    return true;
}


} // namespace sofa::core::behavior

