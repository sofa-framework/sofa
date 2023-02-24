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
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa::core::behavior
{

BaseMass::BaseMass()
    : rayleighMass (initData(&rayleighMass , 0_sreal, "rayleighMass", "Rayleigh damping - mass matrix coefficient"))
{
}

void BaseMass::addMBKdx(const MechanicalParams* mparams, MultiVecDerivId dfId)
{
    if (sofa::core::mechanicalparams::mFactorIncludingRayleighDamping(mparams,rayleighMass.getValue()) != 0.0)
    {
        addMDx(mparams, dfId, sofa::core::mechanicalparams::mFactorIncludingRayleighDamping(mparams,rayleighMass.getValue()));
    }
}

void BaseMass::addMBKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (sofa::core::mechanicalparams::mFactorIncludingRayleighDamping(mparams,rayleighMass.getValue()) != 0.0 )
    {
        addMToMatrix(mparams, matrix);
    }
}

bool BaseMass::insertInNode( objectmodel::BaseNode* node )
{
    node->addMass(this);
    Inherit1::insertInNode(node);
    return true;
}

bool BaseMass::removeInNode( objectmodel::BaseNode* node )
{
    node->removeMass(this);
    Inherit1::removeInNode(node);
    return true;
}


} // namespace sofa::core::behavior

