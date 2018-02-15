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
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa
{

namespace core
{

namespace behavior
{

BaseForceField::BaseForceField()
    : objectmodel::BaseObject()
    , isCompliance( initData(&isCompliance, false, "isCompliance", "Consider the component as a compliance, else as a stiffness"))
    , rayleighStiffness( initData(&rayleighStiffness, SReal(0), "rayleighStiffness", "Rayleigh damping - stiffness matrix coefficient"))
{
}

void BaseForceField::addMBKdx(const MechanicalParams* mparams, MultiVecDerivId dfId)
{
    if (mparams->kFactorIncludingRayleighDamping(rayleighStiffness.getValue()) != 0.0 || mparams->bFactor() != 0.0)
        addDForce(mparams, dfId);
}

void BaseForceField::addBToMatrix(const MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/)
{
}

void BaseForceField::addMBKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (mparams->kFactorIncludingRayleighDamping(rayleighStiffness.getValue()) != 0.0 )
        addKToMatrix(mparams, matrix);
    if (mparams->bFactor() != 0.0)
        addBToMatrix(mparams, matrix);
}

void BaseForceField::addSubMBKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> subMatrixIndex)
{
    if (mparams->kFactorIncludingRayleighDamping(rayleighStiffness.getValue()) != 0.0 )
        addSubKToMatrix(mparams, matrix,subMatrixIndex);
    if (mparams->bFactor() != 0.0)
        addSubBToMatrix(mparams, matrix,subMatrixIndex);
}

void BaseForceField::addSubKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> & /*subMatrixIndex*/) {
    addKToMatrix(mparams,matrix);
}

void BaseForceField::addSubBToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> & /*subMatrixIndex*/) {
    addBToMatrix(mparams,matrix);
}


bool BaseForceField::insertInNode( objectmodel::BaseNode* node )
{
    node->addForceField(this);
    Inherit1::insertInNode(node);
    return true;
}

bool BaseForceField::removeInNode( objectmodel::BaseNode* node )
{
    node->removeForceField(this);
    Inherit1::removeInNode(node);
    return true;
}

} // namespace behavior

} // namespace core

} // namespace sofa
