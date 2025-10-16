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
#include <sofa/core/behavior/BaseMatrixLinearSystem.h>

namespace sofa::core::behavior
{

BaseMatrixLinearSystem::BaseMatrixLinearSystem()
: Inherit1()
, d_matrixSize(initData(&d_matrixSize, "matrixSize", "Size of the global matrix"))
, d_enableAssembly(initData(&d_enableAssembly, true, "enableAssembly", "Allows to assemble the system matrix"))
{
    d_matrixSize.setReadOnly(true);

    d_enableAssembly.setReadOnly(true);
    d_enableAssembly.setDisplayed(false);
}

void BaseMatrixLinearSystem::buildSystemMatrix(const core::MechanicalParams* mparams)
{
    if (d_enableAssembly.getValue())
    {
        preAssembleSystem(mparams);
        assembleSystem(mparams);
        postAssembleSystem(mparams);
    }
}

void BaseMatrixLinearSystem::preAssembleSystem(const core::MechanicalParams* mparams)
{
    SOFA_UNUSED(mparams);
}

void BaseMatrixLinearSystem::assembleSystem(const core::MechanicalParams* mparams)
{
    SOFA_UNUSED(mparams);
}

} //namespace sofa::core::behavior
