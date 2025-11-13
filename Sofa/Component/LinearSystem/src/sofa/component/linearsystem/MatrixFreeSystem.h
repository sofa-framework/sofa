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
#pragma once
#include <sofa/component/linearsystem/config.h>

#include <sofa/component/linearsystem/TypedMatrixLinearSystem.h>

namespace sofa::component::linearsystem
{

template<class TMatrix, class TVector>
class SOFA_COMPONENT_LINEARSYSTEM_API MatrixFreeSystem : public TypedMatrixLinearSystem<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(MatrixFreeSystem, TMatrix, TVector), SOFA_TEMPLATE2(TypedMatrixLinearSystem, TMatrix, TVector));

    void assembleSystem(const core::MechanicalParams* mparams) override
    {
        this->getSystemMatrix()->setMBKFacts(mparams);
    }
    void associateLocalMatrixToComponents(const core::MechanicalParams* /* mparams */) override {}
};

} //namespace sofa::component::linearsystem
