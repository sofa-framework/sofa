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
#include <sofa/component/linearsolver/preconditioner/config.h>
#include <sofa/component/linearsystem/TypedMatrixLinearSystem.h>
#include <sofa/core/behavior/BaseRotationFinder.h>

namespace sofa::component::linearsolver::preconditioner
{

template<class TMatrix, class TVector>
class RotationMatrixSystem : public linearsystem::TypedMatrixLinearSystem<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(RotationMatrixSystem, TMatrix, TVector), SOFA_TEMPLATE2(linearsystem::TypedMatrixLinearSystem, TMatrix, TVector));

    Data<unsigned int> d_assemblingRate;

    SingleLink<MyType, sofa::core::behavior::BaseMatrixLinearSystem, BaseLink::FLAG_DUPLICATE> l_mainAssembledSystem;
    SingleLink<MyType, sofa::core::behavior::BaseRotationFinder, BaseLink::FLAG_DUPLICATE> l_rotationFinder;

    void init() override;
    void reset() override;

    void buildSystemMatrix(const core::MechanicalParams* mparams) override;



protected:
    RotationMatrixSystem();

    // count the number of steps since the last assembly
    unsigned int m_assemblyCounter {};

    void reinitAssemblyCounter();
    void updateMatrixWithRotations();
};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_ROTATIONMATRIXSYSTEM_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API RotationMatrixSystem< RotationMatrix<SReal>, FullVector<SReal> >;
#endif

}  // namespace sofa::component::linearsolver::preconditioner
