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
#include <sofa/component/linearsolver/preconditioner/PrecomputedMatrixSystem.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::linearsolver::preconditioner
{

using sofa::linearalgebra::CompressedRowSparseMatrix;
using sofa::linearalgebra::FullVector;

template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API PrecomputedMatrixSystem< linearalgebra::CompressedRowSparseMatrix<SReal>, FullVector<SReal> >;

void registerPrecomputedMatrixSystem(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Precomputed matrix system.")
        .add<PrecomputedMatrixSystem< CompressedRowSparseMatrix<SReal>, FullVector<SReal> > >());
}

}
