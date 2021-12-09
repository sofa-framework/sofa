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
#include <Sofa.LinearAlgebra.Testing/BaseMatrix_test.h>

#include <SofaNewmat/NewMatMatrix.h>

namespace sofa
{

using namespace sofa::linearalgebra::testing;

using traits_9x9float = TestBaseMatrixTraits< sofa::component::linearsolver::NewMatMatrix, 9, 9, float>;

/// Test instantiation for 9x9 matrices and float scalars
INSTANTIATE_TYPED_TEST_SUITE_P(
    SofaNewmat_test_9x9float,
    TestBaseMatrix,
    traits_9x9float
);

using traits_9x9double = TestBaseMatrixTraits< sofa::component::linearsolver::NewMatMatrix, 9, 9, double>;

/// Test instantiation for 9x9 matrices and double scalars
INSTANTIATE_TYPED_TEST_SUITE_P(
    SofaNewmat_test_9x9double,
    TestBaseMatrix,
    traits_9x9double
);

using traits_9x14float = TestBaseMatrixTraits< sofa::component::linearsolver::NewMatMatrix, 9, 14, float>;

/// Test instantiation for 9x14 matrices and float scalars
INSTANTIATE_TYPED_TEST_SUITE_P(
    SofaNewmat_test_9x14float,
    TestBaseMatrix,
    traits_9x14float
);

using traits_9x14double = TestBaseMatrixTraits< sofa::component::linearsolver::NewMatMatrix, 9, 14, double>;

/// Test instantiation for 9x14 matrices and double scalars
INSTANTIATE_TYPED_TEST_SUITE_P(
    SofaNewmat_test_9x14double,
    TestBaseMatrix,
    traits_9x14double
);
}