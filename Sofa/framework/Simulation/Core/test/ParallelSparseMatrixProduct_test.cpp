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
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/MainTaskSchedulerRegistry.h>
#include <Sofa.LinearAlgebra.Testing/SparseMatrixProduct_test.h>
#include <sofa/simulation/ParallelSparseMatrixProduct.h>

namespace sofa
{

using namespace sofa::linearalgebra::testing;

template<class Lhs, class Rhs, class ResultType>
struct SparseMatrixProductInit<
    sofa::simulation::ParallelSparseMatrixProduct<Lhs, Rhs, ResultType>>
{
    static void init(sofa::simulation::ParallelSparseMatrixProduct<Lhs, Rhs, ResultType>& product)
    {
        product.taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
        product.taskScheduler->init();
    };

    static void cleanup(sofa::simulation::ParallelSparseMatrixProduct<Lhs, Rhs, ResultType>& product)
    {
        SOFA_UNUSED(product);
        // simulation::MainTaskSchedulerRegistry::clear();
    }
};

#define DEFINE_TEST_FOR_TYPE(scalar, StorageLHS, StorageRHS, StorageResult)\
    sofa::simulation::ParallelSparseMatrixProduct<\
        Eigen::SparseMatrix<scalar, StorageLHS>,\
        Eigen::SparseMatrix<scalar, StorageRHS>,\
        Eigen::SparseMatrix<scalar, StorageResult>\
    >
#define DEFINE_TEST_FOR_STORAGE(StorageLHS, StorageRHS, StorageResult)\
    DEFINE_TEST_FOR_TYPE(float, StorageLHS, StorageRHS, StorageResult),\
    DEFINE_TEST_FOR_TYPE(double, StorageLHS, StorageRHS, StorageResult)

using TestSparseMatrixProductImplementations = ::testing::Types<
    DEFINE_TEST_FOR_STORAGE(Eigen::ColMajor, Eigen::ColMajor, Eigen::ColMajor),
    DEFINE_TEST_FOR_STORAGE(Eigen::RowMajor, Eigen::ColMajor, Eigen::ColMajor),
    DEFINE_TEST_FOR_STORAGE(Eigen::ColMajor, Eigen::RowMajor, Eigen::ColMajor),
    DEFINE_TEST_FOR_STORAGE(Eigen::RowMajor, Eigen::RowMajor, Eigen::ColMajor),
    DEFINE_TEST_FOR_STORAGE(Eigen::ColMajor, Eigen::ColMajor, Eigen::RowMajor),
    DEFINE_TEST_FOR_STORAGE(Eigen::RowMajor, Eigen::ColMajor, Eigen::RowMajor),
    DEFINE_TEST_FOR_STORAGE(Eigen::ColMajor, Eigen::RowMajor, Eigen::RowMajor),
    DEFINE_TEST_FOR_STORAGE(Eigen::RowMajor, Eigen::RowMajor, Eigen::RowMajor)
>;

#undef DEFINE_TEST_FOR_STORAGE
#undef DEFINE_TEST_FOR_TYPE

INSTANTIATE_TYPED_TEST_SUITE_P(
    TestParallelSparseMatrixProduct,
    TestSparseMatrixProduct,
    TestSparseMatrixProductImplementations
);

}
