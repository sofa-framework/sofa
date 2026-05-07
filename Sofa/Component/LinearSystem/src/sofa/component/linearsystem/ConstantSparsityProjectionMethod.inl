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
#include <sofa/component/linearsystem/ConstantSparsityProjectionMethod.h>

#include <Eigen/Sparse>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/ParallelSparseMatrixProduct.h>

namespace sofa::component::linearsystem
{

template <class TMatrix>
ConstantSparsityProjectionMethod<TMatrix>::ConstantSparsityProjectionMethod()
    : d_parallelProduct(initData(&d_parallelProduct, true, "parallelProduct", "Compute the matrix product in parallel"))
{}

template <class TMatrix>
ConstantSparsityProjectionMethod<TMatrix>::
~ConstantSparsityProjectionMethod() = default;

template <class TMatrix>
void ConstantSparsityProjectionMethod<TMatrix>::init()
{
    Inherit1::init();

    if (d_parallelProduct.getValue())
    {
        auto* taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
        taskScheduler->init();

        auto* matrixPrductKJ = new sofa::simulation::ParallelSparseMatrixProduct<
            K_Type, J_Type, KJ_Type>();

        m_matrixProductKJ = std::unique_ptr<sofa::simulation::ParallelSparseMatrixProduct<
            K_Type, J_Type, KJ_Type>>(matrixPrductKJ);

        matrixPrductKJ->taskScheduler = taskScheduler;


        auto* matrixPrductJTKJ = new sofa::simulation::ParallelSparseMatrixProduct<
            JT_Type, KJ_Type, JTKJ_Type>();

        m_matrixProductJTKJ = std::unique_ptr<sofa::simulation::ParallelSparseMatrixProduct<
            JT_Type, KJ_Type, JTKJ_Type>>(matrixPrductJTKJ);

        matrixPrductJTKJ->taskScheduler = taskScheduler;
    }
    else
    {
        m_matrixProductKJ = std::make_unique<sofa::linearalgebra::SparseMatrixProduct<
            K_Type, J_Type, KJ_Type>>();

        m_matrixProductJTKJ = std::make_unique<sofa::linearalgebra::SparseMatrixProduct<
            JT_Type, KJ_Type, JTKJ_Type>>();
    }
}

template <class TMatrix>
void ConstantSparsityProjectionMethod<TMatrix>::reinit()
{
    Inherit1::reinit();

    //cached products are invalidated
    m_matrixProductKJ->invalidateIntersection();
    m_matrixProductJTKJ->invalidateIntersection();
}

template <class TMatrix>
void ConstantSparsityProjectionMethod<TMatrix>::computeProjection(
    const Eigen::Map<Eigen::SparseMatrix<Block, Eigen::RowMajor>> KMap,
    const sofa::type::fixed_array<std::shared_ptr<TMatrix>, 2> J,
    Eigen::SparseMatrix<Block, Eigen::RowMajor>& JT_K_J)
{
    if (J[0] && J[1])
    {
        const auto JMap0 = this->makeEigenMap(*J[0]);
        const auto JMap1 = this->makeEigenMap(*J[1]);

        m_matrixProductKJ->m_lhs = &KMap;
        m_matrixProductKJ->m_rhs = &JMap1;
        m_matrixProductKJ->computeProduct();

        const JT_Type JMap0T = JMap0.transpose();
        m_matrixProductJTKJ->m_lhs = &JMap0T;
        m_matrixProductJTKJ->m_rhs = &m_matrixProductKJ->getProductResult();
        m_matrixProductJTKJ->computeProduct();

        JT_K_J = m_matrixProductJTKJ->getProductResult();
    }
    else if (J[0] && !J[1])
    {
        const auto JMap0 = this->makeEigenMap(*J[0]);
        JT_K_J = JMap0.transpose() * KMap;
    }
    else if (!J[0] && J[1])
    {
        const auto JMap1 = this->makeEigenMap(*J[1]);

        m_matrixProductKJ->m_lhs = &KMap;
        m_matrixProductKJ->m_rhs = &JMap1;

        m_matrixProductKJ->computeProduct();

        JT_K_J = m_matrixProductKJ->getProductResult();
    }
    else
    {
        JT_K_J = KMap;
    }
}


}
