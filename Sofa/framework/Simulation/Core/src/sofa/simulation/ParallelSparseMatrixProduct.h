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
#include <sofa/linearalgebra/SparseMatrixProduct.inl>
#include <sofa/simulation/task/ParallelForEach.h>
#include <sofa/simulation/task/TaskScheduler.h>


namespace sofa::simulation
{

template<class Lhs, class Rhs, class ResultType>
class ParallelSparseMatrixProduct
    : public linearalgebra::SparseMatrixProduct<Lhs, Rhs, ResultType>
{
public:
    using linearalgebra::SparseMatrixProduct<Lhs, Rhs, ResultType>::SparseMatrixProduct;
    TaskScheduler* taskScheduler { nullptr };

    void computeProductFromIntersection() override
    {
        assert(this->m_intersectionAB.intersection.size() == this->m_productResult.nonZeros());
        assert(taskScheduler);

        auto* lhs_ptr = this->m_lhs->valuePtr();
        auto* rhs_ptr = this->m_rhs->valuePtr();
        auto* product_ptr = this->m_productResult.valuePtr();

        parallelForEachRange(*taskScheduler,
            this->m_intersectionAB.intersection.begin(), this->m_intersectionAB.intersection.end(),
            [lhs_ptr, rhs_ptr, product_ptr, this](const auto& range)
            {
                auto i = std::distance(this->m_intersectionAB.intersection.begin(), range.start);
                auto* p = product_ptr + i;

                for (auto it = range.start; it != range.end; ++it)
                {
                    auto& value = *p++;
                    value = 0;
                    for (const auto& [lhsIndex, rhsIndex] : *it)
                    {
                        value += lhs_ptr[lhsIndex] * rhs_ptr[rhsIndex];
                    }
                }
            });
    }
};

}
