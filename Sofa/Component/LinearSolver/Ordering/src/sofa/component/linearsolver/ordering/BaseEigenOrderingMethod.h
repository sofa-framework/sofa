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

#include <sofa/core/behavior/BaseOrderingMethod.h>
#include <Eigen/SparseCore>

namespace sofa::component::linearsolver::ordering
{

/**
 * \brief Ordering method implemented in the library Eigen
 * \tparam EigenOrderingMethodType The Eigen type for the ordering algorithm. For example Eigen::AMDOdering<int>
 */
template<class EigenOrderingMethodType>
class BaseEigenOrderingMethod : public core::behavior::BaseOrderingMethod
{
public:
    SOFA_CLASS(BaseEigenOrderingMethod, core::behavior::BaseOrderingMethod);

    using core::behavior::BaseOrderingMethod::SparseMatrixPattern;

    void computePermutation(
        const SparseMatrixPattern& inPattern,
        int* outPermutation,
        int* outInversePermutation) override;
};

template <class EigenOrderingMethodType>
void BaseEigenOrderingMethod<EigenOrderingMethodType>::computePermutation(
    const SparseMatrixPattern& inPattern, int* outPermutation,
    int* outInversePermutation)
{
    EigenOrderingMethodType ordering;
    using PermutationType = typename EigenOrderingMethodType::PermutationType;
    PermutationType permutation;
    using EigenSparseMatrix = Eigen::SparseMatrix<SReal, Eigen::ColMajor>;
    using EigenSparseMatrixMap = Eigen::Map<const EigenSparseMatrix>;

    //the values are not important to compute the permutation, but it is still
    //required to build the Eigen::Map
    const std::vector<SReal> fakeValues(inPattern.numberOfNonZeros);

    const auto map = EigenSparseMatrixMap( inPattern.matrixSize, inPattern.matrixSize, inPattern.numberOfNonZeros,
        inPattern.rowBegin, inPattern.colsIndex, fakeValues.data());

    //permutation
    ordering.template operator()<const EigenSparseMatrix>(map, permutation);

    const PermutationType inversePermutation = permutation.inverse();

    for (int j = 0; j < inPattern.matrixSize; ++j)
    {
        outPermutation[j] = permutation.indices()(j);
        outInversePermutation[j] = inversePermutation.indices()(j) ;
    }
}

}
