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

#include <sofa/component/linearsystem/matrixaccumulators/SparsityPatternLocalMatrix.h>
#include <sofa/component/linearsystem/matrixaccumulators/AssemblingMappedMatrixAccumulator.h>

namespace sofa::component::linearsystem
{

template<core::matrixaccumulator::Contribution c, class TBlockType>
class SparsityPatternLocalMappedMatrix :
    public SparsityPatternLocalMatrix<c>,
    public AssemblingMappedMatrixAccumulator<c, TBlockType>
{
public:
    SOFA_CLASS2(SparsityPatternLocalMappedMatrix, SparsityPatternLocalMatrix<c>, SOFA_TEMPLATE2(AssemblingMappedMatrixAccumulator, c, TBlockType));
    using ComponentType = typename Inherit1::ComponentType;

    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, float value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, double value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, float>& value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, double>& value) override;
};


template <core::matrixaccumulator::Contribution c, class TBlockType>
void SparsityPatternLocalMappedMatrix<c, TBlockType>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
    float value)
{
    this->insertionOrderList.emplace_back(row + this->m_cachedPositionInGlobalMatrix[0], col + this->m_cachedPositionInGlobalMatrix[1]);
    Inherit2::add(core::matrixaccumulator::no_check, row, col, value);
}

template <core::matrixaccumulator::Contribution c, class TBlockType>
void SparsityPatternLocalMappedMatrix<c, TBlockType>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
    double value)
{
    this->insertionOrderList.emplace_back(row + this->m_cachedPositionInGlobalMatrix[0], col + this->m_cachedPositionInGlobalMatrix[1]);
    Inherit2::add(core::matrixaccumulator::no_check, row, col, value);
}

template <core::matrixaccumulator::Contribution c, class TBlockType>
void SparsityPatternLocalMappedMatrix<c, TBlockType>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<3, 3, float>& value)
{
    Inherit1::add(core::matrixaccumulator::no_check, row, col, value);
}

template <core::matrixaccumulator::Contribution c, class TBlockType>
void SparsityPatternLocalMappedMatrix<c, TBlockType>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<3, 3, double>& value)
{
    Inherit1::add(core::matrixaccumulator::no_check, row, col, value);
}


}
