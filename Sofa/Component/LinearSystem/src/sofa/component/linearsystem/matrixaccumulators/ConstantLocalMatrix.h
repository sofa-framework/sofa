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

#include <sofa/component/linearsystem/MatrixLinearSystem.h>

namespace sofa::component::linearsystem
{
/**
 * Local matrix using the insertion order to insert the value directly into the compressed matrix
 */
template<class TMatrix, core::matrixaccumulator::Contribution c>
class ConstantLocalMatrix : public virtual AssemblingMatrixAccumulator<c>
{
public:
    SOFA_CLASS(ConstantLocalMatrix, AssemblingMatrixAccumulator<c>);
    using ComponentType = typename Inherit1::ComponentType;

    using Inherit1::add;
    using Row = sofa::SignedIndex;
    using Col = sofa::SignedIndex;

    /// list of expected rows and columns
    sofa::type::vector<std::pair<Row, Col> > pairInsertionOrderList;

    /// list of indices in the compressed values
    sofa::type::vector<std::size_t> compressedInsertionOrderList;

    std::size_t currentId {};

    /// Enumeration representing possible errors during insertion into the compressed matrix
    enum class InsertionOrderError : char
    {
        NO_INSERTION_ERROR,
        NOT_EXPECTED_ROW_COL,
        TOO_MUCH_INCOMING_VALUES
    };

protected:

    InsertionOrderError checkInsertionOrderIsConstant(sofa::SignedIndex row, sofa::SignedIndex col);

    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, float value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, double value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, float>& value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, double>& value) override;

};


template <class TMatrix, core::matrixaccumulator::Contribution c>
auto ConstantLocalMatrix<TMatrix, c>::checkInsertionOrderIsConstant(
    const sofa::SignedIndex row,
    const sofa::SignedIndex col) -> InsertionOrderError
{
    if (currentId < pairInsertionOrderList.size())
    {
        const auto& [expectedRow, expectedCol] = pairInsertionOrderList[currentId];
        const bool isRowExpected = expectedRow == row;
        const bool isColExpected = expectedCol == col;
        if (!isRowExpected || !isColExpected)
        {
            msg_error() << "According to the constant sparsity pattern, the "
                    "expected row and column are [" << expectedRow << ", " <<
                    expectedCol << "], but " << "[" << row << ", " << col <<
                    "] was received. Insertion is ignored.";
            return InsertionOrderError::NOT_EXPECTED_ROW_COL;
        }
        return InsertionOrderError::NO_INSERTION_ERROR;
    }
    else
    {
        msg_error() <<
                "The constant sparsity pattern did not expect more incoming matrix values at this stage. Insertion is ignored.";
        return InsertionOrderError::TOO_MUCH_INCOMING_VALUES;
    }
}

template <class TMatrix, core::matrixaccumulator::Contribution c>
void ConstantLocalMatrix<TMatrix, c>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, float value)
{
    if (checkInsertionOrderIsConstant(row, col) == InsertionOrderError::NO_INSERTION_ERROR)
    {
        static_cast<TMatrix*>(this->m_globalMatrix)->colsValue[compressedInsertionOrderList[currentId++]]
                += this->m_cachedFactor * value;
    }
}

template <class TMatrix, core::matrixaccumulator::Contribution c>
void ConstantLocalMatrix<TMatrix, c>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, double value)
{
    if (checkInsertionOrderIsConstant(row, col) == InsertionOrderError::NO_INSERTION_ERROR)
    {
        static_cast<TMatrix*>(this->m_globalMatrix)->colsValue[compressedInsertionOrderList[currentId++]]
            += this->m_cachedFactor * value;
    }
}

template <class TMatrix, core::matrixaccumulator::Contribution c>
void ConstantLocalMatrix<TMatrix, c>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<3, 3, float>& value)
{
    for (sofa::SignedIndex i = 0; i < 3; ++i)
    {
        for (sofa::SignedIndex j = 0; j < 3; ++j)
        {
            this->add(core::matrixaccumulator::no_check, row + i, col + j, value(i, j));
        }
    }
}

template <class TMatrix, core::matrixaccumulator::Contribution c>
void ConstantLocalMatrix<TMatrix, c>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<3, 3, double>& value)
{
    for (sofa::SignedIndex i = 0; i < 3; ++i)
    {
        for (sofa::SignedIndex j = 0; j < 3; ++j)
        {
            this->add(core::matrixaccumulator::no_check, row + i, col + j, value(i, j));
        }
    }
}


}
