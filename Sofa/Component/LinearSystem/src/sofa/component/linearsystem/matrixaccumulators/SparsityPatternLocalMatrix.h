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

#include <sofa/component/linearsystem/matrixaccumulators/AssemblingMatrixAccumulator.h>

namespace sofa::component::linearsystem
{


/**
 * Local matrix used to collect the order values are inserted into the matrix
 * This local matrix is used only once.
 */
template<core::matrixaccumulator::Contribution c, class TStrategy = sofa::core::matrixaccumulator::NoIndexVerification>
class SparsityPatternLocalMatrix : public virtual AssemblingMatrixAccumulator<c, TStrategy>
{
public:
    SOFA_CLASS(SparsityPatternLocalMatrix, SOFA_TEMPLATE2(AssemblingMatrixAccumulator, c, TStrategy));
    using ComponentType = typename Inherit1::ComponentType;

    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, float value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, double value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, float>& value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, double>& value) override;

    using Row = sofa::SignedIndex;
    using Col = sofa::SignedIndex;

    [[nodiscard]]
    const sofa::type::vector<std::pair<Row, Col> >& getInsertionOrderList() const
    { return insertionOrderList; }

protected:

    sofa::type::vector<std::pair<Row, Col> > insertionOrderList;
};


template <core::matrixaccumulator::Contribution c, class TStrategy>
void SparsityPatternLocalMatrix<c, TStrategy>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, float value)
{
    insertionOrderList.emplace_back(row + this->m_cachedPositionInGlobalMatrix[0], col + this->m_cachedPositionInGlobalMatrix[1]);
    Inherit1::add(core::matrixaccumulator::no_check, row, col, value);
}

template <core::matrixaccumulator::Contribution c, class TStrategy>
void SparsityPatternLocalMatrix<c, TStrategy>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, double value)
{
    insertionOrderList.emplace_back(row + this->m_cachedPositionInGlobalMatrix[0], col + this->m_cachedPositionInGlobalMatrix[1]);
    Inherit1::add(core::matrixaccumulator::no_check, row, col, value);
}

template <core::matrixaccumulator::Contribution c, class TStrategy>
void SparsityPatternLocalMatrix<c, TStrategy>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
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

template <core::matrixaccumulator::Contribution c, class TStrategy>
void SparsityPatternLocalMatrix<c, TStrategy>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
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
