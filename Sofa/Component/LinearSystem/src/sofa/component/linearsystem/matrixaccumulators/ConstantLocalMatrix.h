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

    sofa::type::vector<std::size_t> insertionOrderList;
    std::size_t currentId {};

protected:

    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, float value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, double value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, float>& value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, double>& value) override;

};

template <class TMatrix, core::matrixaccumulator::Contribution c>
void ConstantLocalMatrix<TMatrix, c>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, float value)
{
    SOFA_UNUSED(row);
    SOFA_UNUSED(col);
    static_cast<TMatrix*>(this->m_globalMatrix)->colsValue[insertionOrderList[currentId++]]
        += this->m_cachedFactor * value;
}

template <class TMatrix, core::matrixaccumulator::Contribution c>
void ConstantLocalMatrix<TMatrix, c>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, double value)
{
    SOFA_UNUSED(row);
    SOFA_UNUSED(col);
    static_cast<TMatrix*>(this->m_globalMatrix)->colsValue[insertionOrderList[currentId++]]
        += this->m_cachedFactor * value;
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
