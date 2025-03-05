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
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>

namespace sofa::component::linearsystem
{

template<class TReal>
using LocalMappedMatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<TReal>;

/**
 * Similar to AssemblingMatrixAccumulator but dedicated for mapped components.
 * Instead of writing directly into the global matrix, AssemblingMappedMatrixAccumulator builds explicitly
 * a local matrix, so it can be transformed from the mapped
 * space to the global space, using matrix product with the Jacobian matrix of the mapping.
 */
template<sofa::core::matrixaccumulator::Contribution c, class TBlockType>
class AssemblingMappedMatrixAccumulator : public virtual AssemblingMatrixAccumulator<c>
{
public:
    SOFA_CLASS(AssemblingMappedMatrixAccumulator, AssemblingMatrixAccumulator<c>);
    using ComponentType = typename Inherit1::ComponentType;

    using Inherit1::add;

    void clear() override;

    void shareMatrix(const std::shared_ptr<LocalMappedMatrixType<TBlockType> >& m);

    const std::shared_ptr<LocalMappedMatrixType<TBlockType> >& getMatrix() const;

protected:

    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, float value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, double value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, float>& value) override;
    void add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, double>& value) override;

    std::shared_ptr<LocalMappedMatrixType<TBlockType> > m_mappedMatrix;

    AssemblingMappedMatrixAccumulator();
};


template <sofa::core::matrixaccumulator::Contribution c, class TBlockType>
void AssemblingMappedMatrixAccumulator<c, TBlockType>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
    float value)
{
    m_mappedMatrix->add(row, col, value * this->m_cachedFactor);
}

template <sofa::core::matrixaccumulator::Contribution c, class TBlockType>
void AssemblingMappedMatrixAccumulator<c, TBlockType>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
    double value)
{
    m_mappedMatrix->add(row, col, value * this->m_cachedFactor);
}

template <sofa::core::matrixaccumulator::Contribution c, class TBlockType>
void AssemblingMappedMatrixAccumulator<c, TBlockType>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<3, 3, float>& value)
{
    m_mappedMatrix->add(row, col, value * this->m_cachedFactor);
}

template <sofa::core::matrixaccumulator::Contribution c, class TBlockType>
void AssemblingMappedMatrixAccumulator<c, TBlockType>::add(const core::matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<3, 3, double>& value)
{
    m_mappedMatrix->add(row, col, value * this->m_cachedFactor);
}

template <sofa::core::matrixaccumulator::Contribution c, class TBlockType>
void AssemblingMappedMatrixAccumulator<c, TBlockType>::clear()
{}

template <sofa::core::matrixaccumulator::Contribution c, class TBlockType>
void AssemblingMappedMatrixAccumulator<c, TBlockType>::shareMatrix(
    const std::shared_ptr<LocalMappedMatrixType<TBlockType>>& m)
{
    m_mappedMatrix = m;
    this->m_globalMatrix = m_mappedMatrix.get();
}

template <sofa::core::matrixaccumulator::Contribution c, class TBlockType>
const std::shared_ptr<LocalMappedMatrixType<TBlockType>>& AssemblingMappedMatrixAccumulator<c, TBlockType>::
getMatrix() const
{
    return m_mappedMatrix;
}

template <sofa::core::matrixaccumulator::Contribution c, class TBlockType>
AssemblingMappedMatrixAccumulator<c, TBlockType>::AssemblingMappedMatrixAccumulator()
: Inherit1()
{
    this->d_positionInGlobalMatrix.setDisplayed(false); //this inherited Data does not make sense for mapped local matrix
    this->m_globalMatrix = m_mappedMatrix.get();
}


}
