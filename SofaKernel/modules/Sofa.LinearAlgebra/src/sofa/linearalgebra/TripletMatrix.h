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
#include <Eigen/src/SparseCore/SparseUtil.h>
#include <sofa/linearalgebra/config.h>

#include <sofa/linearalgebra/BaseMatrix.h>

#include "sofa/type/vector_T.h"


namespace sofa::linearalgebra
{

/**
 * Store matrix entries as a list of triplets containing:
 * - the index of the row
 * - the index of the column
 * - the value at the location pointed by both indices
 *
 * This class is not meant to be used as a matrix, but rather as a proxy of a BaseMatrix to build a triplet list and
 * provide it later to some processing (in Eigen SparseMatrix::setFromTriplets for example).
 *
 * Duplicates are not removed.
 */
template<class TReal>
class TripletMatrix : public linearalgebra::BaseMatrix
{
public:

    Index rowSize() const override;
    Index colSize() const override;

    Index nRow, nCol;

    void resize(Index nbRow, Index nbCol) override;

    SReal element(Index i, Index j) const override;

    void set(Index i, Index j, double v) override;
    void add(Index row, Index col, double v) override;

    void clear() override;

    const sofa::type::vector<Eigen::Triplet<TReal> >& getTripletList() const;

private:

    sofa::type::vector<Eigen::Triplet<TReal> > m_tripletList;
};

template <class TReal>
BaseMatrix::Index TripletMatrix<TReal>::rowSize() const
{
    return nRow;
}

template <class TReal>
BaseMatrix::Index TripletMatrix<TReal>::colSize() const
{
    return nCol;
}

template <class TReal>
void TripletMatrix<TReal>::resize(const Index nbRow, const Index nbCol)
{
    nRow = nbRow;
    nCol = nbCol;
}

template <class TReal>
SReal TripletMatrix<TReal>::element(Index i, Index j) const
{
    return {};
}

template <class TReal>
void TripletMatrix<TReal>::set(Index i, Index j, double v)
{
}

template <class TReal>
void TripletMatrix<TReal>::add(Index row, Index col, double v)
{
    m_tripletList.emplace_back(row, col, v);
}

template <class TReal>
void TripletMatrix<TReal>::clear()
{
    m_tripletList.clear();
}

template <class TReal>
const sofa::type::vector<Eigen::Triplet<TReal>>& TripletMatrix<TReal>::getTripletList() const
{
    return m_tripletList;
}
}