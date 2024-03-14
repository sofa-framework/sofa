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
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraint.h>
#include <Eigen/Sparse>
#include <sofa/type/Vec.h>
//#include <sofa/defaulttype/RigidTypes.h>
#include <cassert>
#include <type_traits>
#include <cstdlib>

namespace sofa::linearalgebra
{

template< class TBlock >
struct CompressedRowSparseMatrixToEigenSparse
{

};

template <typename TVec>
struct CompressedRowSparseMatrixToEigenSparseVec
{
    typedef typename TVec::Real Real;
    typedef CompressedRowSparseMatrixConstraint< TVec > TCompressedRowSparseMatrix;
    typedef Eigen::SparseMatrix<Real, Eigen::RowMajor> EigenSparseMatrix;


    EigenSparseMatrix operator() (const TCompressedRowSparseMatrix& mat, std::size_t size)
    {
        const std::size_t eigenMatSize = size * TVec::size();
        EigenSparseMatrix eigenMat(eigenMatSize, eigenMatSize);

        sofa::type::vector<Eigen::Triplet<Real> > triplets;

        for (auto row = mat.begin(); row != mat.end(); ++row)
        {
            for (auto col = row.begin(), colend = row.end(); col !=colend; ++col)
            {
                const TVec& vec = col.val();
                const int   colIndex  = col.index() * TVec::size();

                for (std::size_t i = 0; i < TVec::size(); ++i)
                {
                    triplets.emplace_back(row.index(), colIndex + i, vec[i]);
                }

            }
        }

        eigenMat.setFromTriplets(triplets.begin(), triplets.end());;

        return eigenMat;
    }

};

template< int N, typename Real >
class CompressedRowSparseMatrixToEigenSparse< sofa::type::Vec<N,Real> >
    : public  CompressedRowSparseMatrixToEigenSparseVec< sofa::type::Vec<N, Real> >
{

};



template< class TBlock >
struct EigenSparseToCompressedRowSparseMatrix
{

};


template <typename TVec>
struct EigenSparseToCompressedRowSparseMatrixVec
{
    typedef typename TVec::Real Real;
    typedef CompressedRowSparseMatrixConstraint< TVec > TCompressedRowSparseMatrix;
    typedef Eigen::SparseMatrix<Real, Eigen::RowMajor> EigenSparseMatrix;


    TCompressedRowSparseMatrix operator() (const EigenSparseMatrix& eigenMat)
    {
        TCompressedRowSparseMatrix mat;

        const int* outerIndexPtr  = eigenMat.outerIndexPtr();
        const int* innerIndexPtr  = eigenMat.innerIndexPtr();
        const Real* valuePtr      = eigenMat.valuePtr();

        for (int rowIndex = 0; rowIndex < eigenMat.outerSize(); ++rowIndex)
        {
            const int offset      = *(outerIndexPtr + rowIndex);
            const int rowNonZeros = *(outerIndexPtr + rowIndex + 1) - *(outerIndexPtr + rowIndex);

            if (rowNonZeros != 0)
            {
                auto rowIterator = mat.writeLine(rowIndex);

                int i = 0;
                const int*  colPtr = innerIndexPtr + offset;

                int   blockIndex   = *colPtr / TVec::size();
                int   blockOffset  = *colPtr - (blockIndex * TVec::size());


                while (i != rowNonZeros)
                {
                    TVec val;
                    int currenTBlockkIndex = blockIndex;
                    while (currenTBlockkIndex == blockIndex && i != rowNonZeros)
                    {
                        val[blockOffset] = *valuePtr;
                        ++i;
                        ++colPtr;
                        ++valuePtr;
                        blockIndex = *colPtr / TVec::size();
                        blockOffset = *colPtr - (blockIndex * TVec::size());
                    }

                    rowIterator.addCol(currenTBlockkIndex, val);
                }
            }
        }

        return mat;
    }
};

template< int N, typename Real>
class EigenSparseToCompressedRowSparseMatrix< sofa::type::Vec<N, Real> > :
    public EigenSparseToCompressedRowSparseMatrixVec<sofa::type::Vec<N, Real> >
{

};

template< int N, typename Real>
class EigenSparseToCompressedRowSparseMatrix< sofa::defaulttype::RigidDeriv<N, Real> > :
    public EigenSparseToCompressedRowSparseMatrixVec<sofa::defaulttype::RigidDeriv<N, Real> >
{

};



/// Computes lhs += jacobian^T * rhs
template< typename LhsMatrixDeriv, typename RhsMatrixDeriv, typename Real >
void addMultTransposeEigen(LhsMatrixDeriv& lhs, const Eigen::SparseMatrix<Real, Eigen::RowMajor>& jacobian, const RhsMatrixDeriv& rhs)
{
    auto rhsRowIt    = rhs.begin();
    auto rhsRowItEnd = rhs.end();

    typedef typename LhsMatrixDeriv::Data LhsDeriv;
    typedef typename RhsMatrixDeriv::Data RhsDeriv;
    typedef Eigen::SparseMatrix<Real, Eigen::RowMajor> EigenSparseMatrix;
    const EigenSparseMatrix jacobianT = jacobian.transpose();

    typedef Eigen::Matrix<typename RhsDeriv::value_type, RhsDeriv::total_size, 1> EigenRhsVector;
    typedef Eigen::Matrix<typename LhsDeriv::value_type, LhsDeriv::total_size, 1> EigenLhsVector;

    // must be passed a valid iterator
    auto isEigenSparseIteratorInsideBlock = [](const typename EigenSparseMatrix::InnerIterator& it,
                                               int bBegin, int bEnd) -> bool
    {
        assert(it);
        return (it.col() >= bBegin && it.col() < bEnd);
    };


    while (rhsRowIt != rhsRowItEnd)
    {
        auto rhsColIt = rhsRowIt.begin();
        auto rhsColItEnd = rhsRowIt.end();

        if (rhsColIt != rhsColItEnd)
        {
            auto lhsRowIt = lhs.writeLine(rhsRowIt.index());
            while (rhsColIt != rhsColItEnd)
            {
                const int bColBegin = rhsColIt.index() * RhsDeriv::total_size;
                const int bColEnd   = bColBegin + RhsDeriv::total_size;

                // read jacobianT rows, block by block
                for (int k = 0; k < jacobianT.outerSize(); k+= LhsDeriv::total_size)
                {
                    // check the next LhsDeriv::total_size rows for potential non zero values
                    // inside the block [k, bCol, k+LhsDeriv::total_size, bCol+RhsDeriv::total_size]
                    bool blockEmpty = true;
                    for (Size j = 0; j < LhsDeriv::total_size; ++j)
                    {
                        typename EigenSparseMatrix::InnerIterator it(jacobianT, k+j);
                        // advance until we are either invalid or inside the block
                        while (it && 
                               !isEigenSparseIteratorInsideBlock(it,bColBegin, bColEnd) )
                        {
                            ++it;
                        }

                        if (it)
                        {
                            blockEmpty = false;
                            break;
                        }
                    }

                    if(!blockEmpty)
                    {
                        auto b = jacobianT.block(k, bColBegin, LhsDeriv::total_size, RhsDeriv::total_size);

                        LhsDeriv lhsToInsert;
                        Eigen::Map< EigenLhsVector >       lhs(lhsToInsert.ptr());
                        Eigen::Map<const EigenRhsVector >  rhs(rhsColIt.val().ptr());
                        lhs = b * rhs;

                        lhsRowIt.addCol( k / LhsDeriv::total_size, lhsToInsert);
                        
                    }

                }

                ++rhsColIt;
            }
        }

        ++rhsRowIt;
    }
}

} // namespace sofa::defaulttype
