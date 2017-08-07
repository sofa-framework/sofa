#include "MapMapSparseMatrix.h"
#include <Eigen/Sparse>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <cassert>
#include <type_traits>
#include <cstdlib>

namespace sofa
{
namespace defaulttype
{


template< class TBloc >
struct MapMapSparseMatrixToEigenSparse
{

};

template <typename TVec, typename Real>
struct MapMapSparseMatrixToEigenSparseVec 
{
    typedef MapMapSparseMatrix< TVec >                 TMapMapSparseMatrix;
    typedef Eigen::SparseMatrix<Real, Eigen::RowMajor> EigenSparseMatrix;


    EigenSparseMatrix operator() (const TMapMapSparseMatrix& mat, std::size_t size)
    {
        std::size_t eigenMatSize = size * TVec::size();
        EigenSparseMatrix eigenMat(eigenMatSize, eigenMatSize);

        std::vector<Eigen::Triplet<Real> > triplets;

        for (auto row = mat.begin(); row != mat.end(); ++row)
        {
            for (auto col = row.begin(), colend = row.end(); col !=colend; ++col)
            {
                const TVec& vec = col.val();
                int   colIndex  = col.index() * TVec::size();

                for (std::size_t i = 0; i < TVec::size(); ++i)
                {
                    triplets.emplace_back(Eigen::Triplet<Real>( row.index(), colIndex + i, vec[i]) );
                }

            }
        }

        eigenMat.setFromTriplets(triplets.begin(), triplets.end());;
        eigenMat.makeCompressed();

        return eigenMat;
    }

};

template< int N, typename Real >
class MapMapSparseMatrixToEigenSparse< sofa::defaulttype::Vec<N,Real> >
    : public  MapMapSparseMatrixToEigenSparseVec< sofa::defaulttype::Vec<N, Real>, Real >
{

};

template< int N, typename Real >
class MapMapSparseMatrixToEigenSparse< sofa::defaulttype::RigidDeriv<N, Real > >
    : public MapMapSparseMatrixToEigenSparseVec< sofa::defaulttype::RigidDeriv<N, Real>, Real >
{

};


template< class TBloc >
struct EigenSparseToMapMapSparseMatrix
{

};


template <typename TVec, typename Real>
struct EigenSparseToMapMapSparseMatrixVec
{
    typedef MapMapSparseMatrix< TVec >                 TMapMapSparseMatrix;
    typedef Eigen::SparseMatrix<Real, Eigen::RowMajor> EigenSparseMatrix;


    TMapMapSparseMatrix operator() (const EigenSparseMatrix& eigenMat)
    {
        TMapMapSparseMatrix mat;

        const int* outerIndexPtr  = eigenMat.outerIndexPtr();
        const int* innerIndexPtr  = eigenMat.innerIndexPtr();
        const Real* valuePtr      = eigenMat.valuePtr();

        for (int rowIndex = 0; rowIndex < eigenMat.outerSize(); ++rowIndex)
        {
            int offset      = *(outerIndexPtr + rowIndex);
            int rowNonZeros = *(outerIndexPtr + rowIndex + 1) - *(outerIndexPtr + rowIndex);

            if (rowNonZeros != 0)
            {
                auto rowIterator = mat.writeLine(rowIndex);

                int i = 0;
                const int*  colPtr = innerIndexPtr + offset;
                const Real* valPtr = valuePtr + offset;
                int   blockIndex   = *colPtr / TVec::size();
                int   blockOffset  = *colPtr - (blockIndex * TVec::size());


                while (i != rowNonZeros)
                {
                    TVec val;
                    int currentBlockIndex = blockIndex;
                    int currentCol   = *colPtr;
                    while (currentBlockIndex == blockIndex && i != rowNonZeros)
                    {
                        val[blockOffset] = *valuePtr;
                        ++i;
                        ++colPtr;
                        ++valuePtr;
                        blockIndex = *colPtr / TVec::size();
                        blockOffset = *colPtr - (blockIndex * TVec::size());
                    }

                    rowIterator.addCol(currentBlockIndex, val);
                }
            }
        }

        return mat;
    }
};

template< int N, typename Real>
class EigenSparseToMapMapSparseMatrix< sofa::defaulttype::Vec<N, Real> > :
    public EigenSparseToMapMapSparseMatrixVec<sofa::defaulttype::Vec<N, Real>, Real>
{

};

template< int N, typename Real>
class EigenSparseToMapMapSparseMatrix< sofa::defaulttype::RigidDeriv<N, Real> > :
    public EigenSparseToMapMapSparseMatrixVec<sofa::defaulttype::RigidDeriv<N, Real>, Real>
{

};



/// Computes lhs += jacobian^T * rhs
template< typename LhsDeriv, typename RhsDeriv, typename Real >
void addMultTransposeEigen(MapMapSparseMatrix<LhsDeriv>& lhs, const Eigen::SparseMatrix<Real, Eigen::RowMajor>& jacobian, const MapMapSparseMatrix<RhsDeriv>& rhs)
{
    auto rhsRowIt    = rhs.begin();
    auto rhsRowItEnd = rhs.end();

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
                    for (int j = 0; j < LhsDeriv::total_size; ++j)
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

}

}