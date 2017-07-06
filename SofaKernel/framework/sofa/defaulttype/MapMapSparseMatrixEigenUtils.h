#include "MapMapSparseMatrix.h"
#include <Eigen/Sparse>
#include <sofa/defaulttype/Vec.h>
#include <cassert>

namespace sofa
{
namespace defaulttype
{


template< class TBloc >
struct MapMapSparseMatrixToEigenSparse
{

};


template <int N, typename Real>
struct MapMapSparseMatrixToEigenSparse < sofa::defaulttype::Vec<N, Real> >
{

    typedef typename sofa::defaulttype::Vec<N, Real>   TVec;
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
                int   colIndex  = col.index() * N;

                for (std::size_t i = 0; i < vec.size(); ++i)
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


template< class TBloc >
struct EigenSparseToMapMapSparseMatrix
{

};

template <int N, typename Real >
struct EigenSparseToMapMapSparseMatrix< sofa::defaulttype::Vec<N, Real> >
{
    typedef typename sofa::defaulttype::Vec<N, Real>   TVec;
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

                assert(rowNonZeros % TVec::size() == 0); // the filling must be compatible with the bloc size

                for (int i=0;i<rowNonZeros; i+=TVec::size() )
                {
                    int colIndex         = *(innerIndexPtr + offset + i);
                    const double* valPtr = (valuePtr + offset + i);
                    rowIterator.setCol(colIndex, TVec(valPtr));
                }
            }
        }

        return mat;
    }
};



}

}