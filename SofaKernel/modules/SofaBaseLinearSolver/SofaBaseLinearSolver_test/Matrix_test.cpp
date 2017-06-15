/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

/** Sparse matrix test suite.
 *
 * The same suite is instanciated using different parameters: entry types
 * (float/double) and BlockMN size in CompressedRowSparse.
*/

#include <SofaTest/Sofa_test.h>

#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/FullVector.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

#include <gtest/gtest.h>


#define BENCHMARK_MATRIX_PRODUCT 0


#if BENCHMARK_MATRIX_PRODUCT
#include <ctime>
using sofa::helper::system::thread::CTime;
double get_time() {
    CTime * timer;
    return (double) timer->getTime();
}
#endif


namespace sofa {


/** Sparse matrix test suite.

  Creates matrices of different types, sets their entries and checks that all the matrices are the same.

  Perform matrix-vector products and compare the results.
  */
template <typename _Real, unsigned NumRows, unsigned NumCols, unsigned BlockRows, unsigned BlockCols>
struct TestSparseMatrices : public Sofa_test<_Real>
{
    // Scalar type and dimensions of the matrices to test
    typedef _Real Real;
    static const unsigned NROWS=NumRows;   // matrix size
    static const unsigned NCOLS=NumCols;
    static const unsigned BROWS=BlockRows; // block size used for matrices with block-wise storage
    static const unsigned BCOLS=BlockCols;


    // Dense implementation
    typedef sofa::component::linearsolver::FullMatrix<Real> FullMatrix;
    typedef sofa::component::linearsolver::FullVector<Real> FullVector;

    // Simple sparse matrix implemented using map< map< > >
    typedef sofa::component::linearsolver::SparseMatrix<Real> MapMatrix;

    // Blockwise Compressed Sparse Row format
    typedef sofa::defaulttype::Mat<BROWS,BCOLS,Real> BlockMN;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<BlockMN> CRSMatrixMN;
    typedef sofa::defaulttype::Mat<BCOLS,BROWS,Real> BlockNM;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<BlockNM> CRSMatrixNM;
    typedef sofa::defaulttype::Mat<BROWS,BROWS,Real> BlockMM;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<BlockMM> CRSMatrixMM;
    typedef sofa::defaulttype::Mat<BCOLS,BCOLS,Real> BlockNN;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<BlockNN> CRSMatrixNN;

    // Implementation based on Eigen
    typedef sofa::defaulttype::StdVectorTypes< sofa::defaulttype::Vec<BCOLS,Real>, sofa::defaulttype::Vec<BCOLS,Real> > InTypes;
    typedef sofa::defaulttype::StdVectorTypes< sofa::defaulttype::Vec<BROWS,Real>, sofa::defaulttype::Vec<BROWS,Real> > OutTypes;
    typedef sofa::component::linearsolver::EigenSparseMatrix<InTypes,OutTypes> EigenBlockSparseMatrix;
    typedef sofa::component::linearsolver::EigenBaseSparseMatrix<Real> EigenBaseSparseMatrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic> EigenDenseMatrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1> EigenDenseVec;


    // Regular Mat implementation
    typedef sofa::defaulttype::Mat<NROWS,NCOLS,Real> MatMN;
    typedef sofa::defaulttype::Mat<NCOLS,NROWS,Real> MatNM;
    typedef sofa::defaulttype::Mat<NROWS,NROWS,Real> MatMM;
    typedef sofa::defaulttype::Mat<NCOLS,NCOLS,Real> MatNN;
    typedef sofa::defaulttype::Vec<NROWS,Real> VecM;
    typedef sofa::defaulttype::Vec<NCOLS,Real> VecN;


    // The matrices used in the tests
    CRSMatrixMN crs1,crs2;
    FullMatrix fullMat;
    MapMatrix mapMat;
    EigenBlockSparseMatrix eiBlock1,eiBlock2,eiBlock3;
    EigenBaseSparseMatrix eiBase;
    // matrices for multiplication test
    CRSMatrixNM crsMultiplier;
    CRSMatrixMM crsMultiplication;
    CRSMatrixNN crsTransposeMultiplication;
    FullMatrix fullMultiplier;
    FullMatrix fullMultiplication;
    FullMatrix fullTransposeMultiplication;
    MatMN mat;
    MatNM matMultiplier;
    MatMM matMultiplication;
    MatNN matTransposeMultiplication;
    EigenBaseSparseMatrix eiBaseMultiplier;
    EigenBaseSparseMatrix eiBaseMultiplication;
    EigenDenseMatrix eiDenseMultiplier;
    EigenDenseMatrix eiDenseMultiplication;


    // The vectors used in the tests
    FullVector fullVec_ncols;
    FullVector fullVec_nrows_reference,fullVec_nrows_result;
    VecM vecM;
    VecN vecN;
    EigenDenseVec eiVecM, eiVecN;


    /// generating a random Mat
    /// if sparse=0 a lot a null entries will be created
    template<int nbrows,int nbcols>
    static void generateRandomMat( defaulttype::Mat<nbrows,nbcols,Real>& mat, bool sparse=false )
    {
        for( int j=0; j<mat.nbCols; j++)
        {
            for( int i=0; i<mat.nbLines; i++)
            {
                Real random = Real(helper::drand(1));
                if( sparse && random > -0.5 && random < 0.5 ) random = 0;
                mat(i,j)=random;
            }
        }
    }

    /// filling a BaseMatrix from a given Mat
    template<int nbrows,int nbcols>
    static void copyFromMat( defaulttype::BaseMatrix& baseMat, const defaulttype::Mat<nbrows,nbcols,Real>& mat )
    {
        baseMat.resize(mat.nbLines,mat.nbCols);

        for( int j=0; j<mat.nbCols; j++)
        {
            for( int i=0; i<mat.nbLines; i++)
            {
                if( !baseMat.isSparse() || mat(i,j)!=0 ) baseMat.add( i, j, mat(i,j) );
            }
        }

        baseMat.compress();
    }

    /// Create the context for the matrix tests.
    TestSparseMatrices()
    {
        //std::cout<<"Matrix_test "<<NumRows<<" "<<NumCols<<" "<<BlockRows<<" "<<BlockCols<<std::endl << "seed number = " << BaseSofa_test::seed<<std::endl;

        // resize and fill the matrices
        generateRandomMat( mat, true );
        copyFromMat( crs1, mat );
        copyFromMat( crs2, mat );
        copyFromMat( fullMat, mat );
        copyFromMat( mapMat, mat );
        copyFromMat( eiBlock1, mat );
        copyFromMat( eiBlock2, mat );
//        copyFromMat( eiBlock3, mat );
        copyFromMat( eiBase, mat );
        eiBlock3.copyFrom(crs1);


        // resize and fill the vectors
        fullVec_ncols.resize(NCOLS);
        fullVec_nrows_reference.resize(NROWS);
        fullVec_nrows_result.resize(NROWS);
        eiVecM.resize(NROWS);
        eiVecN.resize(NCOLS);
        for( unsigned i=0; i<NCOLS; i++)
        {
            fullVec_ncols[i] = (Real)i;
            vecN[i] = (Real)i;
            eiVecN[i] = (Real)i;
        }
        fullMat.mul(fullVec_nrows_reference,fullVec_ncols); //    cerr<<"MatrixTest: vref = " << vref << endl;

        vecM = mat * vecN;




        // matrix multiplication

        generateRandomMat( matMultiplier, true );
        copyFromMat( crsMultiplier, matMultiplier );
        copyFromMat( fullMultiplier, matMultiplier );
        copyFromMat( eiBaseMultiplier, matMultiplier );
        eiDenseMultiplier = Eigen::Map< EigenDenseMatrix >( &(matMultiplier.transposed())[0][0], NCOLS, NROWS ); // need to transpose because EigenDenseMatrix is ColMajor


        matMultiplication = mat * matMultiplier;
        crs1.mul( crsMultiplication, crsMultiplier );
        fullMat.mul( fullMultiplication, fullMultiplier );
        eiBase.mul_MT( eiBaseMultiplication, eiBaseMultiplier ); // sparse x sparse
        eiBase.mul_MT( eiDenseMultiplication, eiDenseMultiplier ); // sparse x dense

        matTransposeMultiplication = mat.multTranspose( mat );
        crs1.mulTranspose( crsTransposeMultiplication, crs1 );
        fullMat.mulT( fullTransposeMultiplication, fullMat );

    }


    /** Check that EigenMatrix update works as well as direct init. Return true if the test succeeds.*/
    bool checkEigenMatrixUpdate()
    {
        // fill two matrices with the same values, one directly (in order), one in two passes, then compare their values
        EigenBlockSparseMatrix a,b;
        a.resize(NROWS,NCOLS);
        b.resize(NROWS,NCOLS);
        for( unsigned i=0; i<NROWS; i++)
        {
            a.beginRow(i);
            for( unsigned j=0; j<NCOLS; j++)
            {
                double valij = i*NCOLS+j;
                a.insertBack(i,j,valij);
                if( i==j )
                    b.add(i,j,valij);
            }
        }
        a.compress();
        b.compress();

        //    cerr<<"MatrixTest<Real,RN,CN>::checkEigenMatrixUpdate, a = " << a << endl;
        //    cerr<<"MatrixTest<Real,RN,CN>::checkEigenMatrixUpdate, b incomplete = " << b << endl;

        // second pass for b. Some values are set with the right value, some with the wrong value, some are not set
        for( unsigned j=0; j<NCOLS; j++)
        {
            for( unsigned i=0; i<NROWS; i++)
            {
                double valij = i*NCOLS+j;
                if( i!=j )
                    b.add(i,j,valij);
            }
        }
        b.compress();
        //    cerr<<"MatrixTest<Real,RN,CN>::checkEigenMatrixUpdate, b complete = " << b << endl;
        return Sofa_test<_Real>::matrixMaxDiff(a,b) < 100 * Sofa_test<_Real>::epsilon();
    }

    /** Check the filling of EigenMatrix per rows of blocks. */
    void checkEigenMatrixBlockRowFilling()
    {
        EigenBlockSparseMatrix mb;
        FullMatrix ma;
        unsigned br=3, bc=3;
        ma.resize(br*BROWS,bc*BCOLS);




        // building with unordered blocks

        mb.resizeBlocks(br,bc);
        for( unsigned i=0; i<br; i++ )
        {
            mb.beginBlockRow(i);
            if( i%2==0 ) // leave some rows empty
            {
                for( int j=bc-1; j>=0; j--) // set the blocs in reverse order, for fun.
                {
                    // create a block and give it some value
                    BlockMN b;
                    for( unsigned k=0; k<BROWS && k<BCOLS; k++ ){
                        b[k][k] = (Real)i+j;
                        ma.set(i*BROWS+k, j*BCOLS+k, i+j);
                    }

                    // insert the block in the matrix
                    mb.createBlock(j,b);
                }
            }
            mb.endBlockRow();
        }
        mb.compress();


        //    serr()<<"MatrixTest<Real,RN,CN>::checkEigenMatrixBlockRowFilling, ma = " << ma << endl;
        //    serr()<<"MatrixTest<Real,RN,CN>::checkEigenMatrixBlockRowFilling, mb = " << mb << endl;
        ASSERT_TRUE( Sofa_test<_Real>::matrixMaxDiff(ma,mb) < 100*Sofa_test<_Real>::epsilon() );





        // building with ordered blocks

        mb.resizeBlocks(br,bc);
        for( unsigned i=0; i<br; i++ )
        {
            mb.beginBlockRow(i);
            if( i%2==0 ) // leave some rows empty
            {
                for( unsigned j=0 ; j<bc; ++j ) // set the blocs in column order
                {
                    // create a block and give it some value
                    BlockMN b;
                    for( unsigned k=0; k<BROWS && k<BCOLS; k++ ){
                        b[k][k] = (Real)i+j;
                    }

                    // insert the block in the matrix
                    mb.createBlock(j,b);
                }
            }
            mb.endSortedBlockRow();
        }
        mb.compress();

        ASSERT_TRUE( Sofa_test<_Real>::matrixMaxDiff(ma,mb) < 100*Sofa_test<_Real>::epsilon() );





        // building with scheduled block additions

        mb.resizeBlocks(br,bc);
        for( unsigned i=0; i<br; i++ )
        {
            if( i%2==0 ) // leave some rows empty
            {
                for( int j=bc-1; j>=0; j--) // set the blocs in reverse order, for fun.
                {
                    // create a block and give it some value
                    BlockMN b;
                    for( unsigned k=0; k<BROWS && k<BCOLS; k++ ){
                        b[k][k] = (Real)i+j;
                    }

                    // insert the block in the matrix
                    mb.addBlock(i,j,b);
                }
            }
        }
        mb.compress();

        ASSERT_TRUE( Sofa_test<_Real>::matrixMaxDiff(ma,mb) < 100*Sofa_test<_Real>::epsilon() );



        // building with one block per row
        ma.clear();
        mb.resizeBlocks(br,bc);
        for( unsigned i=0; i<br; i++ )
        {
            if( i%2==0 ) // leave some rows empty
            {
                // create a block and give it some value
                BlockMN b;
                for( unsigned k=0; k<BROWS && k<BCOLS; k++ ){
                    b[k][k] = (Real)i;
                    ma.set(i*BROWS+k, i*BCOLS+k, i);
                }

                // insert the block in the matrix
                mb.insertBackBlock(i,i,b);
            }
            else
            {
                // empty lines
                mb.beginBlockRow(i);
                mb.endSortedBlockRow();
            }

        }
        mb.compress();

        ASSERT_TRUE( Sofa_test<_Real>::matrixMaxDiff(ma,mb) < 100*Sofa_test<_Real>::epsilon() );
    }

    bool checkEigenMatrixBlockFromCompressedRowSparseMatrix()
    {
//        if( !matricesAreEqual(crs1,eiBlock3)){
//            cout<<"heckEigenMatrixBlockFromCompressedRowSparseMatrix, crs1 = " << crs1 << endl;
//            cout<<"heckEigenMatrixBlockFromCompressedRowSparseMatrix, eiBlock3 = " << eiBlock3 << endl;
//        }
        return Sofa_test<_Real>::matrixMaxDiff(crs1,eiBlock3) < 100*Sofa_test<_Real>::epsilon();
    }

    bool checkEigenDenseMatrix()
    {
        if( matMultiplier.nbCols != eiDenseMultiplier.cols() || matMultiplier.nbLines != eiDenseMultiplier.rows() ) return false;
        for( int j=0; j<matMultiplier.nbCols; j++)
        {
            for( int i=0; i<matMultiplier.nbLines; i++)
            {
                if( matMultiplier(i,j) != eiDenseMultiplier(i,j) ) return false;
            }
        }
        return true;
    }
};

#ifndef SOFA_FLOAT
///////////////////
// double precision
///////////////////
// trivial blocs
typedef TestSparseMatrices<double,4,8,4,8> Ts4848;
#define TestMatrix Ts4848
#include "Matrix_test.inl"
#undef TestMatrix

//// semi-trivial blocs
typedef TestSparseMatrices<double,4,8,4,2> Ts4842;
#define TestMatrix Ts4842
#include "Matrix_test.inl"
#undef TestMatrix

typedef TestSparseMatrices<double,4,8,1,8> Ts4818;
#define TestMatrix Ts4818
#include "Matrix_test.inl"
#undef TestMatrix

// well-fitted blocs
typedef TestSparseMatrices<double,4,8,2,2> Ts4822;
#define TestMatrix Ts4822
#include "Matrix_test.inl"
#undef TestMatrix

/// not fitted blocs
//typedef TestSparseMatrices<double,4,8,2,3> Ts4823;
//#define TestMatrix Ts4823
//#include "Matrix_test.inl"
//#undef TestMatrix


#if BENCHMARK_MATRIX_PRODUCT
///// product timing
typedef TestSparseMatrices<double,360,300,3,3> TsProductTimings;
TEST_F(TsProductTimings, benchmark )
{
    msg_info()<<"=== Matrix-Matrix Products:"<<std::endl;

    double start, stop;

    matMultiplication.clear();
    start = get_time();
    matMultiplication = mat * matMultiplier;
    stop = get_time();
    msg_info()<<"Mat:\t\t"<<stop-start<<" (ms)"<<std::endl;

    fullMultiplication.clear();
    start = get_time();
    fullMat.mul( fullMultiplication, fullMultiplier );
    stop = get_time();
    msg_info()<<"Full:\t\t"<<stop-start<<" (ms)"<<std::endl;

    crsMultiplication.clear();
    start = get_time();
    crs1.mul( crsMultiplication, crsMultiplier );
    stop = get_time();
    msg_info()<<"CRS:\t\t"<<stop-start<<" (ms)"<<std::endl;

    eiBaseMultiplication.clear();
    start = get_time();
    eiBase.mul( eiBaseMultiplication, eiBaseMultiplier );
    stop = get_time();
    msg_info()<<"Eigen Base ST:\t\t"<<stop-start<<" (ms)"<<std::endl;

#ifdef _OPENMP
    eiBaseMultiplication.clear();
    start = get_time();
    eiBase.mul_MT( eiBaseMultiplication, eiBaseMultiplier );
    stop = get_time();
    msg_info()<<"Eigen Base MT:\t\t"<<stop-start<<" (ms)"<<std::endl;
#endif

    start = get_time();
    eiDenseMultiplication = eiBase.compressedMatrix * eiDenseMultiplier;
    stop = get_time();
    msg_info()<<"Eigen Sparse*Dense:\t\t"<<stop-start<<" (ms)"<<std::endl;

#ifdef _OPENMP
    start = get_time();
    eiDenseMultiplication.noalias() = component::linearsolver::mul_EigenSparseDenseMatrix_MT( eiBase.compressedMatrix, eiDenseMultiplier, omp_get_max_threads()/2 );
    stop = get_time();
    msg_info()<<"Eigen Sparse*Dense MT:\t\t"<<stop-start<<" (ms)"<<std::endl;
#endif

    msg_info()<<"=== Eigen Matrix-Vector Products:"<<std::endl;
    unsigned nbrows = 100, nbcols;
    msg_info()<<"=== nb rows:"<<nbrows<<std::endl;


    for( int j=1; j<300 ; j+=30 )
    {
        nbcols = 100 * j;

        msg_info()<<"=== nb cols:"<<nbcols<<std::endl;

        Eigen::SparseMatrix<SReal,Eigen::RowMajor> A;
        A.resize(nbrows,nbcols);
#define NBCOLSRHS 1
        Eigen::Matrix<SReal, Eigen::Dynamic, NBCOLSRHS> res, rhs;
        rhs.resize(nbcols,NBCOLSRHS);
        res.resize(nbrows,NBCOLSRHS);

        for( unsigned j=0; j<nbcols; j++)
        {
            Real random = Real(helper::drand(1));
            for( unsigned i=0; i<NBCOLSRHS; i++)
                rhs.coeffRef(j,i) = random;
            for( unsigned i=0; i<nbrows; i++)
            {
                if( random > -0.5 && random < 0.5 ) A.coeffRef(i,j)=random;
            }
        }

        double min=std::numeric_limits<double>::max(), max=0, sum=0;
        for( int i=0; i<100 ; ++i )
        {
            start = get_time();
            res.noalias() = A * rhs;
            stop = get_time();
            double current = stop-start;
            sum+=current;
            if( current<min ) min=current;
            if( current>max ) max=current;
        }

        msg_info()<<"ST: "<<sum/100.0<<" "<<min<<" "<<max<<std::endl;



    #ifdef _OPENMP
        min=std::numeric_limits<double>::max(), max=0, sum=0;
        for( int i=0; i<100 ; ++i )
        {
            start = get_time();
//            res.noalias() = typename Eigen::SparseDenseProductReturnType_MT<Eigen::SparseMatrix<SReal,Eigen::RowMajor>,Eigen::Matrix<SReal, Eigen::Dynamic, 1> >::Type( A.derived(), rhs.derived() );
//            component::linearsolver::mul_EigenSparseDenseMatrix_MT( res, A, rhs );
            res.noalias() = component::linearsolver::mul_EigenSparseDenseMatrix_MT( A, rhs );
            stop = get_time();
            double current = stop-start;
            sum+=current;
            if( current<min ) min=current;
            if( current>max ) max=current;
        }
        msg_info()<<"MT: "<<sum/100.0<<" "<<min<<" "<<max<<std::endl;
    #endif
    }



    ASSERT_TRUE( true );
}

#endif

#endif

#ifndef SOFA_DOUBLE
///////////////////
// simple precision
// The macro EIGEN_DONT_ALIGN is needed for float on windows
///////////////////
// trivial blocs
typedef TestSparseMatrices<float,4,8,4,8> Ts4848f;
#define TestMatrix Ts4848f
#include "Matrix_test.inl"
#undef TestMatrix

// semi-trivial blocs
typedef TestSparseMatrices<float,4,8,4,2> Ts4842f;
#define TestMatrix Ts4842f
#include "Matrix_test.inl"
#undef TestMatrix

typedef TestSparseMatrices<float,4,8,1,8> Ts4818f;
#define TestMatrix Ts4818f
#include "Matrix_test.inl"
#undef TestMatrix

/// well-fitted blocs
typedef TestSparseMatrices<float,4,8,2,2> Ts4822f;
#define TestMatrix Ts4822f
#include "Matrix_test.inl"
#undef TestMatrix

#endif

}// namespace sofa
