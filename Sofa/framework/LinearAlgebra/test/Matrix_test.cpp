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

/** Sparse matrix test suite.
 *
 * The same suite is instanciated using different parameters: entry types
 * (float/Real) and BlockMN size in CompressedRowSparse.
*/

#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/FullVector.h>

#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest;

#define BENCHMARK_MATRIX_PRODUCT 0


#if BENCHMARK_MATRIX_PRODUCT
#include <ctime>
using sofa::helper::system::thread::CTime;
Real get_time() {
    CTime * timer;
    return (Real) timer->getTime();
}
#endif

namespace sofa
{

template<class TReal, sofa::Index TNbRows, sofa::Index TNbCols, sofa::Index TBlockRows, sofa::Index TBlockCols >
struct TestSparseMatricesTraits
{
    static constexpr sofa::Index NbRows = TNbRows;
    static constexpr sofa::Index NbCols = TNbCols;
    static constexpr sofa::Index BlockRows = TBlockRows;
    static constexpr sofa::Index BlockCols = TBlockCols;
    using Real = TReal;
};


/** Sparse matrix test suite.

  Creates matrices of different types, sets their entries and checks that all the matrices are the same.

  Perform matrix-vector products and compare the results.
  */
template <class T>
struct TestSparseMatrices : public NumericTest<typename T::Real>
{
    using Inherit = NumericTest<typename T::Real>;

    // Scalar type and dimensions of the matrices to test
    typedef typename T::Real Real;
    static const unsigned NROWS=T::NbRows;   // matrix size
    static const unsigned NCOLS=T::NbCols;
    static const unsigned BROWS=T::BlockRows; // block size used for matrices with block-wise storage
    static const unsigned BCOLS=T::BlockCols;


    // Dense implementation
    typedef sofa::linearalgebra::FullMatrix<Real> FullMatrix;
    typedef sofa::linearalgebra::FullVector<Real> FullVector;

    // Simple sparse matrix implemented using map< map< > >
    typedef sofa::linearalgebra::SparseMatrix<Real> MapMatrix;

    // Blockwise Compressed Sparse Row format
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<Real> CRSMatrixScalar;
    typedef sofa::type::Mat<BROWS,BCOLS,Real> BlockMN;
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<BlockMN> CRSMatrixMN;
    typedef sofa::type::Mat<BCOLS,BROWS,Real> BlockNM;
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<BlockNM> CRSMatrixNM;
    typedef sofa::type::Mat<BROWS,BROWS,Real> BlockMM;
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<BlockMM> CRSMatrixMM;
    typedef sofa::type::Mat<BCOLS,BCOLS,Real> BlockNN;
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<BlockNN> CRSMatrixNN;

    // Implementation based on Eigen
    typedef sofa::defaulttype::StdVectorTypes< sofa::type::Vec<BCOLS,Real>, sofa::type::Vec<BCOLS,Real> > InTypes;
    typedef sofa::defaulttype::StdVectorTypes< sofa::type::Vec<BROWS,Real>, sofa::type::Vec<BROWS,Real> > OutTypes;
    typedef sofa::linearalgebra::EigenSparseMatrix<InTypes,OutTypes> EigenBlockSparseMatrix;
    typedef sofa::linearalgebra::EigenBaseSparseMatrix<Real> EigenBaseSparseMatrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic> EigenDenseMatrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1> EigenDenseVec;


    // Regular Mat implementation
    typedef sofa::type::Mat<NROWS,NCOLS,Real> MatMN;
    typedef sofa::type::Mat<NCOLS,NROWS,Real> MatNM;
    typedef sofa::type::Mat<NROWS,NROWS,Real> MatMM;
    typedef sofa::type::Mat<NCOLS,NCOLS,Real> MatNN;
    typedef sofa::type::Vec<NROWS,Real> VecM;
    typedef sofa::type::Vec<NCOLS,Real> VecN;


    // The matrices used in the tests
    CRSMatrixMN crs1,crs2;
    CRSMatrixScalar crsScalar;
    FullMatrix fullMat;
    MapMatrix mapMat;
    EigenBlockSparseMatrix eiBlock1,eiBlock2,eiBlock3;
    EigenBaseSparseMatrix eiBase;
    // matrices for multiplication test
    CRSMatrixScalar crsScalarMultiplier;
    CRSMatrixScalar crsScalarMultiplication;
    CRSMatrixScalar crsScalarTransposeMultiplication;
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
    template<Size nbrows,Size nbcols>
    static void generateRandomMat( type::Mat<nbrows,nbcols,Real>& mat, bool sparse=false )
    {
        for( Size j=0; j<mat.nbCols; j++)
        {
            for( Size i=0; i<mat.nbLines; i++)
            {
                Real random = Real(helper::drand(1));
                if( sparse && random > -0.5 && random < 0.5 ) random = 0;
                mat(i,j)=random;
            }
        }
    }

    /// filling a BaseMatrix from a given Mat
    template<Size nbrows,Size nbcols>
    static void copyFromMat( linearalgebra::BaseMatrix& baseMat, const type::Mat<nbrows,nbcols,Real>& mat )
    {
        baseMat.resize(mat.nbLines,mat.nbCols);

        for( Size j=0; j<mat.nbCols; j++)
        {
            for( Size i=0; i<mat.nbLines; i++)
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
        copyFromMat(crsScalar, mat);
        copyFromMat( fullMat, mat );
        copyFromMat( mapMat, mat );
        copyFromMat( eiBlock1, mat );
        copyFromMat( eiBlock2, mat );
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
        copyFromMat(crsScalarMultiplier, matMultiplier);
        copyFromMat( crsMultiplier, matMultiplier );
        copyFromMat( fullMultiplier, matMultiplier );
        copyFromMat( eiBaseMultiplier, matMultiplier );
        eiDenseMultiplier = Eigen::Map< EigenDenseMatrix >( &(matMultiplier.transposed())[0][0], NCOLS, NROWS ); // need to transpose because EigenDenseMatrix is ColMajor


        matMultiplication = mat * matMultiplier;
        crsScalar.mul(crsScalarMultiplication, crsScalarMultiplier);
        crs1.mul( crsMultiplication, crsMultiplier );
        fullMat.mul( fullMultiplication, fullMultiplier );
        eiBase.mul_MT( eiBaseMultiplication, eiBaseMultiplier ); // sparse x sparse
        eiBase.mul_MT( eiDenseMultiplication, eiDenseMultiplier ); // sparse x dense

        matTransposeMultiplication = mat.multTranspose( mat );
        crsScalar.mulTranspose(crsScalarTransposeMultiplication, crsScalar);
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
                Real valij = i*NCOLS+j;
                a.insertBack(i,j,valij);
                if( i==j )
                    b.add(i,j,valij);
            }
        }
        a.compress();
        b.compress();

        // second pass for b. Some values are set with the right value, some with the wrong value, some are not set
        for( unsigned j=0; j<NCOLS; j++)
        {
            for( unsigned i=0; i<NROWS; i++)
            {
                Real valij = i*NCOLS+j;
                if( i!=j )
                    b.add(i,j,valij);
            }
        }
        b.compress();
        return Inherit::matrixMaxDiff(a,b) < 100 * Inherit::epsilon();
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

        ASSERT_TRUE( Inherit::matrixMaxDiff(ma,mb) < 100*Inherit::epsilon() );

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

        ASSERT_TRUE( Inherit::matrixMaxDiff(ma,mb) < 100*Inherit::epsilon() );





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

        ASSERT_TRUE( Inherit::matrixMaxDiff(ma,mb) < 100*Inherit::epsilon() );



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

        ASSERT_TRUE( Inherit::matrixMaxDiff(ma,mb) < 100*Inherit::epsilon() );
    }

    bool checkEigenMatrixBlockFromCompressedRowSparseMatrix()
    {
        return Inherit::matrixMaxDiff(crs1,eiBlock3) < 100*Inherit::epsilon();
    }

    bool checkEigenDenseMatrix()
    {
        if( matMultiplier.nbCols != eiDenseMultiplier.cols() || matMultiplier.nbLines != eiDenseMultiplier.rows() ) return false;
        for( sofa::Index j=0; j<matMultiplier.nbCols; j++)
        {
            for( sofa::Index i=0; i<matMultiplier.nbLines; i++)
            {
                if( matMultiplier(i,j) != eiDenseMultiplier(i,j) ) return false;
            }
        }
        return true;
    }
};

using TestSparseMatricesImplementations = ::testing::Types<
    TestSparseMatricesTraits<SReal, 4, 8, 4, 8>,
    TestSparseMatricesTraits<SReal, 4, 8, 4, 2>,
    TestSparseMatricesTraits<SReal, 4, 8, 1, 8>,
    TestSparseMatricesTraits<SReal, 4, 8, 2, 8>,
    TestSparseMatricesTraits<SReal, 4, 8, 2, 4>,
    TestSparseMatricesTraits<SReal, 24, 24, 3, 3>
>;

TYPED_TEST_SUITE(TestSparseMatrices, TestSparseMatricesImplementations);

// ==============================
// Set/get value tests
TYPED_TEST(TestSparseMatrices, set_fullMat ) { ASSERT_TRUE( this->matrixMaxDiff(this->mat,this->fullMat) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, set_crs_scalar ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullMat,this->crsScalar ) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, set_crs1 ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullMat,this->crs1) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, set_crs2 ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullMat,this->crs2) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, set_mapMat ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullMat,this->mapMat) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, set_eiBlock1 ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullMat,this->eiBlock1) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, set_eiBlock2 ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullMat,this->eiBlock2) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, set_eiBase ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullMat,this->eiBase) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, eigenMatrix_update ) { ASSERT_TRUE( this->checkEigenMatrixUpdate() ); }
TYPED_TEST(TestSparseMatrices, eigenMatrix_block_row_filling ) { this->checkEigenMatrixBlockRowFilling(); }
TYPED_TEST(TestSparseMatrices, eigenMatrixBlockFromCompressedRowSparseMatrix ) { ASSERT_TRUE( this->checkEigenMatrixBlockFromCompressedRowSparseMatrix() ); }
TYPED_TEST(TestSparseMatrices, eigenMapToDenseMatrix ) { ASSERT_TRUE( this->checkEigenDenseMatrix() ); }

// ==============================
// Matrix-Vector product tests
TYPED_TEST(TestSparseMatrices, set_fullVec_nrows_reference )
{
    ASSERT_TRUE(this->vectorMaxDiff(this->vecM,this->fullVec_nrows_reference) < this->epsilon() );
}
TYPED_TEST(TestSparseMatrices, fullMat_vector_product )
{
    //    fullMat.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    this->fullVec_nrows_result = this->fullMat * this->fullVec_ncols;
    ASSERT_TRUE(this->vectorMaxDiff(this->fullVec_nrows_reference,this->fullVec_nrows_result) < this->epsilon() );
}
TYPED_TEST(TestSparseMatrices, mapMat_vector_product )
{
    //    mapMat.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    this->fullVec_nrows_result = this->mapMat * this->fullVec_ncols;
    ASSERT_TRUE(this->vectorMaxDiff(this->fullVec_nrows_reference,this->fullVec_nrows_result) < this->epsilon() );
}
TYPED_TEST(TestSparseMatrices, eiBlock1_vector_product )
{
    //    eiBlock1.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    //    eiBlock1.multVector(fullVec_nrows_result,fullVec_ncols);
    this->fullVec_nrows_result = this->eiBlock1 * this->fullVec_ncols;
    ASSERT_TRUE(this->vectorMaxDiff(this->fullVec_nrows_reference,this->fullVec_nrows_result) < this->epsilon() );

}
TYPED_TEST(TestSparseMatrices, crs1_vector_product )
{
    //    EXPECT_TRUE(NROWS%BROWS==0 && NCOLS%BCOLS==0) << "Error: CompressedRowSparseMatrix * Vector crashes when the size of the matrix is not a multiple of the size of the matrix blocks. Aborting this test, and reporting a failure."; // otherwise the product crashes
    //    crs1.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    this->fullVec_nrows_result = this->crs1 * this->fullVec_ncols;
    ASSERT_TRUE(this->vectorMaxDiff(this->fullVec_nrows_reference,this->fullVec_nrows_result) < this->epsilon() );
}


// ==============================
// Matrix product tests
TYPED_TEST(TestSparseMatrices, full_matrix_product ) { ASSERT_TRUE( this->matrixMaxDiff(this->matMultiplication,this->fullMultiplication) < 100*this->epsilon() );  }
TYPED_TEST(TestSparseMatrices, crs_scalar_matrix_product ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullMultiplication,this->crsScalarMultiplication) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, crs_matrix_product ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullMultiplication,this->crsMultiplication) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, EigenBase_matrix_product ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullMultiplication,this->eiBaseMultiplication) < 100*this->epsilon() ); }
//TYPED_TEST(TestSparseMatrices, EigenSparseDense_matrix_product ) { ASSERT_TRUE( EigenDenseMatrix(this->eiBaseMultiplication.compressedMatrix) == this->eiDenseMultiplication ); }
TYPED_TEST(TestSparseMatrices, full_matrix_transposeproduct ) { ASSERT_TRUE( this->matrixMaxDiff(this->matTransposeMultiplication,this->fullTransposeMultiplication) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, crs_scalar_matrix_transposeproduct ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullTransposeMultiplication,this->crsScalarTransposeMultiplication) < 100*this->epsilon() ); }
TYPED_TEST(TestSparseMatrices, crs_matrix_transposeproduct ) { ASSERT_TRUE( this->matrixMaxDiff(this->fullTransposeMultiplication,this->crsTransposeMultiplication) < 100*this->epsilon() ); }

// Matrix addition
TYPED_TEST(TestSparseMatrices, crs_matrix_addition )
{
    this->crs2 = this->crs1 + this->crs1;
    ASSERT_TRUE( this->matrixMaxDiff(this->mat*2,this->crs2) < 100*this->epsilon() );

    this->crs2 += this->crs1;
    ASSERT_TRUE( this->matrixMaxDiff(this->mat*3,this->crs2) < 100*this->epsilon() );

    this->crs2 -= this->crs1;
    ASSERT_TRUE( this->matrixMaxDiff(this->mat*2,this->crs2) < 100*this->epsilon() );
}

#if BENCHMARK_MATRIX_PRODUCT
///// product timing
typedef TestSparseMatrices<Real,360,300,3,3> TsProductTimings;
TEST_F(TsProductTimings, benchmark )
{
    msg_info()<<"=== Matrix-Matrix Products:"<<std::endl;

    Real start, stop;

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

        Real min=std::numeric_limits<Real>::max(), max=0, sum=0;
        for( int i=0; i<100 ; ++i )
        {
            start = get_time();
            res.noalias() = A * rhs;
            stop = get_time();
            Real current = stop-start;
            sum+=current;
            if( current<min ) min=current;
            if( current>max ) max=current;
        }

        msg_info()<<"ST: "<<sum/100.0<<" "<<min<<" "<<max<<std::endl;



    #ifdef _OPENMP
        min=std::numeric_limits<Real>::max(), max=0, sum=0;
        for( int i=0; i<100 ; ++i )
        {
            start = get_time();
//            res.noalias() = typename Eigen::SparseDenseProductReturnType_MT<Eigen::SparseMatrix<SReal,Eigen::RowMajor>,Eigen::Matrix<SReal, Eigen::Dynamic, 1> >::Type( A.derived(), rhs.derived() );
//            component::linearsolver::mul_EigenSparseDenseMatrix_MT( res, A, rhs );
            res.noalias() = component::linearsolver::mul_EigenSparseDenseMatrix_MT( A, rhs );
            stop = get_time();
            Real current = stop-start;
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

}// namespace sofa
