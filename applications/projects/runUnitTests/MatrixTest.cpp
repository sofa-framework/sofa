/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/** Sparse matrix test suite.
  Class TestSparseMatrices is used as fixture in Boost.Test suite.
  The actual tests are performed within boost macros at the bottom of the file.
  */

#include <boost/test/auto_unit_test.hpp>

//#include "EigenSparseMatrix.h"
#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/FullVector.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

/** Set up a context for unit tests of sparse matrices.
  Used as fixture in Boost.Test suite.
  */
template <typename _Real, unsigned NumRows, unsigned NumCols, unsigned BlockRows, unsigned BlockCols>
struct TestSparseMatrices
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
    typedef sofa::defaulttype::Mat<BROWS,BCOLS,Real> Block;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<Block> CRSMatrix;

    // Implementation based on Eigen
    typedef sofa::defaulttype::StdVectorTypes< sofa::defaulttype::Vec<BCOLS,Real>, sofa::defaulttype::Vec<BCOLS,Real> > InTypes;
    typedef sofa::defaulttype::StdVectorTypes< sofa::defaulttype::Vec<BROWS,Real>, sofa::defaulttype::Vec<BROWS,Real> > OutTypes;
    typedef sofa::component::linearsolver::EigenSparseMatrix<InTypes,OutTypes> EigenBlockMatrix;
    typedef sofa::component::linearsolver::EigenBaseSparseMatrix<Real> EigenBaseMatrix;


    // The matrices used in the tests
    CRSMatrix crs1,crs2;
    FullMatrix fullMat;
    MapMatrix mapMat;
    EigenBlockMatrix eiBlock1,eiBlock2;
    EigenBaseMatrix eiBase;

    // The vectors used in the tests
    FullVector fullVec_ncols;
    FullVector fullVec_nrows_reference,fullVec_nrows_result;


    /// Create the context for the matrix tests.
    TestSparseMatrices()
    {
        // resize and fill the matrices
        crs1.resize(NROWS,NCOLS);
        crs2.resize(NROWS,NCOLS);
        fullMat.resize(NROWS,NCOLS);
        mapMat.resize(NROWS,NCOLS);
        eiBlock1.resize(NROWS,NCOLS);
        eiBlock2.resize(NROWS,NCOLS);
        eiBase.resize(NROWS,NCOLS);
        for( unsigned j=0; j<NCOLS; j++)
        {
            for( unsigned i=0; i<NROWS; i++)
            {
                double valij = i*NCOLS+j;
                crs1.set(i,j,valij);
                Block* b = crs2.wbloc(i/BROWS,j/BCOLS,true);
                assert(b && "a matrix block exists");
                (*b)[i%BROWS][j%BCOLS] = valij;
                fullMat.set(i,j,valij);
                mapMat.set(i,j,valij);
                eiBlock1.add(i,j,valij);
                eiBase.add(i,j,valij);
                Block& bb = eiBlock2.wBlock(i/BROWS,j/BCOLS);
                bb[i%BROWS][j%BCOLS] = valij;
            }
        }
        crs1.compress(); crs2.compress(); eiBlock1.compress(); eiBlock2.compress(); eiBase.compress();

        // resize and fill the vectors
        fullVec_ncols.resize(NCOLS);
        fullVec_nrows_reference.resize(NROWS);
        fullVec_nrows_result.resize(NROWS);
        for( unsigned i=0; i<NCOLS; i++)
        {
            fullVec_ncols[i] = i;
        }
        fullMat.mul(fullVec_nrows_reference,fullVec_ncols); //    cerr<<"MatrixTest: vref = " << vref << endl;
    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    template<typename Matrix1, typename Matrix2>
    static bool matricesAreEqual( const Matrix1& m1, const Matrix2& m2, double tolerance=std::numeric_limits<double>::epsilon() )
    {
        if(m1.rowSize()!=m2.rowSize() || m2.colSize()!=m1.colSize()) return false;
        for( unsigned i=0; i<m1.rowSize(); i++ )
            for( unsigned j=0; j<m1.colSize(); j++ )
                if( fabs(m1.element(i,j)-m2.element(i,j))>tolerance  ) return false;
        return true;
    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    template< typename Vector1, typename Vector2>
    static bool vectorsAreEqual( const Vector1& m1, const Vector2& m2, double tolerance=std::numeric_limits<double>::epsilon() )
    {
        if( m1.size()!=m2.size() ) return false;
        for( unsigned i=0; i<m1.size(); i++ )
            if( fabs(m1.element(i)-m2.element(i))>tolerance  ) return false;
        return true;
    }

    /** Check that EigenMatrix update works as well as direct init. Return true if the test succeeds.*/
    bool checkEigenMatrixUpdate()
    {
        // fill two matrices with the same values, one directly, one it two passes, then compare their values
        EigenBlockMatrix a,b;
        a.resize(NROWS,NCOLS);
        b.resize(NROWS,NCOLS);
        for( unsigned j=0; j<NCOLS; j++)
        {
            for( unsigned i=0; i<NROWS; i++)
            {
                double valij = i*NCOLS+j;
                a.add(i,j,valij);
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
        return matricesAreEqual(a,b);
    }

    /** Check the filling of EigenMatrix per rows of blocks. Return true if the test succeeds.*/
    bool checkEigenMatrixBlockRowFilling()
    {
        EigenBlockMatrix ma,mb;
        unsigned br=3, bc=3;
        ma.resizeBlocks(br,bc);
        mb.resizeBlocks(br,bc);
        for( unsigned i=0; i<br; i++ )
        {
            mb.beginBlockRow(i);
            if( i%2==0 ) // leave some rows empty
            {
                for( int j=bc-1; j>=0; j--) // set the blocs in reverse order
                {
                    // create a block and give it some value
                    Block b;
                    for( unsigned k=0; k<BROWS && k<BCOLS; k++ )
                        b[k][k] = i+j;

                    // insert the block in the two matrices
                    mb.createBlock(j,b);
                    ma.wBlock(i,j) = b;
                }
            }
            mb.endBlockRow();
        }
        ma.compress();
        mb.compress();
        //    serr()<<"MatrixTest<Real,RN,CN>::checkEigenMatrixBlockRowFilling, ma = " << ma << endl;
        //    serr()<<"MatrixTest<Real,RN,CN>::checkEigenMatrixBlockRowFilling, mb = " << mb << endl;
        return matricesAreEqual(ma,mb);
    }


};


typedef TestSparseMatrices<double,4,8,2,2> Ts1;
BOOST_FIXTURE_TEST_SUITE( SparseMatrix_double_4_8_2_2, Ts1 );
#include "MatrixTest.inl"
BOOST_AUTO_TEST_SUITE_END();


typedef TestSparseMatrices<float,4,8,2,3> Ts2;
BOOST_FIXTURE_TEST_SUITE( SparseMatrix_float_4_8_2_2, Ts2 );
#include "MatrixTest.inl"
BOOST_AUTO_TEST_SUITE_END();


