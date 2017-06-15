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
/**
  This file contains a sequence of tests to perform for different instances of a templatized fixture.
  It is thus inlined several times in the .cpp test file.
  */

// ==============================
// Set/get value tests
TEST_F(TestMatrix, set_fullMat ) { ASSERT_TRUE( matrixMaxDiff(mat,fullMat) < 100*epsilon() ); }
TEST_F(TestMatrix, set_crs1 ) { ASSERT_TRUE( matrixMaxDiff( fullMat,crs1 ) < 100*epsilon() ); }
TEST_F(TestMatrix, set_crs2 ) { ASSERT_TRUE( matrixMaxDiff(fullMat,crs2) < 100*epsilon() ); }
TEST_F(TestMatrix, set_mapMat ) { ASSERT_TRUE( matrixMaxDiff(fullMat,mapMat) < 100*epsilon() ); }
TEST_F(TestMatrix, set_eiBlock1 ) { ASSERT_TRUE( matrixMaxDiff(fullMat,eiBlock1) < 100*epsilon() ); }
TEST_F(TestMatrix, set_eiBlock2 ) { ASSERT_TRUE( matrixMaxDiff(fullMat,eiBlock2) < 100*epsilon() ); }
TEST_F(TestMatrix, set_eiBase ) { ASSERT_TRUE( matrixMaxDiff(fullMat,eiBase) < 100*epsilon() ); }
TEST_F(TestMatrix, eigenMatrix_update ) { ASSERT_TRUE( checkEigenMatrixUpdate() ); }
TEST_F(TestMatrix, eigenMatrix_block_row_filling ) { checkEigenMatrixBlockRowFilling(); }
TEST_F(TestMatrix, eigenMatrixBlockFromCompressedRowSparseMatrix ) { ASSERT_TRUE( checkEigenMatrixBlockFromCompressedRowSparseMatrix() ); }
TEST_F(TestMatrix, eigenMapToDenseMatrix ) { ASSERT_TRUE( checkEigenDenseMatrix() ); }

// ==============================
// Matrix-Vector product tests
TEST_F(TestMatrix, set_fullVec_nrows_reference )
{
    ASSERT_TRUE(vectorMaxDiff(vecM,fullVec_nrows_reference) < epsilon() );
}
TEST_F(TestMatrix, fullMat_vector_product )
{
//    fullMat.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    fullVec_nrows_result = fullMat * fullVec_ncols;
    ASSERT_TRUE(vectorMaxDiff(fullVec_nrows_reference,fullVec_nrows_result) < epsilon() );
}
TEST_F(TestMatrix, mapMat_vector_product )
{
//    mapMat.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    fullVec_nrows_result = mapMat * fullVec_ncols;
    ASSERT_TRUE(vectorMaxDiff(fullVec_nrows_reference,fullVec_nrows_result) < epsilon() );
}
TEST_F(TestMatrix, eiBlock1_vector_product )
{
//    eiBlock1.opMulV(&fullVec_nrows_result,&fullVec_ncols);
//    eiBlock1.multVector(fullVec_nrows_result,fullVec_ncols);
    fullVec_nrows_result = eiBlock1 * fullVec_ncols;
    ASSERT_TRUE(vectorMaxDiff(fullVec_nrows_reference,fullVec_nrows_result) < epsilon() );

}
TEST_F(TestMatrix, crs1_vector_product )
{
//    EXPECT_TRUE(NROWS%BROWS==0 && NCOLS%BCOLS==0) << "Error: CompressedRowSparseMatrix * Vector crashes when the size of the matrix is not a multiple of the size of the matrix blocks. Aborting this test, and reporting a failure."; // otherwise the product crashes
//    crs1.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    fullVec_nrows_result = crs1 * fullVec_ncols;
    ASSERT_TRUE(vectorMaxDiff(fullVec_nrows_reference,fullVec_nrows_result) < epsilon() );
}


// ==============================
// Matrix product tests
TEST_F(TestMatrix, full_matrix_product ) { ASSERT_TRUE( matrixMaxDiff(matMultiplication,fullMultiplication) < 100*epsilon() );  }
TEST_F(TestMatrix, crs_matrix_product ) { ASSERT_TRUE( matrixMaxDiff(fullMultiplication,crsMultiplication) < 100*epsilon() ); }
TEST_F(TestMatrix, EigenBase_matrix_product ) { ASSERT_TRUE( matrixMaxDiff(fullMultiplication,eiBaseMultiplication) < 100*epsilon() ); }
TEST_F(TestMatrix, EigenSparseDense_matrix_product ) { ASSERT_TRUE( EigenDenseMatrix(eiBaseMultiplication.compressedMatrix) == eiDenseMultiplication ); }
TEST_F(TestMatrix, full_matrix_transposeproduct ) { ASSERT_TRUE( matrixMaxDiff(matTransposeMultiplication,fullTransposeMultiplication) < 100*epsilon() ); }
TEST_F(TestMatrix, crs_matrix_transposeproduct ) { ASSERT_TRUE( matrixMaxDiff(fullTransposeMultiplication,crsTransposeMultiplication) < 100*epsilon() ); }

// Matrix addition
TEST_F(TestMatrix, crs_matrix_addition )
{
    crs2 = crs1 + crs1;
    ASSERT_TRUE( matrixMaxDiff(mat*2,crs2) < 100*epsilon() );

    crs2 += crs1;
    ASSERT_TRUE( matrixMaxDiff(mat*3,crs2) < 100*epsilon() );

    crs2 -= crs1;
    ASSERT_TRUE( matrixMaxDiff(mat*2,crs2) < 100*epsilon() );
}
