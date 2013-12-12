/**
  This file contains a sequence of tests to perform for different instances of a templatized fixture.
  It is thus inlined several times in the .cpp test file.
  */

// ==============================
// Set/get value tests
TEST_F(TestMatrix, set_fullMat ) { ASSERT_TRUE( matricesAreEqual(mat,fullMat)); }
TEST_F(TestMatrix, set_crs1 ) { ASSERT_TRUE( matricesAreEqual(fullMat,crs1)); }
TEST_F(TestMatrix, set_crs2 ) { ASSERT_TRUE( matricesAreEqual(fullMat,crs2)); }
TEST_F(TestMatrix, set_mapMat ) { ASSERT_TRUE( matricesAreEqual(fullMat,mapMat)); }
TEST_F(TestMatrix, set_eiBlock1 ) { ASSERT_TRUE( matricesAreEqual(fullMat,eiBlock1)); }
TEST_F(TestMatrix, set_eiBlock2 ) { ASSERT_TRUE( matricesAreEqual(fullMat,eiBlock2)); }
TEST_F(TestMatrix, set_eiBase ) { ASSERT_TRUE( matricesAreEqual(fullMat,eiBase)); }
TEST_F(TestMatrix, eigenMatrix_update ) { ASSERT_TRUE( checkEigenMatrixUpdate() ); }
TEST_F(TestMatrix, eigenMatrix_block_row_filling ) { ASSERT_TRUE( checkEigenMatrixBlockRowFilling() ); }
TEST_F(TestMatrix, eigenMatrixBlockFromCompressedRowSparseMatrix ) { ASSERT_TRUE( checkEigenMatrixBlockFromCompressedRowSparseMatrix() ); }

// ==============================
// Matrix-Vector product tests
TEST_F(TestMatrix, set_fullVec_nrows_reference )
{
    ASSERT_TRUE(vectorsAreEqual(vecM,fullVec_nrows_reference));
}
TEST_F(TestMatrix, fullMat_vector_product )
{
//    fullMat.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    fullVec_nrows_result = fullMat * fullVec_ncols;
    ASSERT_TRUE(vectorsAreEqual(fullVec_nrows_reference,fullVec_nrows_result));
}
TEST_F(TestMatrix, mapMat_vector_product )
{
//    mapMat.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    fullVec_nrows_result = mapMat * fullVec_ncols;
    ASSERT_TRUE(vectorsAreEqual(fullVec_nrows_reference,fullVec_nrows_result));
}
TEST_F(TestMatrix, eiBlock1_vector_product )
{
//    eiBlock1.opMulV(&fullVec_nrows_result,&fullVec_ncols);
//    eiBlock1.multVector(fullVec_nrows_result,fullVec_ncols);
    fullVec_nrows_result = eiBlock1 * fullVec_ncols;
    ASSERT_TRUE(vectorsAreEqual(fullVec_nrows_reference,fullVec_nrows_result));

}
TEST_F(TestMatrix, crs1_vector_product )
{
//    EXPECT_TRUE(NROWS%BROWS==0 && NCOLS%BCOLS==0) << "Error: CompressedRowSparseMatrix * Vector crashes when the size of the matrix is not a multiple of the size of the matrix blocks. Aborting this test, and reporting a failure."; // otherwise the product crashes
//    crs1.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    fullVec_nrows_result = crs1 * fullVec_ncols;
    ASSERT_TRUE(vectorsAreEqual(fullVec_nrows_reference,fullVec_nrows_result));
}


// ==============================
// Matrix product tests
TEST_F(TestMatrix, full_matrix_product ) { ASSERT_TRUE( matricesAreEqual(matMultiplication,fullMultiplication));  }
TEST_F(TestMatrix, crs_matrix_product ) { ASSERT_TRUE( matricesAreEqual(fullMultiplication,crsMultiplication)); }
TEST_F(TestMatrix, full_matrix_transposeproduct ) { ASSERT_TRUE( matricesAreEqual(matTransposeMultiplication,fullTransposeMultiplication)); }
TEST_F(TestMatrix, crs_matrix_transposeproduct ) { ASSERT_TRUE( matricesAreEqual(fullTransposeMultiplication,crsTransposeMultiplication)); }

// Matrix addition
TEST_F(TestMatrix, crs_matrix_addition )
{
    crs2 = crs1 + crs1;
    ASSERT_TRUE( matricesAreEqual(mat*2,crs2));

    crs2 += crs1;
    ASSERT_TRUE( matricesAreEqual(mat*3,crs2));

    crs2 -= crs1;
    ASSERT_TRUE( matricesAreEqual(mat*2,crs2));
}
