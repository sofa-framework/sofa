/**
  This file contains a sequence of tests to perform for different instances of a templatized fixture.
  It is thus inlined several times in the .cpp test file.
  */

// ==============================
// Set/get value tests
BOOST_AUTO_TEST_CASE( set_crs1 ) { BOOST_CHECK( matricesAreEqual(fullMat,crs1)); }
BOOST_AUTO_TEST_CASE( set_crs2 ) { BOOST_CHECK( matricesAreEqual(fullMat,crs2)); }
BOOST_AUTO_TEST_CASE( set_mapMat ) { BOOST_CHECK( matricesAreEqual(fullMat,mapMat)); }
BOOST_AUTO_TEST_CASE( set_eiBlock1 ) { BOOST_CHECK( matricesAreEqual(fullMat,eiBlock1)); }
BOOST_AUTO_TEST_CASE( set_eiBlock2 ) { BOOST_CHECK( matricesAreEqual(fullMat,eiBlock2)); }
BOOST_AUTO_TEST_CASE( set_eiBase ) { BOOST_CHECK( matricesAreEqual(fullMat,eiBase)); }
BOOST_AUTO_TEST_CASE( eigenMatrix_update ) { BOOST_CHECK( checkEigenMatrixUpdate() ); }
BOOST_AUTO_TEST_CASE( eigenMatrix_block_row_filling ) { BOOST_CHECK( checkEigenMatrixBlockRowFilling() ); }
BOOST_AUTO_TEST_CASE( eigenMatrixBlockFromCompressedRowSparseMatrix ) { BOOST_CHECK( checkEigenMatrixBlockFromCompressedRowSparseMatrix() ); }


// ==============================
// Matrix-Vector product tests
BOOST_AUTO_TEST_CASE( fullMat_vector_product )
{
//    fullMat.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    fullVec_nrows_result = fullMat * fullVec_ncols;
    BOOST_CHECK(vectorsAreEqual(fullVec_nrows_reference,fullVec_nrows_result));
}
BOOST_AUTO_TEST_CASE( mapMat_vector_product )
{
//    mapMat.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    fullVec_nrows_result = mapMat * fullVec_ncols;
    BOOST_CHECK(vectorsAreEqual(fullVec_nrows_reference,fullVec_nrows_result));
}
BOOST_AUTO_TEST_CASE( eiBlock1_vector_product )
{
//    eiBlock1.opMulV(&fullVec_nrows_result,&fullVec_ncols);
//    eiBlock1.multVector(fullVec_nrows_result,fullVec_ncols);
    fullVec_nrows_result = eiBlock1 * fullVec_ncols;
    BOOST_CHECK(vectorsAreEqual(fullVec_nrows_reference,fullVec_nrows_result));

}
BOOST_AUTO_TEST_CASE( crs1_vector_product )
{
//    BOOST_REQUIRE_MESSAGE( NROWS%BROWS==0 && NCOLS%BCOLS==0, "Error: CompressedRowSparseMatrix * Vector crashes when the size of the matrix is not a multiple of the size of the matrix blocks. Aborting this test, and reporting a failure." ); // otherwise the product crashes
//    crs1.opMulV(&fullVec_nrows_result,&fullVec_ncols);
    fullVec_nrows_result = crs1 * fullVec_ncols;
    BOOST_CHECK(vectorsAreEqual(fullVec_nrows_reference,fullVec_nrows_result));
}
