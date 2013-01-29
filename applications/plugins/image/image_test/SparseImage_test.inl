BOOST_AUTO_TEST_CASE( creation )
{
    BOOST_CHECK( dimensionsAreEqual(flatImage.getDimensions(),sparseImage.getDimension()) );
    BOOST_CHECK( imagesAreEqual(flatImage,sparseImage) );
}
