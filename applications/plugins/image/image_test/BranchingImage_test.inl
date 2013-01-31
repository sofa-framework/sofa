BOOST_AUTO_TEST_CASE( creation )
{
    BOOST_CHECK( dimensionsAreEqual(flatImage.getDimensions(),branchingImage.getDimension()) );
    BOOST_CHECK( imagesAreEqual(flatImage,branchingImage,true,false) ); // value test
    BOOST_CHECK( imagesAreEqual(flatImage,branchingImage,false,true) ); // neighbour test
    BOOST_CHECK( branchingImage == branchingImage2 );
    BOOST_CHECK( flatImage == flatImage2 );
}
