BOOST_AUTO_TEST_CASE( creation )
{
    BOOST_CHECK( dimensionsAreEqual(flatImage.getDimensions(),branchingImage.getDimension()) );
}
BOOST_AUTO_TEST_CASE( copy )
{
    BOOST_CHECK( branchingImage == branchingImage2 );
}
BOOST_AUTO_TEST_CASE( conversion )
{
    BOOST_CHECK( imagesAreEqual(flatImage,branchingImage,true,false) ); // value test
    BOOST_CHECK( imagesAreEqual(flatImage,branchingImage,false,true) ); // neighbour test
    BOOST_CHECK( flatImage == flatImage2 );
}
BOOST_AUTO_TEST_CASE( neighbouroodValidity )
{
    int r = branchingImage.isNeighbouroodValid(); if( r ) { std::cerr<<r<<std::endl; }
    BOOST_CHECK( !r );

    r = branchingImage2.isNeighbouroodValid(); if( r ) { std::cerr<<r<<std::endl; }
    BOOST_CHECK( !r );
}

