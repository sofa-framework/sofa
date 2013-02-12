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
    int r = imagesAreEqual(flatImage,branchingImage,true,false);
    BOOST_CHECK( !r ); // value test
    if( r ) { std::cerr<<"value test error = "<<r<<std::endl; }

    r = imagesAreEqual(flatImage,branchingImage,false,true);
    BOOST_CHECK( !r ); // neighbour test
    if( r ) { std::cerr<<"neighbour test error = "<<r<<std::endl; }

    BOOST_CHECK( flatImage == flatImage2 );
}
BOOST_AUTO_TEST_CASE( neighbourhoodValidity )
{
    int r = branchingImage.isNeighbourhoodValid();
    BOOST_CHECK( !r );
    if( r ) { std::cerr<<"Neighbourhood error = "<<r<<std::endl; }

    r = branchingImage2.isNeighbourhoodValid();
    BOOST_CHECK( !r );
    if( r ) { std::cerr<<"Neighbourhood error = "<<r<<std::endl; }
}

