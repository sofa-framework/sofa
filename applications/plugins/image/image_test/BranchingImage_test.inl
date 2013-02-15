BOOST_AUTO_TEST_CASE( creation )
{
    BOOST_CHECK( dimensionsAreEqual(flatImage.getDimensions(),branchingImage6.getDimension()) );
}
BOOST_AUTO_TEST_CASE( copy )
{
    BOOST_CHECK( branchingImage6 == branchingImage2 );
}
BOOST_AUTO_TEST_CASE( conversion )
{
    int r = branchingImage6.isEqual(flatImage,sofa::defaulttype::CONNECTIVITY_6,true,false);
    BOOST_CHECK( !r ); // value test
    if( r ) { std::cerr<<"6-connectivity value test error = "<<r<<std::endl; }

    r = branchingImage6.isEqual(flatImage,sofa::defaulttype::CONNECTIVITY_6,false,true);
    BOOST_CHECK( !r ); // neighbour test
    if( r ) { std::cerr<<"6-connectivity neighbour test error = "<<r<<std::endl; }

    r = branchingImage26.isEqual(flatImage,sofa::defaulttype::CONNECTIVITY_26,true,false);
    BOOST_CHECK( !r ); // value test
    if( r ) { std::cerr<<"26-connectivity value test error = "<<r<<std::endl; }

    r = branchingImage26.isEqual(flatImage,sofa::defaulttype::CONNECTIVITY_26,false,true);
    BOOST_CHECK( !r ); // neighbour test
    if( r ) { std::cerr<<"26-connectivity neighbour test error = "<<r<<std::endl; }

    BOOST_CHECK( flatImage == flatImage2 );
}
BOOST_AUTO_TEST_CASE( neighbourhoodValidity )
{
    int r = branchingImage6.isNeighbourhoodValid();
    BOOST_CHECK( !r );
    if( r ) { std::cerr<<"6-connectivity Neighbourhood error = "<<r<<std::endl; }

    r = branchingImage2.isNeighbourhoodValid();
    BOOST_CHECK( !r );
    if( r ) { std::cerr<<"6-connectivity copy Neighbourhood error = "<<r<<std::endl; }

    r = branchingImage26.isNeighbourhoodValid();
    BOOST_CHECK( !r );
    if( r ) { std::cerr<<"26-connectivity Neighbourhood error = "<<r<<std::endl; }
}

