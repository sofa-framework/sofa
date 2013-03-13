TEST_F( BranchingImage, creation )
{
    ASSERT_TRUE( dimensionsAreEqual(flatImage.getDimensions(),branchingImage6.getDimension()) );
}
TEST_F( BranchingImage, copy )
{
    ASSERT_TRUE( branchingImage6 == branchingImage2 );
}
TEST_F( BranchingImage, conversion )
{
    int r = branchingImage6.isEqual(flatImage,sofa::defaulttype::CONNECTIVITY_6,true,false);
    ASSERT_TRUE( !r ); // value test
    if( r ) { std::cerr<<"6-connectivity value test error = "<<r<<std::endl; }

    r = branchingImage6.isEqual(flatImage,sofa::defaulttype::CONNECTIVITY_6,false,true);
    ASSERT_TRUE( !r ); // neighbour test
    if( r ) { std::cerr<<"6-connectivity neighbour test error = "<<r<<std::endl; }

    r = branchingImage26.isEqual(flatImage,sofa::defaulttype::CONNECTIVITY_26,true,false);
    ASSERT_TRUE( !r ); // value test
    if( r ) { std::cerr<<"26-connectivity value test error = "<<r<<std::endl; }

    r = branchingImage26.isEqual(flatImage,sofa::defaulttype::CONNECTIVITY_26,false,true);
    ASSERT_TRUE( !r ); // neighbour test
    if( r ) { std::cerr<<"26-connectivity neighbour test error = "<<r<<std::endl; }

    ASSERT_TRUE( flatImage == flatImage2 );
}
TEST_F( BranchingImage, neighbourhoodValidity )
{
    int r = branchingImage6.isNeighbourhoodValid();
    ASSERT_TRUE( !r );
    if( r ) { std::cerr<<"6-connectivity Neighbourhood error = "<<r<<std::endl; }

    r = branchingImage2.isNeighbourhoodValid();
    ASSERT_TRUE( !r );
    if( r ) { std::cerr<<"6-connectivity copy Neighbourhood error = "<<r<<std::endl; }

    r = branchingImage26.isNeighbourhoodValid();
    ASSERT_TRUE( !r );
    if( r ) { std::cerr<<"26-connectivity Neighbourhood error = "<<r<<std::endl; }
}

