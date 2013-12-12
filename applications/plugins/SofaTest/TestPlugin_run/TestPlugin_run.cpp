#include <boost/test/unit_test.hpp>

// a fixture, in boost::test language, is a helper class that brings
// your program into a state suitable for testing, in the class
// constructor, and cleans up behind it in the destructor.

// anonymous namespace to prevent symbol clashes in case several tests
// have a fixture class
namespace {

  struct fixture {

    // some state
    int i;
    
    fixture() {
      // here goes setup
      i = 1;
    }
    
    ~fixture() {
      // here goes cleanup
      
    }
    
  };
  
}

// test suite declaration: @name and @fixture class
BOOST_FIXTURE_TEST_SUITE( TestPlugin, fixture );


// test case declaration: @name
BOOST_AUTO_TEST_CASE( initialization ) {

  BOOST_CHECK( i == 1 );
  
}

// end of test suite
BOOST_AUTO_TEST_SUITE_END()
