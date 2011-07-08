#include <sofa/helper/system/atomic.h>
#include <boost/test/auto_unit_test.hpp>

using sofa::helper::system::atomic;

BOOST_AUTO_TEST_CASE(dec_and_test_null)
{
    atomic<int> value(3);
    BOOST_CHECK_EQUAL(value.dec_and_test_null(), false);
    BOOST_CHECK_EQUAL(value, 2);
    BOOST_CHECK_EQUAL(value.dec_and_test_null(), false);
    BOOST_CHECK_EQUAL(value, 1);
    BOOST_CHECK_EQUAL(value.dec_and_test_null(), true);
    BOOST_CHECK_EQUAL(value, 0);
}

BOOST_AUTO_TEST_CASE(compare_and_swap)
{
    atomic<int> value(-1);
    BOOST_CHECK_EQUAL(value.compare_and_swap(-1, 10), -1);
    BOOST_CHECK_EQUAL(value, 10);

    BOOST_CHECK_EQUAL(value.compare_and_swap(5, 25), 10);
    BOOST_CHECK_EQUAL(value, 10);
}
