#include <gtest/gtest.h>

namespace sofa {

struct Dummy_test : public ::testing::Test
{
    Dummy_test()
    {
    }

};

TEST_F( Dummy_test, dummy)
{
    EXPECT_EQ(0, 0);
}

}// namespace sofa
