#include <SofaTest/Sofa_test.h>

namespace sofa {

struct Dummy_test : public Sofa_test<>
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
