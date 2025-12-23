#include <sofa/helper/IotaView.h>
#include <gtest/gtest.h>

namespace sofa
{

TEST(IotaView, loop)
{
    const auto range = sofa::helper::IotaView{0, 10};

    int i = 0;
    for (const auto value : range)
    {
        EXPECT_EQ(value, i);
        ++i;
    }
}

TEST(IotaView, empty)
{
    {
        const auto range = sofa::helper::IotaView{0, 10};
        EXPECT_FALSE(range.empty());
    }
    {
        const auto range = sofa::helper::IotaView{0, 0};
        EXPECT_TRUE(range.empty());
    }
}

TEST(IotaView, size)
{
    const auto range = sofa::helper::IotaView{0, 10};
    EXPECT_EQ(range.size(), 10);
}

TEST(IotaView, access)
{
    const auto range = sofa::helper::IotaView{4, 10};
    EXPECT_EQ(range[0], 4);
    EXPECT_EQ(range[9], 4+9);
}

}
