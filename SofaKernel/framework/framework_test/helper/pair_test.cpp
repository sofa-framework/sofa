#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;
using testing::Types;

#include <sofa/helper/pair.h>

template<class T1, class T2>
struct pair_test : public ::testing::Test
{
    std::pair<T1, T2> m_pair; ///< tested pair

    /// reading directly from a string
    void read(const std::string& s)
    {
        std::istringstream ss( s );
        ss >> m_pair;
    }

    void checkRead(std::pair<std::string, std::pair<int, SReal>> const& value) {
        read(value.first);
        EXPECT_EQ(value.second.first, m_pair.first);
        EXPECT_FLOAT_EQ(value.second.second, m_pair.second);
    }


};

typedef pair_test<int, SReal> pair_test_int_real;

std::vector<std::pair<std::string, std::pair<int, SReal>> > int_real_values = {
    {"2 3.45", {2, 3.45}},
    {"[2, 3.45]", {2, 3.45}}
};


TEST_F(pair_test_int_real, checkRead)
{
    for (auto const& value: int_real_values)
        this->checkRead(value);
}
