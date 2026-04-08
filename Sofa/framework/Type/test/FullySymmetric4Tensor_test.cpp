#include <sofa/type/FullySymmetric4Tensor.h>
#include <gtest/gtest.h>

namespace sofa
{

using namespace sofa::type;

template<std::size_t D>
void generateThenAccess()
{
    const FullySymmetric4Tensor<D, SReal> t(
        [](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
        {
            return static_cast<SReal>(i == j) * static_cast<SReal>(k == l);
        });

    for (sofa::Index i = 0; i < D; ++i)
    {
        for (sofa::Index j = 0; j < D; ++j)
        {
            for (sofa::Index k = 0; k < D; ++k)
            {
                for (sofa::Index l = 0; l < D; ++l)
                {
                    EXPECT_DOUBLE_EQ(t(i, j, k, l), static_cast<SReal>(i == j) * static_cast<SReal>(k == l)) << "i = " << i << " j = " << j << " k = " << k << " l = " << l;
                }
            }
        }
    }
}

TEST(FullySymmetric4Tensor, generateThenAccess1d)
{
    generateThenAccess<1>();
}
TEST(FullySymmetric4Tensor, generateThenAccess2d)
{
    generateThenAccess<2>();
}
TEST(FullySymmetric4Tensor, generateThenAccess3d)
{
    generateThenAccess<3>();
}

template<std::size_t D>
void defaultConstructorAndFill()
{
    FullySymmetric4Tensor<D, SReal> t;
    t.fill([](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
    {
        return static_cast<SReal>(i + j + k + l);
    });

    for (sofa::Index i = 0; i < D; ++i)
    {
        for (sofa::Index j = 0; j < D; ++j)
        {
            for (sofa::Index k = 0; k < D; ++k)
            {
                for (sofa::Index l = 0; l < D; ++l)
                {
                    EXPECT_DOUBLE_EQ(t(i, j, k, l), static_cast<SReal>(i + j + k + l));
                }
            }
        }
    }
}

TEST(FullySymmetric4Tensor, defaultConstructorAndFill1d) { defaultConstructorAndFill<1>(); }
TEST(FullySymmetric4Tensor, defaultConstructorAndFill2d) { defaultConstructorAndFill<2>(); }
TEST(FullySymmetric4Tensor, defaultConstructorAndFill3d) { defaultConstructorAndFill<3>(); }

template<std::size_t D>
void readWriteAccess()
{
    FullySymmetric4Tensor<D, SReal> t;
    t.fill([](sofa::Index, sofa::Index, sofa::Index, sofa::Index) { return 0.0; });

    t(0, 0, 0, 0) = 1.0;
    EXPECT_DOUBLE_EQ(t(0, 0, 0, 0), 1.0);

    if (D > 1)
    {
        t(0, 1, 0, 1) = 2.0;
        EXPECT_DOUBLE_EQ(t(0, 1, 0, 1), 2.0);
        EXPECT_DOUBLE_EQ(t(1, 0, 0, 1), 2.0);
        EXPECT_DOUBLE_EQ(t(0, 1, 1, 0), 2.0);
        EXPECT_DOUBLE_EQ(t(1, 0, 1, 0), 2.0);
    }
}

TEST(FullySymmetric4Tensor, readWriteAccess1d) { readWriteAccess<1>(); }
TEST(FullySymmetric4Tensor, readWriteAccess2d) { readWriteAccess<2>(); }
TEST(FullySymmetric4Tensor, readWriteAccess3d) { readWriteAccess<3>(); }

template<std::size_t D>
void toVoigtMatSym()
{
    FullySymmetric4Tensor<D, SReal> t(
        [](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
        {
            return static_cast<SReal>(tensorToVoigtIndex<D>(i, j) + tensorToVoigtIndex<D>(k, l));
        });

    const auto& m = t.toVoigtMatSym();
    constexpr auto N = sofa::type::NumberOfIndependentElements<D>;
    for (sofa::Index a = 0; a < N; ++a)
    {
        for (sofa::Index b = 0; b < N; ++b)
        {
            EXPECT_DOUBLE_EQ(m(static_cast<int>(a), static_cast<int>(b)), static_cast<SReal>(a + b));
        }
    }
}

TEST(FullySymmetric4Tensor, toVoigtMatSym1d) { toVoigtMatSym<1>(); }
TEST(FullySymmetric4Tensor, toVoigtMatSym2d) { toVoigtMatSym<2>(); }
TEST(FullySymmetric4Tensor, toVoigtMatSym3d) { toVoigtMatSym<3>(); }

}
