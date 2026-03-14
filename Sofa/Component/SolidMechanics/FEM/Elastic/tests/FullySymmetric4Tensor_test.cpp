#include <sofa/component/solidmechanics/fem/elastic/impl/FullySymmetric4Tensor.h>
#include <gtest/gtest.h>

namespace sofa
{

using namespace sofa::component::solidmechanics::fem::elastic;

template<class DataTypes>
void generateThenAccess()
{
    constexpr auto spatial_dimensions = DataTypes::spatial_dimensions;
    const FullySymmetric4Tensor<DataTypes> t(
        [](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
        {
            return kroneckerDelta<SReal>(i, j) * kroneckerDelta<SReal>(k, l);
        });

    for (sofa::Index i = 0; i < spatial_dimensions; ++i)
    {
        for (sofa::Index j = 0; j < spatial_dimensions; ++j)
        {
            for (sofa::Index k = 0; k < spatial_dimensions; ++k)
            {
                for (sofa::Index l = 0; l < spatial_dimensions; ++l)
                {
                    EXPECT_DOUBLE_EQ(t(i, j, k, l), static_cast<SReal>(i == j) * static_cast<SReal>(k == l)) << "i = " << i << " j = " << j << " k = " << k << " l = " << l;
                }
            }
        }
    }
}

TEST(FullySymmetric4Tensor, generateThenAccess1d)
{
    generateThenAccess<sofa::defaulttype::Vec1Types>();
}
TEST(FullySymmetric4Tensor, generateThenAccess2d)
{
    generateThenAccess<sofa::defaulttype::Vec2Types>();
}
TEST(FullySymmetric4Tensor, generateThenAccess3d)
{
    generateThenAccess<sofa::defaulttype::Vec3Types>();
}

}
