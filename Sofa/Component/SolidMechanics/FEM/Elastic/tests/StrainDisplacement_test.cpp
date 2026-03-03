#include <sofa/component/solidmechanics/fem/elastic/impl/StrainDisplacement.h>
#include <gtest/gtest.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/geometry/Tetrahedron.h>

namespace sofa
{

using namespace sofa::component::solidmechanics::fem::elastic;

TEST(StraintDisplacement, matrixVectorProduct)
{
    sofa::type::Mat<4, 3, SReal> dN_dq(sofa::type::NOINIT);
    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            dN_dq(i, j) = (i + 4) * (j + 9);
        }
    }

    const auto B = makeStrainDisplacement<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>(dN_dq);

    sofa::type::Vec<12, SReal> v;

    for (std::size_t i = 0; i < 12; ++i)
    {
        v[i] = static_cast<SReal>(i);
    }

    const auto Bv = B * v;
    const auto expectedBv = B.B * v;

    for (std::size_t i = 0; i < 6; ++i)
    {
        EXPECT_DOUBLE_EQ(Bv[i], expectedBv[i]) << "i = " << i;
    }
}

TEST(StraintDisplacement, matrixTransposedVectorProduct)
{
    sofa::type::Mat<4, 3, SReal> dN_dq(sofa::type::NOINIT);
    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            dN_dq(i, j) = (i + 4) * (j + 9);
        }
    }

    const auto B = makeStrainDisplacement<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>(dN_dq);

    sofa::type::Vec<6, SReal> v;
    for (std::size_t i = 0; i < 6; ++i)
    {
        v[i] = static_cast<SReal>(i);
    }

    const auto B_Tv = B.multTranspose(v);
    const auto expectedB_Tv = B.B.multTranspose(v);

    for (std::size_t i = 0; i < 12; ++i)
    {
        EXPECT_DOUBLE_EQ(B_Tv[i], expectedB_Tv[i]) << "i = " << i;
    }
}
}
