/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/IncompressibleMooneyRivlinMaterial.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/MooneyRivlinMaterial.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/OgdenMaterial.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/NeoHookeanMaterial.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/StVenantKirchhoffMaterial.h>
#include <gtest/gtest.h>
#include <sofa/testing/LinearCongruentialRandomGenerator.h>
#include <sofa/testing/BaseTest.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{
template <typename T>
struct SofaClassMaterialTest : public ::testing::Test
{
    void testParentClass()
    {
        using DataTypes = T::DataTypes;
        const auto* HyperelasticMaterialClass = HyperelasticMaterial<DataTypes>::GetClass();
        EXPECT_TRUE(T::GetClass()->hasParent(HyperelasticMaterialClass));
    }
};

using AllSOFAClassMaterials = ::testing::Types<
    StVenantKirchhoffMaterial<sofa::defaulttype::Vec3Types>,
    StVenantKirchhoffMaterial<sofa::defaulttype::Vec2Types>,
    StVenantKirchhoffMaterial<sofa::defaulttype::Vec1Types>,
    NeoHookeanMaterial<sofa::defaulttype::Vec3Types>,
    NeoHookeanMaterial<sofa::defaulttype::Vec2Types>,
    NeoHookeanMaterial<sofa::defaulttype::Vec1Types>,
    IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec3Types>,
    IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec2Types>,
    IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec1Types>,
    MooneyRivlinMaterial<sofa::defaulttype::Vec3Types>,
    MooneyRivlinMaterial<sofa::defaulttype::Vec2Types>,
    MooneyRivlinMaterial<sofa::defaulttype::Vec1Types>,
    OgdenMaterial<sofa::defaulttype::Vec3Types>,
    OgdenMaterial<sofa::defaulttype::Vec2Types>,
    OgdenMaterial<sofa::defaulttype::Vec1Types>
>;
TYPED_TEST_SUITE(SofaClassMaterialTest, AllSOFAClassMaterials);

TYPED_TEST(SofaClassMaterialTest, parentClass)
{
    this->testParentClass();
}


template <typename T>
class BaseMaterialTest : public sofa::testing::BaseTest
{
protected:
    using DataTypes = typename T::DataTypes;
    using Real = sofa::Real_t<DataTypes>;
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    using DeformationGradient = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;

    void doSetUp() override
    {
        material = sofa::core::objectmodel::New<T>();

        if (auto* linearMaterial = dynamic_cast<LinearMechanicalParametersComponent<DataTypes>*>(material.get()) )
        {
            linearMaterial->d_youngModulus.setValue(10.);
        }

        material->init();
    }

    DeformationGradient generatePositiveDefiniteMatrix()
    {
        DeformationGradient F;

        Real J;
        do
        {
            for (int j = 0; j < spatial_dimensions; j++)
            {
                for (int i = 0; i < spatial_dimensions; i++)
                {
                    F(i, j) = lcg.generateInRange(-1., 1.);
                }
            }
            J = determinant(F);
        } while (J < 0.5);

        return F;
    }


    T::SPtr material;
    sofa::testing::LinearCongruentialRandomGenerator lcg { 31321 };
};

template <typename T>
class MaterialTest : public BaseMaterialTest<T>
{
    using BaseMaterialTest<T>::material;

public:

    using DataTypes = typename T::DataTypes;
    using Real = sofa::Real_t<DataTypes>;
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    using DeformationGradient = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;

    void testDerivativePK1()
    {
        const auto F = this->generatePositiveDefiniteMatrix();
        Strain<DataTypes> strain(deformationGradient, F);

        const auto P = material->firstPiolaKirchhoffStress(strain);
        const auto A = material->materialTangentModulus(strain);

        EXPECT_TRUE(std::none_of(P.data(), P.data() + spatial_dimensions * spatial_dimensions, [](Real x) { return x == 0; })) << P;

        // Small perturbation for finite difference
        constexpr Real epsilon = 1e-8;

        // For each component of F
        for(sofa::Size i = 0; i < spatial_dimensions; ++i)
        {
            for(sofa::Size j = 0; j < spatial_dimensions; ++j)
            {
                // Create perturbed deformation gradient
                auto F_perturbed_plus = F;
                F_perturbed_plus(i,j) += epsilon;

                auto F_perturbed_minus = F;
                F_perturbed_minus(i,j) -= epsilon;

                // Compute perturbed stress

                Strain<DataTypes> strainPerturbed_plus(deformationGradient, F_perturbed_plus);
                Strain<DataTypes> strainPerturbed_minus(deformationGradient, F_perturbed_minus);

                const auto P_perturbed_plus = material->firstPiolaKirchhoffStress(strainPerturbed_plus);
                const auto P_perturbed_minus = material->firstPiolaKirchhoffStress(strainPerturbed_minus);

                // Compute numerical derivative
                const auto dP = (P_perturbed_plus - P_perturbed_minus) / (static_cast<Real>(2) * epsilon);

                // For each component of P
                for(sofa::Size k = 0; k < spatial_dimensions; ++k)
                {
                    for(sofa::Size l = 0; l < spatial_dimensions; ++l)
                    {
                        // Compare numerical and analytical derivatives
                        const Real numerical = dP(k,l);
                        const Real analytical = A(i, j, k, l);
                        EXPECT_NEAR(numerical, analytical, 0.25) << "i = " << i << " j = " << j << " k = " << k << " l = " << l;
                    }
                }
            }
        }
    }

    void testMajorSymmetry()
    {
        const auto F = this->generatePositiveDefiniteMatrix();
        Strain<DataTypes> strain(deformationGradient, F);
        const auto A = material->materialTangentModulus(strain);

        for(sofa::Size i = 0; i < spatial_dimensions; ++i)
        {
            for(sofa::Size j = 0; j < spatial_dimensions; ++j)
            {
                for(sofa::Size k = 0; k < spatial_dimensions; ++k)
                {
                    for(sofa::Size l = 0; l < spatial_dimensions; ++l)
                    {
                        EXPECT_NEAR(A(i, j, k, l), A(k, l, i, j), 1e-6);
                    }
                }
            }
        }
    }
};


using AllMaterials = ::testing::Types<
    StVenantKirchhoffMaterial<sofa::defaulttype::Vec3Types>,
    StVenantKirchhoffMaterial<sofa::defaulttype::Vec2Types>,
    StVenantKirchhoffMaterial<sofa::defaulttype::Vec1Types>
>;
TYPED_TEST_SUITE(MaterialTest, AllMaterials);




TYPED_TEST(MaterialTest, derivativePK1)
{
    this->testDerivativePK1();
}

TYPED_TEST(MaterialTest, tensorMajorSymmetry)
{
    this->testMajorSymmetry();
}

template <typename T>
class PK2Material : public T
{
public:
    SOFA_CLASS(PK2Material, T);
    using T::secondPiolaKirchhoffStress;
    using T::elasticityTensor;
    using T::strainEnergyDensity;
};

template <typename T>
class PK2MaterialTest : public BaseMaterialTest<PK2Material<T>>
{
public:
    using DataTypes = typename T::DataTypes;
    using Real = sofa::Real_t<DataTypes>;
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;

    void testMinorSymmetryPK2()
    {
        const auto F = this->generatePositiveDefiniteMatrix();
        Strain<DataTypes> strain(deformationGradient, F);
        const auto S = this->material->secondPiolaKirchhoffStress(strain);

        for(sofa::Size i = 0; i < spatial_dimensions; ++i)
        {
            for(sofa::Size j = 0; j < spatial_dimensions; ++j)
            {
                EXPECT_NEAR(S(i,j), S(j,i), 1e-6);
            }
        }
    }

    void testPK1PK2()
    {
        const auto F = this->generatePositiveDefiniteMatrix();
        Strain<DataTypes> strain(deformationGradient, F);

        const auto S = this->material->secondPiolaKirchhoffStress(strain);
        const auto P = this->material->firstPiolaKirchhoffStress(strain);

        const auto FS = F * S;

        for(sofa::Size i = 0; i < spatial_dimensions; ++i)
        {
            for(sofa::Size j = 0; j < spatial_dimensions; ++j)
            {
                EXPECT_NEAR(FS(i,j), P(i,j), 1e-6);
            }
        }
    }

    void testSymmetryElasticityTensor()
    {
        const auto F = this->generatePositiveDefiniteMatrix();
        Strain<DataTypes> strain(deformationGradient, F);
        const auto C = this->material->elasticityTensor(strain);

        for(sofa::Size i = 0; i < spatial_dimensions; ++i)
        {
            for(sofa::Size j = 0; j < spatial_dimensions; ++j)
            {
                for(sofa::Size k = 0; k < spatial_dimensions; ++k)
                {
                    for(sofa::Size l = 0; l < spatial_dimensions; ++l)
                    {
                        EXPECT_NEAR(C(i, j, k, l), C(k, l, i, j), 1e-6);
                        EXPECT_NEAR(C(i, j, k, l), C(j, i, k, l), 1e-6);
                        EXPECT_NEAR(C(i, j, k, l), C(i, j, l, k), 1e-6);
                    }
                }
            }
        }
    }

    void testDerivativePK2()
    {
        const auto F = this->generatePositiveDefiniteMatrix();
        Strain<DataTypes> strain(deformationGradient, F);

        const auto& C = strain.getRightCauchyGreenTensor();

        const auto S = this->material->secondPiolaKirchhoffStress(strain);
        const auto elasticityTensor = this->material->elasticityTensor(strain);

        // Small perturbation for finite difference
        constexpr Real epsilon = 1e-8;

        // For each component of C
        for(sofa::Size k = 0; k < spatial_dimensions; ++k)
        {
            for(sofa::Size l = 0; l < spatial_dimensions; ++l)
            {
                auto C_perturbed_plus = C;
                auto C_perturbed_minus = C;

                C_perturbed_plus(k, l) += epsilon;
                C_perturbed_minus(k, l) -= epsilon;

                if (k != l)
                {
                    C_perturbed_plus(l, k) += epsilon;
                    C_perturbed_minus(l, k) -= epsilon;
                }

                // Compute perturbed stress
                Strain<DataTypes> strainPerturbed_plus(rightCauchyGreenTensor, C_perturbed_plus);
                const auto S_perturbed_plus = this->material->secondPiolaKirchhoffStress(strainPerturbed_plus);

                Strain<DataTypes> strainPerturbed_minus(rightCauchyGreenTensor, C_perturbed_minus);
                const auto S_perturbed_minus = this->material->secondPiolaKirchhoffStress(strainPerturbed_minus);

                // Compute numerical derivative
                const auto dSdC = (S_perturbed_plus - S_perturbed_minus) / (2. * (1. + (k != l)) * epsilon);
                const auto dSdE = static_cast<Real>(2) * dSdC;

                // For each component of S
                for(sofa::Size i = 0; i < spatial_dimensions; ++i)
                {
                    for(sofa::Size j = 0; j < spatial_dimensions; ++j)
                    {
                        // Compare numerical and analytical derivatives
                        const Real numerical = dSdE(i,j);
                        const Real analytical = elasticityTensor(i, j, k, l);
                        EXPECT_NEAR(numerical, analytical, 1.0) << "i = " << i << " j = " << j << " k = " << k << " l = " << l;
                    }
                }
            }
        }
    }

    void testDerivativeStrainEnergy()
    {
        const auto F = this->generatePositiveDefiniteMatrix();
        Strain<DataTypes> strain(deformationGradient, F);

        const auto P = this->material->firstPiolaKirchhoffStress(strain);

        constexpr Real epsilon = 1e-8;

        for(sofa::Size i = 0; i < spatial_dimensions; ++i)
        {
            for(sofa::Size j = 0; j < spatial_dimensions; ++j)
            {
                auto F_perturbed_plus = F;
                F_perturbed_plus(i,j) += epsilon;

                auto F_perturbed_minus = F;
                F_perturbed_minus(i,j) -= epsilon;

                Strain<DataTypes> strainPerturbed_plus(deformationGradient, F_perturbed_plus);
                Strain<DataTypes> strainPerturbed_minus(deformationGradient, F_perturbed_minus);

                const Real W_perturbed_plus = this->material->strainEnergyDensity(strainPerturbed_plus);
                const Real W_perturbed_minus = this->material->strainEnergyDensity(strainPerturbed_minus);

                // Compare numerical and analytical derivatives
                const Real numerical = (W_perturbed_plus - W_perturbed_minus) / (static_cast<Real>(2) * epsilon);
                const Real analytical = P(i, j);
                EXPECT_NEAR(numerical, analytical, 5e-4) << "i = " << i << " j = " << j;
            }
        }
    }
    
};

using PK2Materials = ::testing::Types<
    StVenantKirchhoffMaterial<sofa::defaulttype::Vec3Types>,
    StVenantKirchhoffMaterial<sofa::defaulttype::Vec2Types>,
    StVenantKirchhoffMaterial<sofa::defaulttype::Vec1Types>,
    NeoHookeanMaterial<sofa::defaulttype::Vec3Types>,
    NeoHookeanMaterial<sofa::defaulttype::Vec2Types>,
    NeoHookeanMaterial<sofa::defaulttype::Vec1Types>,
    IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec3Types>,
    IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec2Types>,
    IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec1Types>,
    MooneyRivlinMaterial<sofa::defaulttype::Vec3Types>,
    MooneyRivlinMaterial<sofa::defaulttype::Vec2Types>,
    MooneyRivlinMaterial<sofa::defaulttype::Vec1Types>,
    OgdenMaterial<sofa::defaulttype::Vec3Types>,
    OgdenMaterial<sofa::defaulttype::Vec2Types>,
    OgdenMaterial<sofa::defaulttype::Vec1Types>
>;
TYPED_TEST_SUITE(PK2MaterialTest, PK2Materials);

TYPED_TEST(PK2MaterialTest, symmetryElasticityTensor)
{
    this->testSymmetryElasticityTensor();
}

TYPED_TEST(PK2MaterialTest, minorSymmetryPK2)
{
    this->testMinorSymmetryPK2();
}

TYPED_TEST(PK2MaterialTest, derivativePK2)
{
    this->testDerivativePK2();
}

TYPED_TEST(PK2MaterialTest, PK1PK2)
{
    this->testPK1PK2();
}

TYPED_TEST(PK2MaterialTest, derivativeStrainEnergy)
{
    this->testDerivativeStrainEnergy();
}

}
