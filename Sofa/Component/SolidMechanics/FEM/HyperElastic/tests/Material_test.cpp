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

#include <gtest/gtest.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/BoyceAndArruda.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/Costa.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/NeoHookean.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/Ogden.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/StableNeoHookean.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/STVenantKirchhoff.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/VerondaWestman.h>
#include <sofa/testing/LinearCongruentialRandomGenerator.h>


namespace sofa
{

using namespace component::solidmechanics::fem::hyperelastic::material;

StrainInformation<defaulttype::Vec3Types>::MatrixSym
generatePositiveDefiniteMatrix(sofa::testing::LinearCongruentialRandomGenerator& lcg)
{
    using Real = defaulttype::Vec3Types::Coord::value_type;

    Eigen::Matrix<Real, 3, 3> A;
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            A(i, j) = lcg.generateInRange(-10., 10.);
        }
    }

    A = 0.5 * (A + A.transpose()); // Ensuring symmetry

    // Making the matrix positive definite
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, 3, 3> > eigensolver(A);
    const Eigen::Matrix<Real, 3, 1>& eigenvalues = eigensolver.eigenvalues();
    const Real min_eigval = eigenvalues.minCoeff();
    if (min_eigval <= 0)
    {
        A += Eigen::Matrix<Real, 3, 3>::Identity() * (-min_eigval + 1e-6); // Adding a small value to ensure positive definiteness
    }

    StrainInformation<defaulttype::Vec3Types>::MatrixSym W;
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i <= j; i++)
        {
            W(i, j) = (A(i, j) + A(j, i)) / 2.0;
        }
    }
    return W;
}

void computeStrainInformation(StrainInformation<defaulttype::Vec3Types>& strain)
{
    strain.trC = sofa::type::trace(strain.deformationTensor);

    //det(C)=J^2
    strain.J = std::sqrt(sofa::type::determinant(strain.deformationTensor));
}

SReal perturbedStrainEnergy(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters,
    const StrainInformation<defaulttype::Vec3Types>& strain_0,
    defaulttype::Vec3Types::Coord::value_type h,
    const sofa::Size i)
{
    StrainInformation<defaulttype::Vec3Types> strain = strain_0;

    //off-diagonal terms: division by 2 because of the symmetric nature of the
    //tensor data structure
    if (i != 0 && i != 2 && i != 5)
    {
        h /= 2;
    }

    strain.deformationTensor[i] += h;
    computeStrainInformation(strain);

    return material.getStrainEnergy(&strain, materialParameters);
}

StrainInformation<defaulttype::Vec3Types>::MatrixSym perturbedPK2(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters,
    const StrainInformation<defaulttype::Vec3Types>& strain_0,
    defaulttype::Vec3Types::Coord::value_type h,
    const sofa::Size i)
{
    StrainInformation<defaulttype::Vec3Types> strain = strain_0;

    //off-diagonal terms: division by 2 because of the symmetric nature of the
    //tensor data structure
    if (i != 0 && i != 2 && i != 5)
    {
        h /= 2;
    }

    strain.deformationTensor[i] += h;
    computeStrainInformation(strain);

    StrainInformation<defaulttype::Vec3Types>::MatrixSym PK2;
    material.deriveSPKTensor(&strain, materialParameters, PK2);
    return PK2;
}

void testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters)
{
    using Real = defaulttype::Vec3Types::Coord::value_type;

    StrainInformation<defaulttype::Vec3Types> strain;

    //random right Cauchy-Green deformation tensor
    sofa::testing::LinearCongruentialRandomGenerator lcg(96547);
    strain.deformationTensor = generatePositiveDefiniteMatrix(lcg);

    computeStrainInformation(strain);

    StrainInformation<defaulttype::Vec3Types>::MatrixSym PK2;
    material.deriveSPKTensor(&strain, materialParameters, PK2);

    static constexpr Real h = 1e-6;

    for (sofa::Size i = 0; i < StrainInformation<defaulttype::Vec3Types>::MatrixSym::size(); ++i)
    // for (sofa::Size i : {0, 2, 5}) //only the diagonal terms
    {
        const Real psiPlus =
            perturbedStrainEnergy(material, materialParameters, strain, h, i);

        const Real psiMinus =
            perturbedStrainEnergy(material, materialParameters, strain, -h, i);

        //approximation of dPsi/dC
        const Real centralDifference = (psiPlus - psiMinus) / (2 * h);

        //PK2 = 2 * dPsi/dC
        const Real PK2Approx = 2 * centralDifference;

        //compare the approximation of PK2 with PK2
        EXPECT_NEAR(PK2[i], PK2Approx, 1e-8) << "i = " << i;
    }
}

void testElasticityTensorFromSecondPiolaKirchhoff(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters)
{
    using Real = defaulttype::Vec3Types::Coord::value_type;
    using MatrixSym = StrainInformation<defaulttype::Vec3Types>::MatrixSym;
    using Matrix6 = HyperelasticMaterial<defaulttype::Vec3Types>::Matrix6;

    StrainInformation<defaulttype::Vec3Types> strain;

    //random right Cauchy-Green deformation tensor
    sofa::testing::LinearCongruentialRandomGenerator lcg(96547);
    strain.deformationTensor = generatePositiveDefiniteMatrix(lcg);

    computeStrainInformation(strain);

    Matrix6 elasticityTensor;
    material.ElasticityTensor(&strain, materialParameters, elasticityTensor);

    static constexpr Real h = 1e-6;

    for (sofa::Size i = 0; i < Matrix6::size(); ++i)
    {
        const MatrixSym pk2Plus =
                perturbedPK2(material, materialParameters, strain, h, i);

        const MatrixSym pk2Minus =
                perturbedPK2(material, materialParameters, strain, -h, i);

        //approximation of dPK2/dC
        const MatrixSym centralDifference = (pk2Plus - pk2Minus) / (2 * h);

        //ElasticityTensor = 2 * dPK2/dC
        MatrixSym elasticityTensorApprox = 2 * centralDifference;

        //compare the approximation of the elasticity tensor
        for (sofa::Size j = 0; j < Matrix6::size(); ++j)
        {
            // Off-diagonal terms are stored doubled; the tensor approximation must be scaled accordingly
            if (j == 1 || j == 3 || j == 4) elasticityTensorApprox[j] *= 2.;
            EXPECT_NEAR(elasticityTensor(i,j), elasticityTensorApprox[j], 1e-7) << "i = " << i << ", j = " << j;
        }
    }
}

void testApplyElasticityTensor(
    HyperelasticMaterial<defaulttype::Vec3Types>& material,
    const MaterialParameters<defaulttype::Vec3Types>& materialParameters)
{
    using MatrixSym = StrainInformation<defaulttype::Vec3Types>::MatrixSym;
    using Matrix6 = HyperelasticMaterial<defaulttype::Vec3Types>::Matrix6;

    StrainInformation<defaulttype::Vec3Types> strain;

    //random right Cauchy-Green deformation tensor
    sofa::testing::LinearCongruentialRandomGenerator lcg(96547);
    strain.deformationTensor = generatePositiveDefiniteMatrix(lcg);

    //another random-generated symmetric second-order tensor
    const StrainInformation<defaulttype::Vec3Types>::MatrixSym e =
            generatePositiveDefiniteMatrix(lcg);
    sofa::type::Vec<6, SReal> eVec;
    for (int i = 0; i < 6; ++i)
    {
        eVec[i] = e[i];
    }

    computeStrainInformation(strain);

    //first method to obtain the application of the elasticity tensor: matrix-vector product
    Matrix6 elasticityTensor;
    material.ElasticityTensor(&strain, materialParameters, elasticityTensor);

    const sofa::type::Vec<6, SReal> appliedTensor_1 = elasticityTensor * eVec;

    //second method to obtain the application of the elasticity tensor: matrix-free product
    MatrixSym appliedTensor_2;
    material.applyElasticityTensor(&strain, materialParameters, e, appliedTensor_2);

    for (int i = 0; i < 6; ++i)
    {
        EXPECT_NEAR(appliedTensor_1[i], 2 * appliedTensor_2[i], 1e-3) << "i = " << i;
    }
}

TEST(HyperelasticMaterial, PK2_StVenantKirchhoff)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    STVenantKirchhoff<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_NeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    NeoHookean<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_Ogden)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 2., 3., 2.};

    Ogden<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_ArrudaBoyce)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    BoyceAndArruda<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_VerondaWestman)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1., 1.};

    VerondaWestman<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_Costa)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1., 1., 1., 1., 1., 1., 1.};

    Costa<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}

TEST(HyperelasticMaterial, PK2_StableNeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    StableNeoHookean<defaulttype::Vec3Types> material{};
    testSecondPiolaKirchhoffFromStrainEnergyDensityFunction(material, materialParameters);
}



TEST(HyperelasticMaterial, ElasticityTensor_StVenantKirchhoff)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    STVenantKirchhoff<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}

TEST(HyperelasticMaterial, ElasticityTensor_NeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    NeoHookean<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}

TEST(HyperelasticMaterial, ElasticityTensor_Ogden)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 2., 3., 2.};

    Ogden<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}

TEST(HyperelasticMaterial, ElasticityTensor_StableNeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    StableNeoHookean<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}

TEST(HyperelasticMaterial, ElasticityTensor_ArrudaBoyce)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    BoyceAndArruda<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}

TEST(HyperelasticMaterial, ElasticityTensor_VerondaWestman)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1., 1.};

    VerondaWestman<defaulttype::Vec3Types> material{};
    testElasticityTensorFromSecondPiolaKirchhoff(material, materialParameters);
}





TEST(HyperelasticMaterial, applyElasticityTensor_StVenantKirchhoff)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    STVenantKirchhoff<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}

TEST(HyperelasticMaterial, applyElasticityTensor_NeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    NeoHookean<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}

TEST(HyperelasticMaterial, applyElasticityTensor_Ogden)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 2., 3., 2.};

    Ogden<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}

TEST(HyperelasticMaterial, applyElasticityTensor_StableNeoHookean)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    StableNeoHookean<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}

TEST(HyperelasticMaterial, applyElasticityTensor_ArrudaBoyce)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1.};

    BoyceAndArruda<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}

TEST(HyperelasticMaterial, applyElasticityTensor_VerondaWestman)
{
    MaterialParameters<defaulttype::Vec3Types> materialParameters;
    materialParameters.parameterArray = { 1., 1., 1.};

    VerondaWestman<defaulttype::Vec3Types> material{};
    testApplyElasticityTensor(material, materialParameters);
}


}
