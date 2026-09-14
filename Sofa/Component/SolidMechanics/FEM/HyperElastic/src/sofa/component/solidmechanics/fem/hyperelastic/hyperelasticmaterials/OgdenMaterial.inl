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
#pragma once

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/OgdenMaterial.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/PK2HyperelasticMaterial.inl>
#include <Eigen/Eigenvalues>
#include <iomanip>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
OgdenMaterial<DataTypes>::OgdenMaterial()
    : m_mu(initData(&m_mu, static_cast<Real>(1e2), "mu",
                      "Material constant relevant to the shear modulus"))
    , m_alpha(initData(&m_alpha, static_cast<Real>(1.5), "alpha",
                      "Material constant exponent related to strain-stiffening"))
    , m_kappa(initData(&m_kappa, static_cast<Real>(1e3), "kappa",
                      "Material constant related to the bulk modulus"))
{}

template <class DataTypes>
auto OgdenMaterial<DataTypes>::secondPiolaKirchhoffStress(Strain<DataTypes>& strain) -> StressTensor
{
    using EigenMatrix = Eigen::Matrix<Real, spatial_dimensions, spatial_dimensions>;

    const auto& C = strain.getRightCauchyGreenTensor();
    const auto C_1 = sofa::type::inverse(C);

    const auto J = strain.getDeterminantDeformationGradient();
    assert(J > 0);

    const auto S_isochoric = [this, J, &strain, &C_1, &C] ()
    {
        const Real mu = m_mu.getValue();
        const Real alpha = m_alpha.getValue();

        const Real FJ = pow(J, -alpha / static_cast<Real>(spatial_dimensions));

        EigenMatrix CEigen;
        for (sofa::Index m = 0; m < spatial_dimensions; ++m)
            for (sofa::Index n = 0; n < spatial_dimensions; ++n) 
                CEigen(m, n) = C(m, n);

        Eigen::SelfAdjointEigenSolver<EigenMatrix> EigenProblemSolver(CEigen, Eigen::ComputeEigenvectors);
        if (EigenProblemSolver.info() != Eigen::Success)
            dmsg_warning("OgdenMaterial") << "EigenSolver iterations failed to converge";

        const auto eigenvectors = EigenProblemSolver.eigenvectors();
        const auto eigenvalues = EigenProblemSolver.eigenvalues();

        // Precompute trace of C^(alpha/2) and eigenvalue powers: lambda^(alpha/2 - 1)
        const Real aBy2 = static_cast<Real>(0.5) * alpha;
        const Real aBy2Minus1 = aBy2 - static_cast<Real>(1);
        Real trCaBy2{static_cast<Real>(0)};
        typename DataTypes::Coord eigenvaluePowers{};
        for (sofa::Index n = 0; n < spatial_dimensions; ++n)
        {
            trCaBy2 += pow(eigenvalues(n), aBy2);
            eigenvaluePowers(n) = pow(eigenvalues(n), aBy2Minus1);
        }

        const Real coefS1 = FJ * mu / (alpha * static_cast<Real>(2));
        const Real coefS2 = -FJ * mu / (alpha * static_cast<Real>(6)) * trCaBy2;

        StressTensor S{};

        for (sofa::Index i = 0; i < spatial_dimensions; ++i)
        {
            for (sofa::Index j = 0; j < spatial_dimensions; ++j)
            {
                for (sofa::Index n = 0; n < spatial_dimensions; ++n)
                    S(i,j) += coefS1 * eigenvaluePowers(n) * eigenvectors(i,n) * eigenvectors(j,n);
                S(i,j) += coefS2 * C_1(j, i);
            }
        }

        return static_cast<Real>(2) * S;
    }();

    const auto S_volumetric = [this, J, &C_1]()
    {
        const Real kappa = m_kappa.getValue();
        return kappa * log(J) * C_1;
    }();

    return S_isochoric + S_volumetric;
}

template <class DataTypes>
auto OgdenMaterial<DataTypes>::elasticityTensor(Strain<DataTypes>& strain) -> ElasticityTensor
{
    using EigenMatrix = Eigen::Matrix<Real, spatial_dimensions, spatial_dimensions>;

    const Real mu = m_mu.getValue();
    const Real alpha = m_alpha.getValue();
    const Real kappa = m_kappa.getValue();

    const auto& C = strain.getRightCauchyGreenTensor();
    const auto C_1 = sofa::type::inverse(C);

    const Real J = strain.getDeterminantDeformationGradient();
    assert(J > 0);
    const Real FJ = pow(J, -alpha / static_cast<Real>(3));

    EigenMatrix CEigen;
    for (sofa::Index m = 0; m < spatial_dimensions; ++m)
        for (sofa::Index n = 0; n < spatial_dimensions; ++n) 
            CEigen(m, n) = C(m, n);

        Eigen::SelfAdjointEigenSolver<EigenMatrix> EigenProblemSolver(CEigen, Eigen::ComputeEigenvectors);
    if (EigenProblemSolver.info() != Eigen::Success)
        dmsg_warning("OgdenMaterial") << "EigenSolver iterations failed to converge";
    const auto eigenvectors = EigenProblemSolver.eigenvectors();
    const auto eigenvalues = EigenProblemSolver.eigenvalues();

    // Precompute trace of C^(alpha/2) and eigenvalue powers: lambda^(a/2 - 1) and lambda^(a/2 - 2)
    const Real aBy2 = static_cast<Real>(0.5) * alpha;
    const Real aBy2Minus1 = aBy2 - static_cast<Real>(1);
    Real trCaBy2{static_cast<Real>(0)};
    typename DataTypes::Coord eigenvaluePowers1{}, eigenvaluePowers2{};
    for (sofa::Index n = 0; n < spatial_dimensions; ++n)
    {
        trCaBy2 += pow(eigenvalues(n), aBy2);
        eigenvaluePowers1(n) = pow(eigenvalues(n), aBy2Minus1);
        eigenvaluePowers2(n) = pow(eigenvalues(n), aBy2Minus1 - static_cast<Real>(1));
    }

    const Real coefSpectral = FJ * mu / alpha * aBy2Minus1;
    const Real coefRotational = FJ * mu / alpha;

    // Precompute C^(alpha1/2 - 1) from eigenbasis: V * D * V^T; D_i = lambda_i^(alpha1/2 - 1)
    RightCauchyGreenTensor D, EigenBasis;
    for (sofa::Index n = 0; n < spatial_dimensions; ++n)
    {
        for (sofa::Index m = 0; m < spatial_dimensions; ++m) 
            EigenBasis(n,m) = eigenvectors(n,m);
        D(n,n) = eigenvaluePowers1(n);
    }
    const RightCauchyGreenTensor CaBy2Minus1(EigenBasis*D*(EigenBasis.transposed()));

    return ElasticityTensor(
        [&](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
        {
            // Derivative of S_isochoric with respect to C
            const Real T_isochoric = [&]()
            {
                Real sum{static_cast<Real>(0)};

                // Spectral decomposition contributions; derivatives of eigenvalues and eigenvectors
                for (sofa::Index n = 0; n < spatial_dimensions; ++n)
                {
                    sum += coefSpectral * eigenvaluePowers2(n) * eigenvectors(i,n) * eigenvectors(j,n)
                         * eigenvectors(k,n) * eigenvectors(l,n);

                    for (sofa::Index m = 0; m < spatial_dimensions; ++m)
                    {
                        // This sum is taken over only for m != n
                        if (m == n) continue;
                        
                        // Eigenvalue multiplicity (val1 == val2) causes an indeterminate form;
                        // In that case, the limit of the expression is used
                        constexpr Real tol = static_cast<Real>(1e-8);
                        const Real maxEval = std::max(std::fabs(eigenvalues(n)), std::fabs(eigenvalues(m)));
                        const bool isDegenerate = std::fabs(eigenvalues(n) - eigenvalues(m)) < maxEval * tol;
                        const Real eigenvalueMultiplier = isDegenerate
                            ? aBy2Minus1 * eigenvaluePowers2(n) 
                            : (eigenvaluePowers1(n) - eigenvaluePowers1(m)) / (eigenvalues(n) - eigenvalues(m));

                        sum += static_cast<Real>(0.5) * coefRotational * eigenvalueMultiplier *
                                   (eigenvectors(i,n) * eigenvectors(j,m) * eigenvectors(k,m) * eigenvectors(l,n)
                                    + eigenvectors(i,n) * eigenvectors(j,m) * eigenvectors(k,n) * eigenvectors(l,m));
                    }
                }

                // Remaining terms grouped to highlight respective contribution from 
                // differentiating coefS1, coefS2 and C^(-1) from S_isochoric
                sum += FJ * mu * (
                        // Contribution from derivative of F(J)
                        trCaBy2 / static_cast<Real>(18) * C_1(j,i) * C_1(l,k) - 
                        // Contribution from derivatives of C^(alpha/2 - 1) and tr(C^(alpha/2))
                        (CaBy2Minus1(j,i) * C_1(l,k) + C_1(j,i) * CaBy2Minus1(k,l)) 
                        / static_cast<Real>(6) +
                        // Contribution from derivative of C^(-1) 
                        trCaBy2 / (static_cast<Real>(6) * alpha) 
                        * (C_1(j,k) * C_1(l,i) + C_1(j,l) * C_1(k,i))
                );

                return sum;
            }();

            // Derivative of S_volumetric with respect to C
            const Real T_volumetric = static_cast<Real>(0.5) * kappa * C_1(l,k) * C_1(j,i) 
                - static_cast<Real>(0.5) * kappa * log(J) 
                    * (C_1(j,k) * C_1(l,i) + C_1(j,l) * C_1(k,i));

            return static_cast<Real>(2) * (T_isochoric + T_volumetric);
        });
}

}  // namespace elasticity
