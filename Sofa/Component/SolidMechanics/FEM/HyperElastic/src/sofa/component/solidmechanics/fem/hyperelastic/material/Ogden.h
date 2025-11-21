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

#include <sofa/component/solidmechanics/fem/hyperelastic/config.h>


#include <sofa/component/solidmechanics/fem/hyperelastic/material/HyperelasticMaterial.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/type/MatSym.h>
#include <string>

#include <Eigen/QR>
#include <Eigen/Eigenvalues>

namespace sofa::component::solidmechanics::fem::hyperelastic::material
{

/** a Class that describe a generic hyperelastic material : example of Boyce and Arruda
The material is described based on continuum mechanics and the description is independent
to any discretization method like the finite element method.
A material is generically described by a strain energy function and its first and second derivatives.
In practice the energy is the sum of several energy terms which depends on 2 quantities :
the determinant of the deformation gradient J and the right Cauchy Green deformation tensor */



template<class DataTypes>
class Ogden: public HyperelasticMaterial<DataTypes>
{
public:
    static constexpr std::string_view Name = "Ogden";

    typedef typename DataTypes::Coord::value_type Real;
    typedef type::Mat<3,3,Real> Matrix3;
    typedef type::Mat<6,6,Real> Matrix6;
    typedef type::MatSym<3,Real> MatrixSym;
    typedef type::Vec<3,Real> Vect;
    typedef typename Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real,3,3> >::MatrixType EigenMatrix;
    typedef typename Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real,3,3> >::RealVectorType CoordEigen;

    MatrixSym m_CaBy2Minus1, m_invC; 
    Real m_trCaBy2{static_cast<Real>(0)}, m_FJ{static_cast<Real>(0)}, m_logJ{static_cast<Real>(0)};

    virtual void precomputeVariables(StrainInformation<DataTypes>* sinfo,
                                const MaterialParameters<DataTypes>& param)
    {
        const MatrixSym& C = sinfo->deformationTensor;
        const Real mu1 = param.parameterArray[0];
        const Real alpha1 = param.parameterArray[1];
        const Real k0 = param.parameterArray[2];

        m_FJ = pow(sinfo->J, -alpha1/static_cast<Real>(3));
        m_logJ = log(sinfo->J);

        // Solve eigen problem for C and store eigenvalues/vectors
        Eigen::Matrix<Real, 3, 3> CEigen;
        for (sofa::Index i = 0; i < 3; ++i)
            for (sofa::Index j = 0; j < 3; ++j) 
                CEigen(i, j) = C[MatrixSym::voigtID(i, j)];

        // 17/11/2025: Disable /*Eigen::SelfAdjointEigenSolver<EigenMatrix>*/
        // due to incorrect eigenvector computation for 3x3 matrices.
        Eigen::EigenSolver<Eigen::Matrix<Real, 3, 3> > EigenProblemSolver(CEigen, true);
        if (EigenProblemSolver.info() != Eigen::Success)
        {
            dmsg_warning("Ogden") << "EigenSolver iterations failed to converge";
            return;
        }

        sinfo->Evalue = EigenProblemSolver.eigenvalues().real().eval();
        sinfo->Evect = EigenProblemSolver.eigenvectors().real().eval();

        const Real aBy2{alpha1/static_cast<Real>(2)};
        m_trCaBy2 = static_cast<Real>(0);
        for (sofa::Index n = 0; n < sinfo->Evalue.rows(); ++n)
            m_trCaBy2 += pow(sinfo->Evalue[n], aBy2);

        // Transpose (also inverse) of the eigenvector matrix
        Matrix3 EigenBasis;
        for (auto m = 0; m < sinfo->Evect.rows(); ++m)
            for (auto n = 0; n < sinfo->Evect.cols(); ++n) 
                EigenBasis(m, n) = sinfo->Evect(m, n);
        
        // Construct C^(alpha1/2 - 1) from eigenbasis: V * D * V^T; D_i = lambda_i^(alpha1/2 - 1)
        const Real aBy2Minus1 = aBy2 - static_cast<Real>(1);
        const MatrixSym D = MatrixSym(pow(sinfo->Evalue[0], aBy2Minus1), 0, pow(sinfo->Evalue[1], aBy2Minus1), 0, 0, pow(sinfo->Evalue[2], aBy2Minus1));
        const Matrix3 Ca = EigenBasis*D.SymMatMultiply(EigenBasis.transposed());
        sofa::type::MatSym<3, Real>::Mat2Sym(Ca, m_CaBy2Minus1);

        // Invert deformation tensor
        invertMatrix(m_invC, C);
    }

    Real getStrainEnergy(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param) override
    {
        this->precomputeVariables(sinfo, param);

        const Real mu1 = param.parameterArray[0];
        const Real alpha1 = param.parameterArray[1];
        const Real k0 = param.parameterArray[2];
        const Real muByAlphaSqr = mu1 / pow(alpha1, static_cast<Real>(2));

        // Isochoric and volumetric parts 
        const Real Wiso = m_FJ * muByAlphaSqr * m_trCaBy2 - static_cast<Real>(3) * muByAlphaSqr;
        const Real Wvol = k0 * m_logJ * m_logJ / static_cast<Real>(2);

        return Wiso + Wvol;
    }

    void deriveSPKTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,MatrixSym &SPKTensorGeneral) override
    {
        this->precomputeVariables(sinfo, param);

        const Real mu1 = param.parameterArray[0];
        const Real alpha1 = param.parameterArray[1];
        const Real k0 = param.parameterArray[2];

        // trace of C^(alpha1/2)
        const Real aBy2 = alpha1/static_cast<Real>(2);

        // Siso = dWiso/dlambda*dlambda/dC + dWiso/dF*dF/dC
        const MatrixSym S_isochoric = m_CaBy2Minus1 * m_FJ * mu1 / alpha1
                                     -m_invC * m_FJ * mu1 / (static_cast<Real>(3)*alpha1) * m_trCaBy2;
        // Svol = dWvol/dC_{ij}
        const MatrixSym S_volumetric = m_invC * k0 * m_logJ;

        SPKTensorGeneral = S_isochoric + S_volumetric;
    }

    void ElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param, Matrix6& outputTensor) override
    {
        this->precomputeVariables(sinfo, param);

        const MatrixSym& C = sinfo->deformationTensor;
        const Real mu1 = param.parameterArray[0];
        const Real alpha1 = param.parameterArray[1];
        const Real k0 = param.parameterArray[2];

        // Coefficient terms
        const Real aBy2 = alpha1/static_cast<Real>(2);
        const Real aBy2Minus1 = aBy2 - static_cast<Real>(1);
        const Real aBy2Minus2 = aBy2 - static_cast<Real>(2);

        const CoordEigen& Evalue = sinfo->Evalue;
        const EigenMatrix& Evect = sinfo->Evect;

        // Build the 4th-order tensor contribution in Voigt notation
        Matrix6 elasticityTensor;

        // Loop over Voigt indices. Sum contributions from derivatives of SPK tensor w.r.t. C_{kl}
        // T1, T2, T3: Derivatives of S1, S2, S3 w.r.t. C_{kl}, respectively
        for (sofa::Index m = 0; m < 6; m++)
        {
            sofa::Index i, j;
            std::tie(i, j) = MatrixSym::fromVoigt[m];

            for (sofa::Index n = 0; n < 6; n++) 
            {
                sofa::Index k, l;
                std::tie(k, l) = MatrixSym::fromVoigt[n];

                // Derivative of S_isochoric; terms contributing from spectral decomposition
                for (sofa::Index eI = 0 ; eI < 3; eI++)
                {
                    // Variation of eigenvalues
                    const Real evalPowI2 = pow(Evalue[eI], aBy2Minus2);
                    elasticityTensor(m, n) += m_FJ * mu1 / alpha1 * aBy2Minus1 * evalPowI2 
                        * Evect(i, eI) * Evect(j, eI) * Evect(k, eI) * Evect(l, eI);

                    // Variation of eigenvectors
                    const Real evalPowI = pow(Evalue[eI], aBy2Minus1);
                    for (sofa::Index eJ = 0 ; eJ < 3; eJ++)
                    {
                        if (eJ == eI) continue;

                        const bool isDegenerate = std::fabs(Evalue[eI] - Evalue[eJ]) <
                                                   std::numeric_limits<Real>::epsilon();
                        const Real coefRot = isDegenerate 
                            ? aBy2Minus1 * evalPowI2
                            : (evalPowI - pow(Evalue[eJ], aBy2Minus1))/(Evalue[eI] - Evalue[eJ]);

                        elasticityTensor(m, n) += static_cast<Real>(0.5) * m_FJ * mu1 / alpha1 * coefRot *
                        (
                            Evect(i, eI) * Evect(j, eJ) * Evect(k, eJ) * Evect(l, eI) +
                            Evect(i, eI) * Evect(j, eJ) * Evect(k, eI) * Evect(l, eJ)
                        );
                    }
                }

                // Remaining terms from S_isochoric
                elasticityTensor(m, n) += 
                    // Lumped contributions from trace of C^(alpha1/2) and FJ in dWiso/dlambda*dlambda/dC
                    - m_FJ * mu1 / static_cast<Real>(6) * (m_CaBy2Minus1(i,j) * m_invC(l,k) + m_CaBy2Minus1(k,l) * m_invC(j,i))
                    // Contributions from FJ in dWiso/dF*dF/dC
                    + m_FJ * mu1 * m_trCaBy2 / static_cast<Real>(18) * m_invC(j,i) * m_invC(l,k)
                    // Contributions from inverse of C
                    + m_FJ * mu1 / (static_cast<Real>(6) * alpha1) * m_trCaBy2 * (m_invC(k,i) * m_invC(l,j) + m_invC(l,i) * m_invC(k,j));

                // Derivative of S_volumetric; dependence on lnJ and C^{-1}
                elasticityTensor(m, n) += static_cast<Real>(0.5) * k0 * m_invC(j,i) * m_invC(l,k)
                                - static_cast<Real>(0.5) * k0 * m_logJ * 
                    (m_invC(j,k) * m_invC(l,i) + m_invC(j,l) * m_invC(k,i));
            }
        }

        outputTensor = static_cast<Real>(2) * elasticityTensor ;

        // Adjust for Voigt notation using 2x factor on the off-diagonal
        for (sofa::Index m = 0; m < 6; m++)
        {
            outputTensor(m,1) *= static_cast<Real>(2);
            outputTensor(m,3) *= static_cast<Real>(2);
            outputTensor(m,4) *= static_cast<Real>(2);
        }
    }

    void applyElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,
                               const MatrixSym& inputTensor, MatrixSym& outputTensor) override
    {
        // For now, let's just multiply matrices using the ElasticityTensor explicitly
        Matrix6 elasticityTensor;
        this->ElasticityTensor(sinfo, param, elasticityTensor);
        auto temp = elasticityTensor * inputTensor;
        for (size_t i = 0; i < 6; i++) 
            outputTensor[i] = temp[i]/static_cast<Real>(2);
    }
};

} // namespace sofa::component::solidmechanics::fem::hyperelastic::material
