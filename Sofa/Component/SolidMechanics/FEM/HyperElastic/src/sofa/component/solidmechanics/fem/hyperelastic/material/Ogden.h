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
private:
    // In SOFA Voigt order for 3D symmetric tensors is non-standard: xx, xy, yy, xz, yz, zz
    const std::array<sofa::Index, 3*3> toVoigt = {0, 1, 3, 1, 2, 4, 3, 4, 5};
    inline sofa::Index vId(sofa::Index i, sofa::Index j) {return toVoigt[i * 3 + j];}
    const std::array<std::tuple<sofa::Index, sofa::Index>, 6> fromVoigt =
    {
        std::make_tuple(0,0),
        std::make_tuple(0,1),
        std::make_tuple(1,1),
        std::make_tuple(0,2),
        std::make_tuple(1,2),
        std::make_tuple(2,2)
    };

public:
    static constexpr std::string_view Name = "Ogden";

    typedef typename DataTypes::Coord::value_type Real;
    typedef type::Mat<3,3,Real> Matrix3;
    typedef type::Mat<6,6,Real> Matrix6;
    typedef type::MatSym<3,Real> MatrixSym;
    typedef type::Vec<3,Real> Vect;
    typedef typename Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real,3,3> >::MatrixType EigenMatrix;
    typedef typename Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real,3,3> >::RealVectorType CoordEigen;

    Real getStrainEnergy(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param) override
    {
        const MatrixSym& C = sinfo->deformationTensor;
        const Real k0 = param.parameterArray[0];
        const Real mu1 = param.parameterArray[1];
        const Real alpha1 = param.parameterArray[2];
        const Real fj = pow(sinfo->J, -alpha1/3.);
        const Real logJSqr = pow(log(sinfo->J), 2.);

        // Solve eigen problem for C
        Eigen::Matrix<Real, 3, 3> CEigen;
        for (sofa::Index i = 0; i < 3; ++i)
            for (sofa::Index j = 0; j < 3; ++j) CEigen(i, j) = C[vId(i, j)];

        // Disable temporarilly until fixed /*Eigen::SelfAdjointEigenSolver<EigenMatrix>*/
        Eigen::EigenSolver<Eigen::Matrix<Real, 3, 3> > EigenProblemSolver(CEigen, true);
        if (EigenProblemSolver.info() != Eigen::Success)
        {
            dmsg_warning("Ogden") << "EigenSolver iterations failed to converge";
            return 0.;
        }
        sinfo->Evalue = EigenProblemSolver.eigenvalues().real();
        sinfo->Evect = EigenProblemSolver.eigenvectors().real();

        // trace of C^(alpha1/2)
        const Real aBy2 = alpha1*0.5;
        const Real trCaBy2 = pow(sinfo->Evalue[0], aBy2) +
                    pow(sinfo->Evalue[1], aBy2) +
                    pow(sinfo->Evalue[2], aBy2);

        const Real muByAlphaSqr = mu1 / (alpha1*alpha1);

        return fj*muByAlphaSqr*trCaBy2 - 3.*muByAlphaSqr + k0*logJSqr/2.;
    }

    void deriveSPKTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,MatrixSym &SPKTensorGeneral) override
    {
        const MatrixSym& C=sinfo->deformationTensor;
        const Real k0 = param.parameterArray[0];
        const Real mu1 = param.parameterArray[1];
        const Real alpha1 = param.parameterArray[2];
        const Real fj = pow(sinfo->J, -alpha1/3.0);

        // Solve eigen problem for C
        Eigen::Matrix<Real, 3, 3> CEigen;
        for (sofa::Index i = 0; i < 3; ++i)
            for (sofa::Index j = 0; j < 3; ++j) CEigen(i, j) = C[vId(i, j)];

        // Disable temporarilly until fixed /*Eigen::SelfAdjointEigenSolver<EigenMatrix>*/
        Eigen::EigenSolver<Eigen::Matrix<Real, 3, 3> > EigenProblemSolver(CEigen, true);
        if (EigenProblemSolver.info() != Eigen::Success)
        {
            dmsg_warning("Ogden") << "SelfAdjointEigenSolver iterations failed to converge";
            return;
        }
        const EigenMatrix Evect = EigenProblemSolver.eigenvectors().real(); // orthonormal eigenvectors
        const CoordEigen Evalue = EigenProblemSolver.eigenvalues().real();

        // trace of C^(alpha1/2)
        const Real aBy2 = alpha1*0.5;
        const Real trCaBy2 = pow(Evalue[0], aBy2) + pow(Evalue[1], aBy2) + pow(Evalue[2], aBy2);

        // Transpose (also inverse) of the eigenvector matrix
        Matrix3 EigenBasis;
        for (auto m = 0; m < Evect.rows(); ++m)
            for (auto n = 0; n < Evect.cols(); ++n) EigenBasis(m, n) = Evect(m, n);
        
        // Construct C^(alpha1/2 - 1) from eigenbasis: V * D * V^T; D_i = lambda_i^(alpha1/2 - 1)
        const Real aBy2Minus1 = aBy2 - 1.;
        const MatrixSym D = MatrixSym(pow(Evalue[0], aBy2Minus1), 0, pow(Evalue[1], aBy2Minus1), 0, 0, pow(Evalue[2], aBy2Minus1));
        const Matrix3 Ca = EigenBasis*D.SymMatMultiply(EigenBasis.transposed());
        MatrixSym CaBy2Minus1; 
        sofa::type::MatSym<3, Real>::Mat2Sym(Ca, CaBy2Minus1);

        // Invert deformation tensor
        MatrixSym invC;
        invertMatrix(invC, C);

        SPKTensorGeneral = fj * mu1 / alpha1 * (CaBy2Minus1 - invC * trCaBy2 * 1./3.) + invC * k0 * log(sinfo->J);
    }

    void ElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param, Matrix6& outputTensor) override
    {
        const MatrixSym& C = sinfo->deformationTensor;
        const Real k0 = param.parameterArray[0];
        const Real mu1 = param.parameterArray[1];
        const Real alpha1 = param.parameterArray[2];
        const Real fj = pow(sinfo->J, -alpha1/3.0);

        // Solve eigen problem for C
        Eigen::Matrix<Real, 3, 3> CEigen;
        for (sofa::Index i = 0; i < 3; ++i)
            for (sofa::Index j = 0; j < 3; ++j) CEigen(i, j) = C[vId(i, j)];

        // Disable temporarilly until fixed /*Eigen::SelfAdjointEigenSolver<EigenMatrix>*/
        Eigen::EigenSolver<Eigen::Matrix<Real, 3, 3> > EigenProblemSolver(CEigen, true);
        if (EigenProblemSolver.info() != Eigen::Success)
        {
            dmsg_warning("Ogden") << "SelfAdjointEigenSolver iterations failed to converge";
            return;
        }
        const EigenMatrix Evect = EigenProblemSolver.eigenvectors().real();
        const CoordEigen Evalue = EigenProblemSolver.eigenvalues().real();

        // trace of C^(alpha1/2)
        const Real aBy2 = alpha1*0.5;
        const Real trCaBy2 = pow(Evalue[0], aBy2) + pow(Evalue[1], aBy2) + pow(Evalue[2], aBy2);

        // Transpose (also inverse) of the eigenvector matrix
        Matrix3 EigenBasis;
        for (auto m = 0; m < Evect.rows(); ++m)
            for (auto n = 0; n < Evect.cols(); ++n) EigenBasis(m, n) = Evect(m, n);

        // Construct C^(alpha1/2 - 1) from eigenbasis: V * D * V^T; D_i = lambda_i^(alpha1/2 - 1)
        const Real aBy2Minus1 = aBy2 - 1.;
        MatrixSym D(pow(Evalue[0], aBy2Minus1), 0, pow(Evalue[1], aBy2Minus1), 0, 0, pow(Evalue[2], aBy2Minus1));
        MatrixSym CaBy2Minus1;
        sofa::type::MatSym<3, Real>::Mat2Sym(EigenBasis*D.SymMatMultiply(EigenBasis.transposed()), CaBy2Minus1);

        // Invert deformation tensor
        MatrixSym invC;
        invertMatrix(invC, C);

        // Build the 4th-order tensor contribution in Voigt notation
        Matrix6 elasticityTensor;

        const Real aBy2Minus2 = aBy2 - 2.;
        for (sofa::Index m = 0; m < 6; m++)
        {
            sofa::Index i, j;
            std::tie(i, j) = fromVoigt[m];

            for (sofa::Index n = 0; n < 6; n++) 
            {
                sofa::Index k, l;
                std::tie(k, l) = fromVoigt[n];

                // SPK derivative contribution from spectral part
                for (sofa::Index eI = 0 ; eI < 3; eI++)
                {
                    // Distortion term from differenting of eigenvalues
                    const Real evalPowI2 = pow(Evalue[eI], aBy2Minus2);
                    elasticityTensor(m, n) += aBy2Minus1 * evalPowI2 
                        * Evect(i, eI) * Evect(j, eI) * Evect(k, eI) * Evect(l, eI);

                    // Rotational part from differenting of eigenvectors
                    const Real evalPowI = pow(Evalue[eI], aBy2Minus1);
                    for (sofa::Index eJ = 0 ; eJ < 3; eJ++)
                    {
                        if (eJ == eI) continue;

                        Real coefRot{0};

                        if (std::fabs(Evalue[eI] - Evalue[eJ]) < std::numeric_limits<Real>::epsilon()) 
                            coefRot = aBy2Minus1 * evalPowI2;
                        else
                        {
                            const Real evalPowJ = pow(Evalue[eJ], aBy2Minus1);
                            coefRot = (evalPowI - evalPowJ)/(Evalue[eI] - Evalue[eJ]);
                        }

                        elasticityTensor(m, n) += coefRot * 0.5 *
                        (
                            Evect(i, eI) * Evect(j, eJ) * Evect(k, eJ) * Evect(l, eI) +
                            Evect(i, eI) * Evect(j, eJ) * Evect(k, eI) * Evect(l, eJ)
                        );
                    }
                }

                // SPK derivative contributions from isochoric part; product rule applies
                // Factor 1 - Directly differentiate F(J)
                elasticityTensor(m, n) -= alpha1/6. * (invC(i,j) * CaBy2Minus1(k,l)
                    - trCaBy2 / 3. * invC(i,j) * invC(k,l));

                // Factor 2 - 1st term - trace(C^(alpha1/2)) contribution
                elasticityTensor(m, n) -= alpha1/6. * CaBy2Minus1(i,j) * invC(k,l);
                
                // Factor 2 - 2nd term - C inverse contribution
                elasticityTensor(m, n) += alpha1/3. * trCaBy2 * 0.5 * (
                    invC(i,k) * invC(j,l) 
                    + invC(i,l) * invC(j,k));

                // SPK derivative contribution from the volumetric part
                elasticityTensor(m, n) += 0.5 * alpha1 / mu1 / fj * 
                    (
                        k0 * invC(i,j) * invC(k,l)
                        - k0*log(sinfo->J) *(invC(i,k) * invC(j,l) 
                            + invC(i,l) * invC(j,k))
                    ) ;
            }
        }

        outputTensor = 2. * fj * mu1 / alpha1 * elasticityTensor ;

        // Adjust for Voigt notation using 2x factor on the off-diagonal
        for (sofa::Index m = 0; m < 6; m++)
        {
            outputTensor(m,1) *= 2.;
            outputTensor(m,3) *= 2.;
            outputTensor(m,4) *= 2.;
        }
    }

    void applyElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,
                               const MatrixSym& inputTensor, MatrixSym& outputTensor) override
    {
        // For now, let's just multiply matrices using the ElasticityTensor explicitly
        Matrix6 elasticityTensor;
        this->ElasticityTensor(sinfo, param, elasticityTensor);
        auto temp = elasticityTensor * inputTensor;
        for (size_t i = 0; i < 6; i++) outputTensor[i] = temp[i]/2.;
    }
};

} // namespace sofa::component::solidmechanics::fem::hyperelastic::material
