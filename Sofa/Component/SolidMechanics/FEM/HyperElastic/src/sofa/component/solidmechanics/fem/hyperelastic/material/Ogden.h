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
        const Real fj = pow(sinfo->J, -alpha1/3_sreal);
        const Real logJSqr = pow(log(sinfo->J), 2_sreal);

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
        const Real aBy2 = alpha1*0.5_sreal;
        const Real trCaBy2 = pow(sinfo->Evalue[0], aBy2) +
                    pow(sinfo->Evalue[1], aBy2) +
                    pow(sinfo->Evalue[2], aBy2);

        const Real muByAlphaSqr = mu1 / (alpha1*alpha1);

        return fj*muByAlphaSqr*trCaBy2 - 3_sreal*muByAlphaSqr + k0*logJSqr/2_sreal;
    }

    void deriveSPKTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,MatrixSym &SPKTensorGeneral) override
    {
        const MatrixSym& C=sinfo->deformationTensor;
        const Real k0 = param.parameterArray[0];
        const Real mu1 = param.parameterArray[1];
        const Real alpha1 = param.parameterArray[2];
        const Real fj = pow(sinfo->J, -alpha1/3.0_sreal);

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
        const Real aBy2 = alpha1*0.5_sreal;
        const Real trCaBy2 = pow(Evalue[0], aBy2) + pow(Evalue[1], aBy2) + pow(Evalue[2], aBy2);

        // Transpose (also inverse) of the eigenvector matrix
        Matrix3 EigenBasis;
        for (auto m = 0; m < Evect.rows(); ++m)
            for (auto n = 0; n < Evect.cols(); ++n) EigenBasis(m, n) = Evect(m, n);
        
        // Construct C^(alpha1/2 - 1) from eigenbasis: V * D * V^T; D_i = lambda_i^(alpha1/2 - 1)
        const Real aBy2Minus1 = aBy2 - 1_sreal;
        const MatrixSym D = MatrixSym(pow(Evalue[0], aBy2Minus1), 0, pow(Evalue[1], aBy2Minus1), 0, 0, pow(Evalue[2], aBy2Minus1));
        const Matrix3 Ca = EigenBasis*D.SymMatMultiply(EigenBasis.transposed());
        MatrixSym CaBy2Minus1; 
        sofa::type::MatSym<3>::Mat2Sym(Ca, CaBy2Minus1);

        // Invert deformation tensor
        MatrixSym invC;
        invertMatrix(invC, C);

        SPKTensorGeneral = fj * mu1 / alpha1 * (CaBy2Minus1 + -1_sreal/3_sreal * trCaBy2 * invC) + k0*log(sinfo->J)*invC;
    }

    void applyElasticityTensor_old(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,
                               const MatrixSym& inputTensor, MatrixSym& outputTensor)
    {
        Real k0=param.parameterArray[0];
        Real mu1=param.parameterArray[1];
        Real alpha1=param.parameterArray[2];
        MatrixSym C=sinfo->deformationTensor;
        EigenMatrix CEigen;
        CEigen(0,0)=C[0]; CEigen(0,1)=C[1]; CEigen(1,0)=C[1]; CEigen(1,1)=C[2]; CEigen(1,2)=C[4]; CEigen(2,1)=C[4];
        CEigen(2,0)=C[3]; CEigen(0,2)=C[3]; CEigen(2,2)=C[5];
        Eigen::SelfAdjointEigenSolver<EigenMatrix> Vect(CEigen,true);
        EigenMatrix Evect=Vect.eigenvectors();
        CoordEigen Evalue=Vect.eigenvalues();

        Real trCalpha=pow(Evalue[0],alpha1/(Real)2)+pow(Evalue[1],alpha1/(Real)2)+pow(Evalue[2],alpha1/(Real)2);
        Matrix3 Pinverse;
        Pinverse(0,0)=Evect(0,0); Pinverse(1,1)=Evect(1,1); Pinverse(2,2)=Evect(2,2); Pinverse(0,1)=Evect(1,0); Pinverse(1,0)=Evect(0,1); Pinverse(2,0)=Evect(0,2);
        Pinverse(0,2)=Evect(2,0); Pinverse(2,1)=Evect(1,2); Pinverse(1,2)=Evect(2,1);
        MatrixSym Dalpha_1=MatrixSym(pow(Evalue[0],alpha1/(Real)2.0-(Real)1.0),0,pow(Evalue[1],alpha1/(Real)2.0-(Real)1.0),0,0,pow(Evalue[2],alpha1/(Real)2.0-(Real)1.0));
        MatrixSym Calpha_1; Matrix3 Ca;
        Ca=Pinverse.transposed()*Dalpha_1.SymMatMultiply(Pinverse);
        Calpha_1.Mat2Sym(Ca,Calpha_1);
        MatrixSym Dalpha_2=MatrixSym(pow(Evalue[0],alpha1/(Real)4.0-(Real)1.0),0,pow(Evalue[1],alpha1/(Real)4.0-(Real)1.0),0,0,pow(Evalue[2],alpha1/(Real)4.0-(Real)1.0));
        MatrixSym Calpha_2;
        Calpha_2.Mat2Sym(Pinverse.transposed()*Dalpha_2.SymMatMultiply(Pinverse),Calpha_2);
        MatrixSym inversematrix;
        invertMatrix(inversematrix,sinfo->deformationTensor);
        Real _trHCalpha_1=inputTensor[0]*Calpha_1[0]+inputTensor[2]*Calpha_1[2]+inputTensor[5]*Calpha_1[5]
                +2*inputTensor[1]*Calpha_1[1]+2*inputTensor[3]*Calpha_1[3]+2*inputTensor[4]*Calpha_1[4];
        Real _trHC=inputTensor[0]*inversematrix[0]+inputTensor[2]*inversematrix[2]+inputTensor[5]*inversematrix[5]
                +2*inputTensor[1]*inversematrix[1]+2*inputTensor[3]*inversematrix[3]+2*inputTensor[4]*inversematrix[4];
        //C-1HC-1 convert to sym matrix
        MatrixSym Firstmatrix;
        Firstmatrix.Mat2Sym(inversematrix.SymMatMultiply(inputTensor.SymSymMultiply(inversematrix)),Firstmatrix);
        MatrixSym Secondmatrix;
        Secondmatrix.Mat2Sym(Calpha_2.SymMatMultiply(inputTensor.SymSymMultiply(Calpha_2)),Secondmatrix);
        outputTensor =
                (_trHC*(-alpha1/(Real)6.0)*(-(Real)1.0/(Real)3.0*inversematrix*trCalpha+Calpha_1)+(Real)1.0/(Real)3.0*Firstmatrix*trCalpha-(Real)1.0/(Real)3.0*inversematrix*_trHCalpha_1*alpha1/(Real)2.0
                +(alpha1/(Real)2.0-(Real)1)*Secondmatrix) * (mu1/alpha1*pow(sinfo->J,-alpha1/(Real)3.0))
                +k0/(Real)2.0*_trHC*inversematrix-(Real)(k0*log(sinfo->J))*Firstmatrix;

    }

    void ElasticityTensor_old(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param, Matrix6& outputTensor)
    {
        Real k0=param.parameterArray[0];
        Real mu1=param.parameterArray[1];
        Real alpha1=param.parameterArray[2];
        MatrixSym C=sinfo->deformationTensor;
        EigenMatrix CEigen;
        CEigen(0,0)=C[0]; CEigen(0,1)=C[1]; CEigen(1,0)=C[1]; CEigen(1,1)=C[2]; CEigen(1,2)=C[4]; CEigen(2,1)=C[4];
        CEigen(2,0)=C[3]; CEigen(0,2)=C[3]; CEigen(2,2)=C[5];

        Eigen::SelfAdjointEigenSolver<EigenMatrix> Vect(CEigen,true);
        EigenMatrix Evect=Vect.eigenvectors();
        CoordEigen Evalue=Vect.eigenvalues();

        Real trCalpha=pow(Evalue[0],alpha1/(Real)2)+pow(Evalue[1],alpha1/(Real)2)+pow(Evalue[2],alpha1/(Real)2);
        Matrix3 Pinverse;
        Pinverse(0,0)=Evect(0,0); Pinverse(1,1)=Evect(1,1); Pinverse(2,2)=Evect(2,2); Pinverse(0,1)=Evect(1,0); Pinverse(1,0)=Evect(0,1); Pinverse(2,0)=Evect(0,2);
        Pinverse(0,2)=Evect(2,0); Pinverse(2,1)=Evect(1,2); Pinverse(1,2)=Evect(2,1);
        MatrixSym Dalpha_1=MatrixSym(pow(Evalue[0],alpha1/(Real)2.0-(Real)1.0),0,pow(Evalue[1],alpha1/(Real)2.0-(Real)1.0),0,0,pow(Evalue[2],alpha1/(Real)2.0-(Real)1.0));
        MatrixSym Calpha_1; Matrix3 Ca;
        Ca=Pinverse.transposed()*Dalpha_1.SymMatMultiply(Pinverse);
        Calpha_1.Mat2Sym(Ca,Calpha_1);
        MatrixSym Dalpha_2=MatrixSym(pow(Evalue[0],alpha1/(Real)4.0-(Real)1.0),0,pow(Evalue[1],alpha1/(Real)4.0-(Real)1.0),0,0,pow(Evalue[2],alpha1/(Real)4.0-(Real)1.0));
        MatrixSym Calpha_2;
        Calpha_2.Mat2Sym(Pinverse.transposed()*Dalpha_2.SymMatMultiply(Pinverse),Calpha_2);
        MatrixSym _C;
        invertMatrix(_C,sinfo->deformationTensor);

        MatrixSym CC;
        CC=_C;
        CC[1]+=_C[1]; CC[3]+=_C[3]; CC[4]+=_C[4];
        Matrix6 C_H_C;
        C_H_C(0,0)=_C[0]*_C[0]; C_H_C(1,1)=_C[1]*_C[1]+_C[0]*_C[2]; C_H_C(2,2)=_C[2]*_C[2]; C_H_C(3,3)=_C[3]*_C[3]+_C[0]*_C[5]; C_H_C(4,4)=_C[4]*_C[4]+_C[2]*_C[5];
        C_H_C(5,5)=_C[5]*_C[5];
        C_H_C(1,0)=_C[0]*_C[1]; C_H_C(0,1)=2*C_H_C(1,0);
        C_H_C(2,0)=C_H_C(0,2)=_C[1]*_C[1]; C_H_C(5,0)=C_H_C(0,5)=_C[3]*_C[3];
        C_H_C(3,0)=_C[0]*_C[3]; C_H_C(0,3)=2*C_H_C(3,0); C_H_C(4,0)=_C[1]*_C[3]; C_H_C(0,4)=2*C_H_C(4,0);
        C_H_C(1,2)=_C[2]*_C[1]; C_H_C(2,1)=2*C_H_C(1,2); C_H_C(1,5)=_C[3]*_C[4]; C_H_C(5,1)=2*C_H_C(1,5);
        C_H_C(3,1)=C_H_C(1,3)=_C[0]*_C[4]+_C[1]*_C[3]; C_H_C(1,4)=C_H_C(4,1)=_C[1]*_C[4]+_C[2]*_C[3];
        C_H_C(3,2)=_C[4]*_C[1]; C_H_C(2,3)=2*C_H_C(3,2); C_H_C(4,2)=_C[4]*_C[2]; C_H_C(2,4)=2*C_H_C(4,2);
        C_H_C(2,5)=C_H_C(5,2)=_C[4]*_C[4];
        C_H_C(3,5)=_C[3]*_C[5]; C_H_C(5,3)=2*C_H_C(3,5);
        C_H_C(4,3)=C_H_C(3,4)=_C[3]*_C[4]+_C[5]*_C[1];
        C_H_C(4,5)=_C[4]*_C[5]; C_H_C(5,4)=2*C_H_C(4,5);
        Matrix6 trC_HC_;
        trC_HC_(0)=_C[0]*CC;
        trC_HC_(1)=_C[1]*CC;
        trC_HC_(2)=_C[2]*CC;
        trC_HC_(3)=_C[3]*CC;
        trC_HC_(4)=_C[4]*CC;
        trC_HC_(5)=_C[5]*CC;
        Matrix6 Calpha_H_Calpha;
        Calpha_H_Calpha(0,0)=Calpha_2[0]*Calpha_2[0]; Calpha_H_Calpha(1,1)=Calpha_2[1]*Calpha_2[1]+Calpha_2[0]*Calpha_2[2]; Calpha_H_Calpha(2,2)=Calpha_2[2]*Calpha_2[2]; Calpha_H_Calpha(3,3)=Calpha_2[3]*Calpha_2[3]+Calpha_2[0]*Calpha_2[5]; Calpha_H_Calpha(4,4)=Calpha_2[4]*Calpha_2[4]+Calpha_2[2]*Calpha_2[5];
        Calpha_H_Calpha(5,5)=Calpha_2[5]*Calpha_2[5];
        Calpha_H_Calpha(1,0)=Calpha_2[0]*Calpha_2[1]; Calpha_H_Calpha(0,1)=2*Calpha_H_Calpha(1,0);
        Calpha_H_Calpha(2,0)=Calpha_H_Calpha(0,2)=Calpha_2[1]*Calpha_2[1]; Calpha_H_Calpha(5,0)=Calpha_H_Calpha(0,5)=Calpha_2[3]*Calpha_2[3];
        Calpha_H_Calpha(3,0)=Calpha_2[0]*Calpha_2[3]; Calpha_H_Calpha(0,3)=2*Calpha_H_Calpha(3,0); Calpha_H_Calpha(4,0)=Calpha_2[1]*Calpha_2[3]; Calpha_H_Calpha(0,4)=2*Calpha_H_Calpha(4,0);
        Calpha_H_Calpha(1,2)=Calpha_2[2]*Calpha_2[1]; Calpha_H_Calpha(2,1)=2*Calpha_H_Calpha(1,2); Calpha_H_Calpha(1,5)=Calpha_2[3]*Calpha_2[4]; Calpha_H_Calpha(5,1)=2*Calpha_H_Calpha(1,5);
        Calpha_H_Calpha(3,1)=Calpha_H_Calpha(1,3)=Calpha_2[0]*Calpha_2[4]+Calpha_2[1]*Calpha_2[3]; Calpha_H_Calpha(1,4)=Calpha_H_Calpha(4,1)=Calpha_2[1]*Calpha_2[4]+Calpha_2[2]*Calpha_2[3];
        Calpha_H_Calpha(3,2)=Calpha_2[4]*Calpha_2[1]; Calpha_H_Calpha(2,3)=2*Calpha_H_Calpha(3,2); Calpha_H_Calpha(4,2)=Calpha_2[4]*Calpha_2[2]; Calpha_H_Calpha(2,4)=2*Calpha_H_Calpha(4,2);
        Calpha_H_Calpha(2,5)=Calpha_H_Calpha(5,2)=Calpha_2[4]*Calpha_2[4];
        Calpha_H_Calpha(3,5)=Calpha_2[3]*Calpha_2[5]; Calpha_H_Calpha(5,3)=2*Calpha_H_Calpha(3,5);
        Calpha_H_Calpha(4,3)=Calpha_H_Calpha(3,4)=Calpha_2[3]*Calpha_2[4]+Calpha_2[5]*Calpha_2[1];
        Calpha_H_Calpha(4,5)=Calpha_2[4]*Calpha_2[5]; Calpha_H_Calpha(5,4)=2*Calpha_H_Calpha(4,5);
        Matrix6 trCalpha_HC_;
        trCalpha_HC_[0]=Calpha_1[0]*CC;
        trCalpha_HC_[1]=Calpha_1[1]*CC;
        trCalpha_HC_[2]=Calpha_1[2]*CC;
        trCalpha_HC_[3]=Calpha_1[3]*CC;
        trCalpha_HC_[4]=Calpha_1[4]*CC;
        trCalpha_HC_[5]=Calpha_1[5]*CC;
        MatrixSym CCalpha_1;
        CCalpha_1=Calpha_1;
        CCalpha_1[1]+=Calpha_1[1]; CCalpha_1[3]+=Calpha_1[3]; CCalpha_1[4]+=Calpha_1[4];
        Matrix6 trC_HCalpha;
        trC_HCalpha[0]=_C[0]*CCalpha_1;
        trC_HCalpha[1]=_C[1]*CCalpha_1;
        trC_HCalpha[2]=_C[2]*CCalpha_1;
        trC_HCalpha[3]=_C[3]*CCalpha_1;
        trC_HCalpha[4]=_C[4]*CCalpha_1;
        trC_HCalpha[5]=_C[5]*CCalpha_1;

        outputTensor=(Real)2.0*(mu1/alpha1*pow(sinfo->J,-alpha1/(Real)3.0)*((-alpha1/(Real)6.0)*(-(Real)1.0/(Real)3.0*trC_HC_*trCalpha+trCalpha_HC_)+(Real)1.0/(Real)3.0*C_H_C*trCalpha-(Real)1.0/(Real)3.0*trC_HCalpha*alpha1/(Real)2.0
                +(alpha1/(Real)2.0-(Real)1)*Calpha_H_Calpha)+k0/(Real)2.0*trC_HC_-(Real)(k0*log(sinfo->J))*C_H_C);
    }

    void ElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param, Matrix6& outputTensor) override
    {
        const MatrixSym& C = sinfo->deformationTensor;
        const Real k0 = param.parameterArray[0];
        const Real mu1 = param.parameterArray[1];
        const Real alpha1 = param.parameterArray[2];
        const Real fj = pow(sinfo->J, -alpha1/3.0_sreal);

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
        const Real aBy2 = alpha1*0.5_sreal;
        const Real trCaBy2 = pow(Evalue[0], aBy2) + pow(Evalue[1], aBy2) + pow(Evalue[2], aBy2);

        // Transpose (also inverse) of the eigenvector matrix
        Matrix3 EigenBasis;
        for (auto m = 0; m < Evect.rows(); ++m)
            for (auto n = 0; n < Evect.cols(); ++n) EigenBasis(m, n) = Evect(m, n);

        // Construct C^(alpha1/2 - 1) from eigenbasis: V * D * V^T; D_i = lambda_i^(alpha1/2 - 1)
        const Real aBy2Minus1 = aBy2 - 1_sreal;
        MatrixSym D(pow(Evalue[0], aBy2Minus1), 0, pow(Evalue[1], aBy2Minus1), 0, 0, pow(Evalue[2], aBy2Minus1));
        MatrixSym CaBy2Minus1;
        sofa::type::MatSym<3>::Mat2Sym(EigenBasis*D.SymMatMultiply(EigenBasis.transposed()), CaBy2Minus1);

        // Invert deformation tensor
        MatrixSym invC;
        invertMatrix(invC, C);

        // Build the 4th-order tensor contribution in Voigt notation
        Matrix6 elasticityTensor;

        const Real aBy2Minus2 = aBy2 - 2_sreal;
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

                        elasticityTensor(m, n) += coefRot * 0.5_sreal *
                        (
                            Evect(i, eI) * Evect(j, eJ) * Evect(k, eJ) * Evect(l, eI) +
                            Evect(i, eI) * Evect(j, eJ) * Evect(k, eI) * Evect(l, eJ)
                        );
                    }
                }

                // SPK derivative contributions from isochoric part; product rule applies
                // Factor 1 - Directly differentiate F(J)
                elasticityTensor(m, n) -= alpha1/6_sreal * (invC(i,j) * CaBy2Minus1(k,l)
                    - trCaBy2 / 3_sreal * invC(i,j) * invC(k,l));

                // Factor 2 - 1st term - trace(C^(alpha1/2)) contribution
                elasticityTensor(m, n) -= alpha1/6_sreal * CaBy2Minus1(i,j) * invC(k,l);
                
                // Factor 2 - 2nd term - C inverse contribution
                elasticityTensor(m, n) += alpha1/3_sreal * trCaBy2 * 0.5_sreal * (
                    invC(i,k) * invC(j,l) 
                    + invC(i,l) * invC(j,k));

                // SPK derivative contribution from the volumetric part
                elasticityTensor(m, n) += 0.5_sreal * alpha1 / mu1 / fj * 
                    (
                        k0 * invC(i,j) * invC(k,l)
                        - k0*log(sinfo->J) *(invC(i,k) * invC(j,l) 
                            + invC(i,l) * invC(j,k))
                    ) ;
            }
        }

        outputTensor = 2_sreal * fj * mu1 / alpha1 * elasticityTensor ;

        // Adjust for Voigt notation using 2x factor on the off-diagonal
        for (sofa::Index m = 0; m < 6; m++)
        {
            outputTensor(m,1) *= 2_sreal;
            outputTensor(m,3) *= 2_sreal;
            outputTensor(m,4) *= 2_sreal;
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
