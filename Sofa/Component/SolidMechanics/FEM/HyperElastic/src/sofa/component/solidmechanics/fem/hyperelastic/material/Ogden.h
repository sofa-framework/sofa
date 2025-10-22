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

    Real getStrainEnergy(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param) override
    {
        MatrixSym C=sinfo->deformationTensor;
        Real k0=param.parameterArray[0];
        Real mu1=param.parameterArray[1];
        Real alpha1=param.parameterArray[2];
        Real fj= (Real)(pow(sinfo->J,(Real)(-alpha1/3.0)));
        Eigen::Matrix<Real,3,3> CEigen;
        CEigen(0,0)=C[0]; CEigen(0,1)=C[1]; CEigen(1,0)=C[1]; CEigen(1,1)=C[2];
        CEigen(1,2)=C[4]; CEigen(2,1)=C[4]; CEigen(2,0)=C[3]; CEigen(0,2)=C[3]; CEigen(2,2)=C[5];
        /*Eigen::SelfAdjointEigenSolver<EigenMatrix>*/Eigen::EigenSolver<Eigen::Matrix<Real, 3, 3> > EigenProblemSolver(CEigen,true);
        if (EigenProblemSolver.info() != Eigen::Success)
        {
            dmsg_warning("Ogden") << "EigenSolver iterations failed to converge";
            return 0.;
        }
        sinfo->Evalue = EigenProblemSolver.eigenvalues().real();
        sinfo->Evect = EigenProblemSolver.eigenvectors().real();

        //Real val=pow(sinfo->Evalue[0],alpha1/(Real)2)+pow(sinfo->Evalue[1],alpha1/(Real)2)+pow(sinfo->Evalue[2],alpha1/(Real)2);
        //return (Real)fj*val*mu1/(alpha1*alpha1)+k0*log(sinfo->J)*log(sinfo->J)/(Real)2.0-(Real)3.0*mu1/(alpha1*alpha1);
        Real trCalpha2 = pow(sinfo->Evalue[0], alpha1 / 2_sreal) +
                    pow(sinfo->Evalue[1], alpha1 / 2_sreal) +
                    pow(sinfo->Evalue[2], alpha1 / 2_sreal);
        return fj * mu1 / (alpha1 * alpha1) * trCalpha2
               - 3_sreal * mu1 / (alpha1 * alpha1)
               + k0 * log(sinfo->J) * log(sinfo->J) / 2_sreal;
    }

    void deriveSPKTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,MatrixSym &SPKTensorGeneral) override
    {
        Real k0=param.parameterArray[0];
        Real mu1=param.parameterArray[1];
        Real alpha1=param.parameterArray[2];
        MatrixSym C=sinfo->deformationTensor;
        Eigen::Matrix<Real,3,3> CEigen;
        CEigen(0,0)=C[0]; CEigen(0,1)=C[1]; CEigen(1,0)=C[1]; CEigen(1,1)=C[2]; CEigen(1,2)=C[4]; CEigen(2,1)=C[4];
        CEigen(2,0)=C[3]; CEigen(0,2)=C[3]; CEigen(2,2)=C[5];

        /*Eigen::SelfAdjointEigenSolver<EigenMatrix>*/Eigen::EigenSolver<Eigen::Matrix<Real, 3, 3> > EigenProblemSolver(CEigen,true);
        if (EigenProblemSolver.info() != Eigen::Success)
        {
            dmsg_warning("Ogden") << "SelfAdjointEigenSolver iterations failed to converge";
            return;
        }
        EigenMatrix Evect=EigenProblemSolver.eigenvectors().real();
        CoordEigen Evalue=EigenProblemSolver.eigenvalues().real();

        Real trCalpha=pow(Evalue[0],alpha1/(Real)2)+pow(Evalue[1],alpha1/(Real)2)+pow(Evalue[2],alpha1/(Real)2);
        Matrix3 Pinverse;
        Pinverse(0,0)=Evect(0,0); Pinverse(1,1)=Evect(1,1); Pinverse(2,2)=Evect(2,2); Pinverse(0,1)=Evect(1,0); Pinverse(1,0)=Evect(0,1); Pinverse(2,0)=Evect(0,2);
        Pinverse(0,2)=Evect(2,0); Pinverse(2,1)=Evect(1,2); Pinverse(1,2)=Evect(2,1);
        MatrixSym Dalpha_1=MatrixSym(pow(Evalue[0],alpha1/(Real)2.0-(Real)1.0),0,pow(Evalue[1],alpha1/(Real)2.0-(Real)1.0),0,0,pow(Evalue[2],alpha1/(Real)2.0-(Real)1.0));
        MatrixSym Calpha_1; Matrix3 Ca;
        Ca=Pinverse.transposed()*Dalpha_1.SymMatMultiply(Pinverse);
        Calpha_1.Mat2Sym(Ca,Calpha_1);
        MatrixSym inversematrix;
        invertMatrix(inversematrix,sinfo->deformationTensor);
        //SPKTensorGeneral=(-(Real)1.0/(Real)3.0*trCalpha*inversematrix+Calpha_1)*(mu1/alpha1*pow(sinfo->J,-alpha1/(Real)3.0))+inversematrix*k0*log(sinfo->J);
        Real fj= (Real)(pow(sinfo->J,(Real)(-alpha1/3.0)));
        // Contributions to S from derivatives of strain energy w.r.t. C from 
        const MatrixSym partialLambda = 0.5 * Calpha_1; 
        const MatrixSym partialFJ = -1 / 6. * trCalpha * inversematrix;
        const MatrixSym partialLogJ = k0 * log(sinfo->J) * inversematrix;
        SPKTensorGeneral = 2. * fj * mu1 / alpha1 * (partialLambda + partialFJ) + partialLogJ;
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
        Real k0=param.parameterArray[0];
        Real mu1=param.parameterArray[1];
        Real alpha1=param.parameterArray[2];
        MatrixSym C=sinfo->deformationTensor;
        Eigen::Matrix<Real,3,3> CEigen;
        CEigen(0,0)=C[0]; CEigen(0,1)=C[1]; CEigen(1,0)=C[1]; CEigen(1,1)=C[2]; CEigen(1,2)=C[4]; CEigen(2,1)=C[4];
        CEigen(2,0)=C[3]; CEigen(0,2)=C[3]; CEigen(2,2)=C[5];

        /*Eigen::SelfAdjointEigenSolver<EigenMatrix>*/Eigen::EigenSolver<Eigen::Matrix<Real, 3, 3> > EigenProblemSolver(CEigen,true);
        if (EigenProblemSolver.info() != Eigen::Success)
        {
            dmsg_warning("Ogden") << "SelfAdjointEigenSolver iterations failed to converge";
            return;
        }
        EigenMatrix Evect=EigenProblemSolver.eigenvectors().real();
        CoordEigen Evalue=EigenProblemSolver.eigenvalues().real();

        Real trCalpha=pow(Evalue[0],alpha1/(Real)2)+pow(Evalue[1],alpha1/(Real)2)+pow(Evalue[2],alpha1/(Real)2);
        Matrix3 Pinverse;
        Pinverse(0,0)=Evect(0,0); Pinverse(1,1)=Evect(1,1); Pinverse(2,2)=Evect(2,2); Pinverse(0,1)=Evect(1,0); Pinverse(1,0)=Evect(0,1); Pinverse(2,0)=Evect(0,2);
        Pinverse(0,2)=Evect(2,0); Pinverse(2,1)=Evect(1,2); Pinverse(1,2)=Evect(2,1);

        MatrixSym Dalpha_1(
            pow(Evalue[0], alpha1/2.0 - 1.0), 0,
            pow(Evalue[1], alpha1/2.0 - 1.0), 0, 0,
            pow(Evalue[2], alpha1/2.0 - 1.0)
        );
        MatrixSym Calpha_1;
        Calpha_1.Mat2Sym(Pinverse.transposed()*Dalpha_1.SymMatMultiply(Pinverse),Calpha_1);
        
        MatrixSym Dalpha_2(
            pow(Evalue[0], alpha1/2.0 - 2.0), 0,
            pow(Evalue[1], alpha1/2.0 - 2.0), 0, 0,
            pow(Evalue[2], alpha1/2.0 - 2.0)
        );
        MatrixSym Calpha_2; 
        Calpha_2.Mat2Sym(Pinverse.transposed() * Dalpha_2.SymMatMultiply(Pinverse), Calpha_2);

        MatrixSym inversematrix;
        invertMatrix(inversematrix,sinfo->deformationTensor);

        // build 4th-order tensor contribution in Voigt notation
        Matrix6 elasticityTensor;

        // diagonal entries
        const Real coef = 0.5 * alpha1 - 1.;
        Real fj= (Real)(pow(sinfo->J,(Real)(-alpha1/3.0)));
        for (int n = 0; n < 6; n++)
        {
            int i, j;
            switch(n)
            {
                case 0: i=0; j=0; break;
                case 1: i=0; j=1; break;
                case 2: i=1; j=1; break;
                case 3: i=0; j=2; break;
                case 4: i=1; j=2; break;
                case 5: i=2; j=2; break;
            }
            for (int m = 0; m < 6; m++) 
            {
                int k, l;
                switch(m)
                {
                    case 0: k=0; l=0; break;
                    case 1: k=0; l=1; break;
                    case 2: k=1; l=1; break;
                    case 3: k=0; l=2; break;
                    case 4: k=1; l=2; break;
                    case 5: k=2; l=2; break;
                }
                // Simple Ogden summation
                for (int ii = 0 ; ii < 3; ii++)
                {
                    Real eigenTerm = pow(Evalue[ii], alpha1/2. - 2.);
                    elasticityTensor(n, m) += coef * eigenTerm 
                        * Evect(i, ii) * Evect(j, ii) * Evect(k, ii) * Evect(l, ii);

                    const Real eigenTerm_ii = pow(Evalue[ii], alpha1/2. - 1.);
                    for (int jj = 0 ; jj < 3; jj++)
                    {
                        if (jj == ii) continue;

                        Real coefRot{0};
                        if (std::fabs(Evalue[ii] - Evalue[jj]) < std::numeric_limits<Real>::epsilon()) 
                            coefRot = (alpha1 / 2. - 1.) * pow(Evalue[ii], alpha1 / 2. - 2.);
                        else
                        {
                            const Real eigenTerm_jj = pow(Evalue[jj], alpha1/2. - 1.);
                            coefRot = (eigenTerm_ii - eigenTerm_jj)/(Evalue[ii] - Evalue[jj]);
                        }
                        elasticityTensor(n, m) += coefRot * 0.5 *
                        (
                            Evect(i, ii) * Evect(j, jj) * Evect(k, jj) * Evect(l, ii) +
                            Evect(i, ii) * Evect(j, jj) * Evect(k, ii) * Evect(l, jj)
                        );
                    }
                }
                // F(J) term
                elasticityTensor(n, m) -= alpha1/6. * (inversematrix(i,j) * Calpha_1(k,l)
                    - trCalpha / 3. * inversematrix(i,j) * inversematrix(k,l));

                // T22 term 1
                elasticityTensor(n, m) -= alpha1/6. * Calpha_1(i,j) * inversematrix(k,l);
                
                // T22 term 2
                elasticityTensor(n, m) += alpha1/3. * trCalpha * 0.5 * (
                    inversematrix(i,k) * inversematrix(j,l) 
                    + inversematrix(i,l) * inversematrix(j,k));

                elasticityTensor(n, m) += 0.5 * alpha1 / mu1 / fj * 
                    (
                        k0 * inversematrix(i,j) * inversematrix(k,l)
                        - k0*log(sinfo->J) *(inversematrix(i,k) * inversematrix(j,l) 
                            + inversematrix(i,l) * inversematrix(j,k))
                    ) ;
            }
        }

        // multiply by scalar factor
        outputTensor = 2.0 * fj * mu1 / alpha1 * elasticityTensor ;
        for (int n = 0; n < 6; n++)
        {
            outputTensor(n,1) *= 2.;
            outputTensor(n,3) *= 2.;
            outputTensor(n,4) *= 2.;
        }
    }

    void applyElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,
                               const MatrixSym& inputTensor, MatrixSym& outputTensor) override
    {
        // For now, let's just multiply matrices using the ElasticityTensor explicitely
        Matrix6 elasticityTensor;
        this->ElasticityTensor(sinfo, param, elasticityTensor);
        auto temp = elasticityTensor * inputTensor;
        for (size_t i = 0; i < 6; i++) outputTensor[i] = temp[i]/2.;
    }
};

} // namespace sofa::component::solidmechanics::fem::hyperelastic::material
