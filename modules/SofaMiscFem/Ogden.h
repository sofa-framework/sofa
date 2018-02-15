/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FEM_OGDEN_H
#define SOFA_COMPONENT_FEM_OGDEN_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif
#include <SofaMiscFem/HyperelasticMaterial.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/MatSym.h>
#include <string>

#include <Eigen/QR>
#include <Eigen/Eigenvalues>
namespace sofa
{

namespace component
{

namespace fem
{

/** a Class that describe a generic hyperelastic material : exemple of Boyce and Arruda
The material is described based on continuum mechanics and the description is independent
to any discretization method like the finite element method.
A material is generically described by a strain energy function and its first and second derivatives.
In practice the energy is the sum of several energy terms which depends on 2 quantities :
the determinant of the deformation gradient J and the right Cauchy Green deformation tensor */



template<class DataTypes>
class Ogden: public HyperelasticMaterial<DataTypes>
{

    typedef typename DataTypes::Coord::value_type Real;
    typedef defaulttype::Mat<3,3,Real> Matrix3;
    typedef defaulttype::Mat<6,6,Real> Matrix6;
    typedef defaulttype::MatSym<3,Real> MatrixSym;
    typedef defaulttype::Vec<3,Real> Vect;
    typedef typename Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real,3,3> >::MatrixType EigenMatrix;
    typedef typename Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real,3,3> >::RealVectorType CoordEigen;

    virtual Real getStrainEnergy(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param)
    {
        MatrixSym C=sinfo->deformationTensor;
        Real k0=param.parameterArray[0];
        Real mu1=param.parameterArray[1];
        Real alpha1=param.parameterArray[2];
        Real fj= (Real)(pow(sinfo->J,(Real)(-alpha1/3.0)));
        EigenMatrix CEigen;
        CEigen(0,0)=C[0]; CEigen(0,1)=C[1]; CEigen(1,0)=C[1]; CEigen(1,1)=C[2];
        CEigen(1,2)=C[4]; CEigen(2,1)=C[4]; CEigen(2,0)=C[3]; CEigen(0,2)=C[3]; CEigen(2,2)=C[5];
        Eigen::SelfAdjointEigenSolver<EigenMatrix> Vect(CEigen,true);
        sinfo->Evalue=Vect.eigenvalues();
        sinfo->Evect=Vect.eigenvectors();
        Real val=pow(sinfo->Evalue[0],alpha1/(Real)2)+pow(sinfo->Evalue[1],alpha1/(Real)2)+pow(sinfo->Evalue[2],alpha1/(Real)2);
        return (Real)fj*val*mu1/(alpha1*alpha1)+k0*log(sinfo->J)*log(sinfo->J)/(Real)2.0-(Real)3.0*mu1/(alpha1*alpha1);
    }

    virtual void deriveSPKTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,MatrixSym &SPKTensorGeneral)
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
        MatrixSym inversematrix;
        invertMatrix(inversematrix,sinfo->deformationTensor);
        SPKTensorGeneral=(-(Real)1.0/(Real)3.0*trCalpha*inversematrix+Calpha_1)*(mu1/alpha1*pow(sinfo->J,-alpha1/(Real)3.0))+inversematrix*k0*log(sinfo->J);
    }


    virtual void applyElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,
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

    virtual void ElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param, Matrix6& outputTensor)
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
        C_H_C[0][0]=_C[0]*_C[0]; C_H_C[1][1]=_C[1]*_C[1]+_C[0]*_C[2]; C_H_C[2][2]=_C[2]*_C[2]; C_H_C[3][3]=_C[3]*_C[3]+_C[0]*_C[5]; C_H_C[4][4]=_C[4]*_C[4]+_C[2]*_C[5];
        C_H_C[5][5]=_C[5]*_C[5];
        C_H_C[1][0]=_C[0]*_C[1]; C_H_C[0][1]=2*C_H_C[1][0];
        C_H_C[2][0]=C_H_C[0][2]=_C[1]*_C[1]; C_H_C[5][0]=C_H_C[0][5]=_C[3]*_C[3];
        C_H_C[3][0]=_C[0]*_C[3]; C_H_C[0][3]=2*C_H_C[3][0]; C_H_C[4][0]=_C[1]*_C[3]; C_H_C[0][4]=2*C_H_C[4][0];
        C_H_C[1][2]=_C[2]*_C[1]; C_H_C[2][1]=2*C_H_C[1][2]; C_H_C[1][5]=_C[3]*_C[4]; C_H_C[5][1]=2*C_H_C[1][5];
        C_H_C[3][1]=C_H_C[1][3]=_C[0]*_C[4]+_C[1]*_C[3]; C_H_C[1][4]=C_H_C[4][1]=_C[1]*_C[4]+_C[2]*_C[3];
        C_H_C[3][2]=_C[4]*_C[1]; C_H_C[2][3]=2*C_H_C[3][2]; C_H_C[4][2]=_C[4]*_C[2]; C_H_C[2][4]=2*C_H_C[4][2];
        C_H_C[2][5]=C_H_C[5][2]=_C[4]*_C[4];
        C_H_C[3][5]=_C[3]*_C[5]; C_H_C[5][3]=2*C_H_C[3][5];
        C_H_C[4][3]=C_H_C[3][4]=_C[3]*_C[4]+_C[5]*_C[1];
        C_H_C[4][5]=_C[4]*_C[5]; C_H_C[5][4]=2*C_H_C[4][5];
        Matrix6 trC_HC_;
        trC_HC_[0]=_C[0]*CC;
        trC_HC_[1]=_C[1]*CC;
        trC_HC_[2]=_C[2]*CC;
        trC_HC_[3]=_C[3]*CC;
        trC_HC_[4]=_C[4]*CC;
        trC_HC_[5]=_C[5]*CC;
        Matrix6 Calpha_H_Calpha;
        Calpha_H_Calpha[0][0]=Calpha_2[0]*Calpha_2[0]; Calpha_H_Calpha[1][1]=Calpha_2[1]*Calpha_2[1]+Calpha_2[0]*Calpha_2[2]; Calpha_H_Calpha[2][2]=Calpha_2[2]*Calpha_2[2]; Calpha_H_Calpha[3][3]=Calpha_2[3]*Calpha_2[3]+Calpha_2[0]*Calpha_2[5]; Calpha_H_Calpha[4][4]=Calpha_2[4]*Calpha_2[4]+Calpha_2[2]*Calpha_2[5];
        Calpha_H_Calpha[5][5]=Calpha_2[5]*Calpha_2[5];
        Calpha_H_Calpha[1][0]=Calpha_2[0]*Calpha_2[1]; Calpha_H_Calpha[0][1]=2*Calpha_H_Calpha[1][0];
        Calpha_H_Calpha[2][0]=Calpha_H_Calpha[0][2]=Calpha_2[1]*Calpha_2[1]; Calpha_H_Calpha[5][0]=Calpha_H_Calpha[0][5]=Calpha_2[3]*Calpha_2[3];
        Calpha_H_Calpha[3][0]=Calpha_2[0]*Calpha_2[3]; Calpha_H_Calpha[0][3]=2*Calpha_H_Calpha[3][0]; Calpha_H_Calpha[4][0]=Calpha_2[1]*Calpha_2[3]; Calpha_H_Calpha[0][4]=2*Calpha_H_Calpha[4][0];
        Calpha_H_Calpha[1][2]=Calpha_2[2]*Calpha_2[1]; Calpha_H_Calpha[2][1]=2*Calpha_H_Calpha[1][2]; Calpha_H_Calpha[1][5]=Calpha_2[3]*Calpha_2[4]; Calpha_H_Calpha[5][1]=2*Calpha_H_Calpha[1][5];
        Calpha_H_Calpha[3][1]=Calpha_H_Calpha[1][3]=Calpha_2[0]*Calpha_2[4]+Calpha_2[1]*Calpha_2[3]; Calpha_H_Calpha[1][4]=Calpha_H_Calpha[4][1]=Calpha_2[1]*Calpha_2[4]+Calpha_2[2]*Calpha_2[3];
        Calpha_H_Calpha[3][2]=Calpha_2[4]*Calpha_2[1]; Calpha_H_Calpha[2][3]=2*Calpha_H_Calpha[3][2]; Calpha_H_Calpha[4][2]=Calpha_2[4]*Calpha_2[2]; Calpha_H_Calpha[2][4]=2*Calpha_H_Calpha[4][2];
        Calpha_H_Calpha[2][5]=Calpha_H_Calpha[5][2]=Calpha_2[4]*Calpha_2[4];
        Calpha_H_Calpha[3][5]=Calpha_2[3]*Calpha_2[5]; Calpha_H_Calpha[5][3]=2*Calpha_H_Calpha[3][5];
        Calpha_H_Calpha[4][3]=Calpha_H_Calpha[3][4]=Calpha_2[3]*Calpha_2[4]+Calpha_2[5]*Calpha_2[1];
        Calpha_H_Calpha[4][5]=Calpha_2[4]*Calpha_2[5]; Calpha_H_Calpha[5][4]=2*Calpha_H_Calpha[4][5];
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

};


} // namespace fem

} // namespace component

} // namespace sofa

#endif
