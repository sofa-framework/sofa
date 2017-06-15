/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FEM_COSTA_H
#define SOFA_COMPONENT_FEM_COSTA_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif
#include <SofaMiscFem/initMiscFEM.h>
#include <SofaMiscFem/HyperelasticMaterial.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <string>


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
class Costa: public HyperelasticMaterial<DataTypes>{

  typedef typename DataTypes::Coord::value_type Real;
  typedef defaulttype::Mat<3,3,Real> Matrix3;
  typedef defaulttype::Mat<6,6,Real> Matrix6;
  typedef defaulttype::MatSym<3,Real> MatrixSym;
 
  public:

	 virtual Real getStrainEnergy(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param) {
		Real a=param.parameterArray[0];
		Real k0=param.parameterArray[1];
		Real bff=param.parameterArray[2];
		Real bfs=param.parameterArray[3];
		Real bss=param.parameterArray[4];
		Real bfn=param.parameterArray[5];
		Real bsn=param.parameterArray[6];
		Real bnn=param.parameterArray[7];
		MatrixSym Id;
				Id.identity();
				MatrixSym E=(sinfo->deformationTensor-Id)/2.0;
				Real Q=bff*E[0]*E[0]+2*bfs*E[1]*E[1]+bss*E[2]*E[2]+2*bfn*E[3]*E[3]+2*bsn*E[4]*E[4]+bnn*E[5]*E[5]; 
				Real Qbar= pow(sinfo->J,(Real)(-4.0/3.0))*Q;
				return a*(exp(Qbar)-1)/2+k0*(sinfo->J*log(sinfo->J)-sinfo->J-1);
	 }
	virtual void deriveSPKTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,MatrixSym &SPKTensorGeneral){
		MatrixSym inversematrix;
		invertMatrix(inversematrix,sinfo->deformationTensor);
		Real a=param.parameterArray[0];
		Real k0=param.parameterArray[1];
		Real bff=param.parameterArray[2];
		Real bfs=param.parameterArray[3];
		Real bss=param.parameterArray[4];
		Real bfn=param.parameterArray[5];
		Real bsn=param.parameterArray[6];
		Real bnn=param.parameterArray[7];
				MatrixSym Id;
				Id.identity();
				MatrixSym E=(sinfo->deformationTensor-Id)/2.0;
				Real Q=bff*E[0]*E[0]+2*bfs*E[1]*E[1]+bss*E[2]*E[2]+2*bfn*E[3]*E[3]+2*bsn*E[4]*E[4]+bnn*E[5]*E[5]; 
				Real Qbar= pow(sinfo->J,(Real)(-4.0/3.0))*Q;
				MatrixSym UE=MatrixSym(bff*E[0],bfs*E[1],bss*E[2],bfn*E[3],bsn*E[4],bnn*E[5]);
				SPKTensorGeneral=(UE-inversematrix*(Real)(2.0/3.0)*Q)*(a*exp(Qbar)*pow(sinfo->J,(Real)(-4.0/3.0)))+inversematrix*k0*sinfo->J*log(sinfo->J);
	}
	

    virtual void applyElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,const MatrixSym& inputTensor, MatrixSym &outputTensor)  {
		MatrixSym inversematrix;
		invertMatrix(inversematrix,sinfo->deformationTensor);
		Real a=param.parameterArray[0];
		Real k0=param.parameterArray[1];
		Real bff=param.parameterArray[2];
		Real bfs=param.parameterArray[3];
		Real bss=param.parameterArray[4];
		Real bfn=param.parameterArray[5];
		Real bsn=param.parameterArray[6];
		Real bnn=param.parameterArray[7];
				MatrixSym Id;
				Id.identity();
				MatrixSym E=(sinfo->deformationTensor-Id)/2.0;
				Real Q=bff*E[0]*E[0]+2*bfs*E[1]*E[1]+bss*E[2]*E[2]+2*bfn*E[3]*E[3]+2*bsn*E[4]*E[4]+bnn*E[5]*E[5];
				Real Qbar= pow(sinfo->J,(Real)(-4.0/3.0))*Q;
				MatrixSym UE=MatrixSym(bff*E[0],bfs*E[1],bss*E[2],bfn*E[3],bsn*E[4],bnn*E[5]);
				MatrixSym UH=MatrixSym(bff*inputTensor[0],bfs*inputTensor[1],bss*inputTensor[2],bfn*inputTensor[3],bsn*inputTensor[4],bnn*inputTensor[5]);
		Real trHUE=inputTensor[0]*UE[0]+inputTensor[2]*UE[2]+inputTensor[5]*UE[5]+2*inputTensor[1]*UE[1]+2*inputTensor[3]*UE[3]+2*inputTensor[4]*UE[4];
		//C-1:H
		Real _trHC=inputTensor[0]*inversematrix[0]+inputTensor[2]*inversematrix[2]+inputTensor[5]*inversematrix[5]
		+2*inputTensor[1]*inversematrix[1]+2*inputTensor[3]*inversematrix[3]+2*inputTensor[4]*inversematrix[4];
		//C-1HC-1 convert to sym matrix
		MatrixSym Firstmatrix;
		Firstmatrix.Mat2Sym(inversematrix.SymMatMultiply(inputTensor.SymSymMultiply(inversematrix)),Firstmatrix);		

	//	outputTensor=a/2.0*exp(val)*UH+a*trHUE*UE*exp(val)+k0*(sinfo->J*_trHC*inversematrix/4.0+sinfo->J*log(sinfo->J)*_trHC*inversematrix/4.0-sinfo->J*log(sinfo->J)*Firstmatrix/2.0);
	outputTensor=((UE-inversematrix*(Real)(2.0/3.0)*Q)*((Real)(-2.0/3.0)*_trHC)+UH/(Real)2.0+Firstmatrix*(Real)(2.0/3.0)*Q-inversematrix*(Real)(2.0/3.0)*trHUE)*(a*pow(sinfo->J,(Real)(-4.0/3.0))*exp(Qbar))
		+(UE-inversematrix*(Real)(2.0/3.0)*Q)*(a*pow(sinfo->J,(Real)(-8.0/3.0))*(trHUE-(Real)(2.0/3.0)*Q*_trHC)*exp(Qbar))+inversematrix*k0/(Real)2.0*sinfo->J*(log(sinfo->J)+(Real)1)*_trHC-Firstmatrix*sinfo->J*log(sinfo->J);


	}

	virtual void ElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,Matrix6 &outputTensor)  {
		Real a=param.parameterArray[0];
		Real k0=param.parameterArray[1];
		Real bff=param.parameterArray[2];
		Real bfs=param.parameterArray[3];
		Real bss=param.parameterArray[4];
		Real bfn=param.parameterArray[5];
		Real bsn=param.parameterArray[6];
		Real bnn=param.parameterArray[7];
		MatrixSym _C;
		MatrixSym Id;
		Id.identity();
		MatrixSym E=(sinfo->deformationTensor-Id)/2.0;
		invertMatrix(_C,sinfo->deformationTensor);
		Real Q=bff*E[0]*E[0]+2*bfs*E[1]*E[1]+bss*E[2]*E[2]+2*bfn*E[3]*E[3]+2*bsn*E[4]*E[4]+bnn*E[5]*E[5];
		Real Qbar= pow(sinfo->J,(Real)(-4.0/3.0))*Q;
		MatrixSym UE=MatrixSym(bff*E[0],bfs*E[1],bss*E[2],bfn*E[3],bsn*E[4],bnn*E[5]);
		MatrixSym CC;
		CC=_C;
		CC[1]+=_C[1];CC[3]+=_C[3];CC[4]+=_C[4];
		Matrix6 C_H_C;
		C_H_C[0][0]=_C[0]*_C[0]; C_H_C[1][1]=_C[1]*_C[1]+_C[0]*_C[2]; C_H_C[2][2]=_C[2]*_C[2]; C_H_C[3][3]=_C[3]*_C[3]+_C[0]*_C[5]; C_H_C[4][4]=_C[4]*_C[4]+_C[2]*_C[5];
		C_H_C[5][5]=_C[5]*_C[5];
		C_H_C[1][0]=_C[0]*_C[1];C_H_C[0][1]=2*C_H_C[1][0]; 
		C_H_C[2][0]=C_H_C[0][2]=_C[1]*_C[1]; C_H_C[5][0]=C_H_C[0][5]=_C[3]*_C[3];
		C_H_C[3][0]=_C[0]*_C[3];C_H_C[0][3]=2*C_H_C[3][0]; C_H_C[4][0]=_C[1]*_C[3];C_H_C[0][4]=2*C_H_C[4][0];
		C_H_C[1][2]=_C[2]*_C[1];C_H_C[2][1]=2*C_H_C[1][2]; C_H_C[1][5]=_C[3]*_C[4];C_H_C[5][1]=2*C_H_C[1][5];
		C_H_C[3][1]=C_H_C[1][3]=_C[0]*_C[4]+_C[1]*_C[3]; C_H_C[1][4]=C_H_C[4][1]=_C[1]*_C[4]+_C[2]*_C[3];
		C_H_C[3][2]=_C[4]*_C[1];C_H_C[2][3]=2*C_H_C[3][2]; C_H_C[4][2]=_C[4]*_C[2];C_H_C[2][4]=2*C_H_C[4][2];
		C_H_C[2][5]=C_H_C[5][2]=_C[4]*_C[4];
		C_H_C[3][5]=_C[3]*_C[5];C_H_C[5][3]=2*C_H_C[3][5];
		C_H_C[4][3]=C_H_C[3][4]=_C[3]*_C[4]+_C[5]*_C[1];
		C_H_C[4][5]=_C[4]*_C[5];C_H_C[5][4]=2*C_H_C[4][5];
		Matrix6 trC_HC_;
		trC_HC_[0]=_C[0]*CC;
		trC_HC_[1]=_C[1]*CC;
		trC_HC_[2]=_C[2]*CC;
		trC_HC_[3]=_C[3]*CC;
		trC_HC_[4]=_C[4]*CC;
		trC_HC_[5]=_C[5]*CC; 
		Matrix6 trUE_HC_;
		trUE_HC_[0]=UE[0]*CC;
		trUE_HC_[1]=UE[1]*CC;
		trUE_HC_[2]=UE[2]*CC;
		trUE_HC_[3]=UE[3]*CC;
		trUE_HC_[4]=UE[4]*CC;
		trUE_HC_[5]=UE[5]*CC;
		Matrix6 trC_HUE_;
		MatrixSym UUE;
		UUE=UE;
		UUE[1]+=UE[1];UUE[3]+=UE[3];UUE[4]+=UE[4];
		trC_HUE_[0]=_C[0]*UUE;
		trC_HUE_[1]=_C[1]*UUE;
		trC_HUE_[2]=_C[2]*UUE;
		trC_HUE_[3]=_C[3]*UUE;
		trC_HUE_[4]=_C[4]*UUE;
		trC_HUE_[5]=_C[5]*UUE;

		Matrix6 trUEHUE;
		trUEHUE[0]=UE[0]*UUE;
		trUEHUE[1]=UE[1]*UUE;
		trUEHUE[2]=UE[2]*UUE;
		trUEHUE[3]=UE[3]*UUE;
		trUEHUE[4]=UE[4]*UUE;
		trUEHUE[5]=UE[5]*UUE;
		Matrix6 U;
		U[0][0]=bff;U[0][0]=bff;U[1][1]=bfs;U[2][2]=bss;U[3][3]=bfn;U[4][4]=bsn;U[5][5]=bnn;
		outputTensor=(((trUE_HC_-trC_HC_*(Real)(2.0/3.0)*Q)*(Real)(-2.0/3.0)+U/(Real)2.0+C_H_C*(Real)(2.0/3.0)*Q-trC_HUE_*(Real)(2.0/3.0))*(a*pow(sinfo->J,(Real)(-4.0/3.0))*exp(Qbar))
		+(trUEHUE-trC_HUE_*(Real)(2.0/3.0)*Q -trUE_HC_*(Real)(2.0/3.0)*Q*Q+trC_HC_*(Real)(4.0/9.0)*Q*Q)*a*pow(sinfo->J,(Real)(-8.0/3.0))*exp(Qbar)+
		trC_HC_*k0/(Real)2.0*sinfo->J*(log(sinfo->J)+(Real)1)-C_H_C*sinfo->J*log(sinfo->J))*2;
	
	}

};


} // namespace fem

} // namespace component

} // namespace sofa

#endif
