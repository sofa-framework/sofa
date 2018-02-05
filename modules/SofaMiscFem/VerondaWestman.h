/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_FEM_VERONDAWESTMAN_H
#define SOFA_COMPONENT_FEM_VERONDAWESTMAN_H
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
class VerondaWestman : public HyperelasticMaterial<DataTypes>{

  typedef typename DataTypes::Coord::value_type Real;
  typedef defaulttype::Mat<3,3,Real> Matrix3;
  typedef defaulttype::Mat<6,6,Real> Matrix6;
  typedef defaulttype::MatSym<3,Real> MatrixSym;

	virtual Real getStrainEnergy(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param) {
		MatrixSym C=sinfo->deformationTensor;
		Real I1=sinfo->trC;
		Real I1square=(Real)(C[0]*C[0] + C[2]*C[2]+ C[5]*C[5]+2*(C[1]*C[1] + C[3]*C[3] + C[4]*C[4]));
		Real I2=(Real)((pow(I1,(Real)2)- I1square)/2);
		Real c1=param.parameterArray[0];
		Real c2=param.parameterArray[1];
		Real k0=param.parameterArray[2];
		return c1*(exp(c2*(pow(sinfo->J,(Real)(-2.0/3.0))*I1-3))-1)-c1*c2*(pow(sinfo->J,(Real)(-4.0/3.0))*I2-3)/2+k0*log(sinfo->J)*log(sinfo->J)/2;

	}

	  virtual void deriveSPKTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,MatrixSym &SPKTensorGeneral){
		MatrixSym inversematrix;
		MatrixSym C=sinfo->deformationTensor;
		invertMatrix(inversematrix,C);
		Real I1=sinfo->trC;
		Real I1square=(Real)(C[0]*C[0] + C[2]*C[2]+ C[5]*C[5]+2*(C[1]*C[1] + C[3]*C[3] + C[4]*C[4]));
		Real I2=(Real)((pow(I1,(Real)2)- I1square)/2);
		Real c1=param.parameterArray[0];
		Real c2=param.parameterArray[1];
		Real k0=param.parameterArray[2];
		MatrixSym ID;
		ID.identity();
		SPKTensorGeneral=(ID-inversematrix*I1/3)*2*c1*c2*pow(sinfo->J,(Real)(-2.0/3.0))*exp(c2*(pow(sinfo->J,(Real)(-2.0/3.0))*I1-3))
			+(ID*I1-C-inversematrix*I2*2/3)*(-c1*c2*pow(sinfo->J,(Real)(-4.0/3.0)))+inversematrix*k0*log(sinfo->J);
	}
	

    virtual void applyElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,const MatrixSym& inputTensor, MatrixSym &outputTensor)  {
		MatrixSym inversematrix;
		MatrixSym C=sinfo->deformationTensor;
		invertMatrix(inversematrix,C);
		Real I1=sinfo->trC;
		Real I1square=(Real)(C[0]*C[0] + C[2]*C[2]+ C[5]*C[5]+2*(C[1]*C[1] + C[3]*C[3] + C[4]*C[4]));
		Real I2=(Real)((pow(I1,(Real)2)- I1square)/2);
		Real c1=param.parameterArray[0];
		Real c2=param.parameterArray[1];
		Real k0=param.parameterArray[2];
		MatrixSym ID;
		ID.identity();
		// C-1:H
		Real _trHC=inputTensor[0]*inversematrix[0]+inputTensor[2]*inversematrix[2]+inputTensor[5]*inversematrix[5]
		+2*inputTensor[1]*inversematrix[1]+2*inputTensor[3]*inversematrix[3]+2*inputTensor[4]*inversematrix[4];
		MatrixSym Firstmatrix;
		//C-1HC-1 convert to sym matrix
		Firstmatrix.Mat2Sym(inversematrix.SymMatMultiply(inputTensor.SymSymMultiply(inversematrix)),Firstmatrix);	
		//C:H
		Real trHC=inputTensor[0]*C[0]+inputTensor[2]*C[2]+inputTensor[5]*C[5]
		+2*inputTensor[1]*C[1]+2*inputTensor[3]*C[3]+2*inputTensor[4]*C[4];

		//trH
		Real trH=inputTensor[0]+inputTensor[2]+inputTensor[5];

		outputTensor=(inversematrix*I1*_trHC/(Real)9.0-inversematrix*trH/(Real)3.0+Firstmatrix/(Real)3.0*I1-ID*_trHC/(Real)3.0)*(Real)2.0*c1*c2*pow(sinfo->J,(Real)(-2.0/3.0))*exp(c2*(I1*pow(sinfo->J,(Real)(-2.0/3.0))-(Real)3.0))
			+(ID-inversematrix*I1/(Real)3.0)*exp(c2*(I1*pow(sinfo->J,(Real)(-2.0/3.0))-(Real)3))*(Real)2*c1*c2*c2*pow(sinfo->J,(Real)(-4.0/3.0))*(-I1*_trHC/(Real)3.0+trH)
			+(inversematrix*(Real)(-4.0)*_trHC*I2/9.0+inversematrix*(Real)2.0*I1*trH/3.0-inversematrix*(Real)2.0*trHC/3.0-Firstmatrix*(Real)2.0*I2/3.0+ID*_trHC*I1*(Real)(2.0/3.0)-ID*trH-C*_trHC*(Real)(2.0/3.0)+inputTensor)*c1*c2*pow(sinfo->J,(Real)(-4.0/3.0))
			+Firstmatrix*(Real)(-k0)*log(sinfo->J)+inversematrix*k0*_trHC/(Real)2.0; 
	
	}
	virtual void ElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,Matrix6 &outputTensor)  {
		MatrixSym _C;
		MatrixSym ID;
		ID.identity();
		invertMatrix(_C,sinfo->deformationTensor);
		MatrixSym CC;
		CC=_C;
		CC[1]+=_C[1];CC[3]+=_C[3];CC[4]+=_C[4];
		MatrixSym C;
		C=sinfo->deformationTensor;
		C[1]+=sinfo->deformationTensor[1];C[3]+=sinfo->deformationTensor[3];C[4]+=sinfo->deformationTensor[4];
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
		Matrix6 trCH_C;//(C-1:H)C
		trCH_C[0]=sinfo->deformationTensor[0]*CC;
		trCH_C[1]=sinfo->deformationTensor[1]*CC;
		trCH_C[2]=sinfo->deformationTensor[2]*CC;
		trCH_C[3]=sinfo->deformationTensor[3]*CC;
		trCH_C[4]=sinfo->deformationTensor[4]*CC;
		trCH_C[5]=sinfo->deformationTensor[5]*CC;
		Matrix6 trC_HC;
		trC_HC[0]=_C[0]*C;
		trC_HC[1]=_C[1]*C;
		trC_HC[2]=_C[2]*C;
		trC_HC[3]=_C[3]*C;
		trC_HC[4]=_C[4]*C;
		trC_HC[5]=_C[5]*C;
		Matrix6 trID_HC_;
		trID_HC_[0]=ID[0]*CC;
		trID_HC_[1]=ID[1]*CC;
		trID_HC_[2]=ID[2]*CC;
		trID_HC_[3]=ID[3]*CC;
		trID_HC_[4]=ID[4]*CC;
		trID_HC_[5]=ID[5]*CC;
		Matrix6 trC_HID;
		trC_HID[0]=_C[0]*ID;
		trC_HID[1]=_C[1]*ID;
		trC_HID[2]=_C[2]*ID;
		trC_HID[3]=_C[3]*ID;
		trC_HID[4]=_C[4]*ID;
		trC_HID[5]=_C[5]*ID;
		Matrix6 IDHID;
		IDHID.identity();
		Matrix6 trIDHID;
		trIDHID[0]=ID[0]*ID;
		trIDHID[1]=ID[1]*ID;
		trIDHID[2]=ID[2]*ID;
		trIDHID[3]=ID[3]*ID;
		trIDHID[4]=ID[4]*ID;
		trIDHID[5]=ID[5]*ID;
		Real I1=sinfo->trC;
		Real I1square=(Real)(sinfo->deformationTensor[0]*sinfo->deformationTensor[0] + sinfo->deformationTensor[2]*sinfo->deformationTensor[2]+ sinfo->deformationTensor[5]*sinfo->deformationTensor[5]+
			2*(sinfo->deformationTensor[1]*sinfo->deformationTensor[1] + sinfo->deformationTensor[3]*sinfo->deformationTensor[3] + sinfo->deformationTensor[4]*sinfo->deformationTensor[4]));
		Real I2=(Real)((pow(I1,(Real)2)- I1square)/2);
		Real c1=param.parameterArray[0];
		Real c2=param.parameterArray[1];
		Real k0=param.parameterArray[2];

	/*	// C-1:H
		Real _trHC=inputTensor[0]*inversematrix[0]+inputTensor[2]*inversematrix[2]+inputTensor[5]*inversematrix[5]
		+2*inputTensor[1]*inversematrix[1]+2*inputTensor[3]*inversematrix[3]+2*inputTensor[4]*inversematrix[4];
		MatrixSym Firstmatrix;
		//C-1HC-1 convert to sym matrix
		Firstmatrix.Mat2Sym(inversematrix.SymMatMultiply(inputTensor.SymSymMultiply(inversematrix)),Firstmatrix);	
		//C:H
		Real trHC=inputTensor[0]*C[0]+inputTensor[2]*C[2]+inputTensor[5]*C[5]
		+2*inputTensor[1]*C[1]+2*inputTensor[3]*C[3]+2*inputTensor[4]*C[4];

		//trH
		Real trH=inputTensor[0]+inputTensor[2]+inputTensor[5];*/
		
		outputTensor=((trC_HC_*I1/(Real)9.0-trC_HID/(Real)3.0+C_H_C/(Real)3.0*I1-trID_HC_/(Real)3.0)*(Real)2.0*c1*c2*pow(sinfo->J,(Real)(-2.0/3.0))*exp(c2*(I1*pow(sinfo->J,(Real)(-2.0/3.0))-(Real)3.0))
			+(trC_HC_*I1*I1/(Real)9.0-trC_HID*I1/(Real)3.0-trID_HC_*I1/(Real)3.0+trIDHID)*(Real)2*c1*c2*c2*pow(sinfo->J,(Real)(-4.0/3.0))*exp(c2*(I1*pow(sinfo->J,(Real)(-2.0/3.0))-(Real)3))
			+(trC_HC_*(-1.0)*(Real)4.0*I2/(Real)9.0+trC_HID*(Real)2.0*I1/(Real)3.0-trC_HC*(Real)2.0/(Real)3.0-C_H_C*(Real)2.0*I2/(Real)3.0+trID_HC_*I1*(Real)(2.0/3.0)-trIDHID-trCH_C*(Real)(2.0/3.0)+IDHID)*c1*c2*pow(sinfo->J,(Real)(-4.0/3.0))
			-C_H_C*k0*log(sinfo->J)+trC_HC_*k0/(Real)2.0)*2.0; 
	
	

		
	}

};


} // namespace fem

} // namespace component

} // namespace sofa

#endif
