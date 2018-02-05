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
#ifndef SOFA_COMPONENT_FEM_BOYCEANDARRUDA_H
#define SOFA_COMPONENT_FEM_BOYCEANDARRUDA_H
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
class BoyceAndArruda : public HyperelasticMaterial<DataTypes>{

    typedef typename DataTypes::Coord::value_type Real;
    typedef defaulttype::Mat<3,3,Real> Matrix3;
    typedef defaulttype::Mat<6,6,Real> Matrix6;
    typedef defaulttype::MatSym<3,Real> MatrixSym;

  virtual Real getStrainEnergy(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param) {
		Real I1=sinfo->trC;
		Real mu=param.parameterArray[0];
		Real k0=param.parameterArray[1];
	  
	  return (Real)mu*((Real)(1.0/2.0)*(pow(sinfo->J,(Real)(-2.0/3.0))*I1-3)+(Real)(1.0/160.0)*(pow(sinfo->J,(Real)(-4.0/3.0))*I1*I1-9)+(Real)(11.0/(1050.0*8*8))*(pow(sinfo->J,(Real)-(Real)2.0)*I1*I1*I1-27)
			+(Real)(19.0/(7000.0*8.0*8.0*8.0))*(pow(sinfo->J,(Real)(-8.0/3.0))*pow(I1,(Real)4.0)-pow((Real)3,(Real)4))+(Real)(519.0/(673750.0*8.0*8.0*8.0*8.0))*(pow(sinfo->J,(Real)(-10.0/3.0))*pow(I1,(Real)5.0)-pow((Real)3,(Real)5)))
			+k0*log(sinfo->J)*log(sinfo->J)/2;

  }
  virtual void deriveSPKTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,MatrixSym &SPKTensorGeneral){
		MatrixSym inversematrix;
		MatrixSym C=sinfo->deformationTensor;
		invertMatrix(inversematrix,C);
		Real I1=sinfo->trC;
		Real mu=param.parameterArray[0];
		Real k0=param.parameterArray[1];
		MatrixSym ID;
		ID.identity();
		SPKTensorGeneral=((inversematrix*(Real)(-1.0/3.0)*I1+ID)*(Real)(1.0/2.0)*pow(sinfo->J,(Real)(-2.0/3.0))+(inversematrix*(Real)(-2.0/3.0)*I1*I1+ID*(Real)2.0*I1)*(Real)(1.0/160.0)*pow(sinfo->J,(Real)(-4.0/3.0))+(ID*(Real)3.0*I1*I1-inversematrix*I1*I1*I1)*(Real)(11.0/(1050.0*8*8))*pow(sinfo->J,(Real)(-2.0))
			+(inversematrix*(Real)(-4.0/3.0)*pow(I1,(Real)4.0)+ID*(Real)4.0*pow(I1,(Real)3.0))*(Real)(19.0/(7000.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-8.0/3.0))+(inversematrix*(Real)(-5.0/3.0)*pow(I1,(Real)5.0)+ID*(Real)5.0*pow(I1,(Real)4.0))*(Real)(519.0/(673750.0*8.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-10.0/3.0)))*2.0*mu
			+inversematrix*k0*log(sinfo->J);
	}
	

    virtual void applyElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,const MatrixSym& inputTensor, MatrixSym &outputTensor)  {
		MatrixSym inversematrix;
		MatrixSym C=sinfo->deformationTensor;
		invertMatrix(inversematrix,C);
		Real I1=sinfo->trC;
		Real mu=param.parameterArray[0];
		Real k0=param.parameterArray[1];
		MatrixSym ID;
		ID.identity();
		// C-1:H
		Real _trHC=inputTensor[0]*inversematrix[0]+inputTensor[2]*inversematrix[2]+inputTensor[5]*inversematrix[5]
		+2*inputTensor[1]*inversematrix[1]+2*inputTensor[3]*inversematrix[3]+2*inputTensor[4]*inversematrix[4];
		MatrixSym Firstmatrix;
		//C-1HC-1 convert to sym matrix
		Firstmatrix.Mat2Sym(inversematrix.SymMatMultiply(inputTensor.SymSymMultiply(inversematrix)),Firstmatrix);	
		//trH
		Real trH=inputTensor[0]+inputTensor[2]+inputTensor[5];

		outputTensor=(((inversematrix*(Real)(-1.0/3.0)*I1+ID)*(Real)(-1.0/3.0)*_trHC-inversematrix*(Real)(1.0/3.0)*trH+Firstmatrix*(Real)(1.0/3.0)*I1)*(Real)(1.0/2.0)*pow(sinfo->J,(Real)(-2.0/3.0))
			+((inversematrix*(Real)(-2.0/3.0)*I1*I1+ID*(Real)2.0*I1)*(Real)(-2.0/3.0)*_trHC-inversematrix*(Real)(4.0/3.0)*I1*trH+Firstmatrix*(Real)(2.0/3.0)*I1*I1+ID*(Real)(2.0)*trH)*(Real)(1.0/160.0)*pow(sinfo->J,(Real)(-4.0/3.0))
			+((inversematrix*(Real)(-I1)*I1*I1+ID*(Real)3.0*I1*I1)*(-_trHC)-inversematrix*(Real)3.0*I1*I1*trH+Firstmatrix*I1*I1*I1+ID*(Real)6.0*I1*trH)*(Real)(11.0/(1050.0*8.0*8.0))*pow(sinfo->J,(Real)-2.0)
			+((inversematrix*(Real)(-4.0/3.0)*pow(I1,(Real)4.0)+ID*(Real)4.0*pow(I1,(Real)3))*(Real)(-4.0/3.0)*_trHC-inversematrix*(Real)(16.0/3.0)*pow(I1,(Real)3.0)*trH+Firstmatrix*(Real)(4.0/3.0)*pow(I1,(Real)4.0)+ID*(Real)12.0*I1*I1*trH)*(Real)(19.0/(7000.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-8.0/3.0))+
			((inversematrix*(Real)(-5.0/3.0)*pow(I1,(Real)5.0)+ID*(Real)5.0*pow(I1,(Real)4))*(Real)(-5.0/3.0)*_trHC-inversematrix*(Real)(25.0/3.0)*pow(I1,(Real)4.0)*trH+Firstmatrix*(Real)(5.0/3.0)*pow(I1,(Real)5.0)+ID*(Real)20.0*pow(I1,(Real)3.0)*trH)*(Real)(519.0/(673750.0*8.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-10.0/3.0)))*2.0*mu
			+inversematrix*(Real)(k0/(Real)2.0)*_trHC-Firstmatrix*(Real)(k0*log(sinfo->J));
	}
	virtual void ElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,Matrix6 &outputTensor)  {
		MatrixSym ID;
		ID.identity();
		MatrixSym _C;
		invertMatrix(_C,sinfo->deformationTensor);
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
		Matrix6 trIDHID;
		trIDHID[0]=ID[0]*ID;
		trIDHID[1]=ID[1]*ID;
		trIDHID[2]=ID[2]*ID;
		trIDHID[3]=ID[3]*ID;
		trIDHID[4]=ID[4]*ID;
		trIDHID[5]=ID[5]*ID;
		Real I1=sinfo->trC;
	
		
		Real mu=param.parameterArray[0];
		Real k0=param.parameterArray[1];

		outputTensor=((((trC_HC_*(Real)(1.0/3.0)*I1-trID_HC_)*(Real)(1.0/3.0)-trC_HID*(Real)(1.0/3.0)+C_H_C*(Real)(1.0/3.0)*I1)*(Real)(1.0/2.0)*pow(sinfo->J,(Real)(-2.0/3.0))
			+((trC_HC_*(Real)(2.0/3.0)*I1*I1-trID_HC_*(Real)2.0*I1)*(Real)(2.0/3.0)-trC_HID*(Real)(4.0/3.0)*I1+C_H_C*(Real)(2.0/3.0)*I1*I1+trIDHID*(Real)2.0)*(Real)(1.0/160.0)*pow(sinfo->J,(Real)(-4.0/3.0))
			+(trC_HC_*I1*I1*I1-trID_HC_*(Real)3.0*I1*I1-trC_HID*(Real)3.0*I1*I1+C_H_C*I1*I1*I1+trIDHID*(Real)6.0*I1)*(Real)(11.0/(1050.0*8.0*8.0))*pow(sinfo->J,(Real)-2.0)
			+((trC_HC_*(Real)(4.0/3.0)*pow(I1,(Real)4.0)-trID_HC_*(Real)4.0*pow(I1,(Real)3.0))*(Real)(4.0/3.0)-trC_HID*(Real)(16.0/3.0)*pow(I1,(Real)3.0)+C_H_C*(Real)(4.0/3.0)*pow(I1,(Real)4.0)+trIDHID*(Real)12.0*I1*I1)*(Real)(19.0/(7000.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-8.0/3.0))
			+((trC_HC_*(Real)(5.0/3.0)*pow(I1,(Real)5.0)-trID_HC_*(Real)5*pow(I1,(Real)4.0))*(Real)(5.0/3.0)-trC_HID*(Real)(25.0/3.0)*pow(I1,(Real)4.0)+C_H_C*(Real)(5.0/3.0)*pow(I1,(Real)5.0)+trIDHID*(Real)20.0*pow(I1,(Real)3.0))*(Real)(519.0/(673750.0*8.0*8.0*8.0*8.0))*pow(sinfo->J,(Real)(-10.0/3.0)))*2.0*mu
			+trC_HC_*(Real)k0/(Real)2.0-C_H_C*k0*log(sinfo->J))*2.0;
			
	}


};
  	
	


} // namespace fem

} // namespace component

} // namespace sofa

#endif
