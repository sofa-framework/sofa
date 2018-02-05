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
#ifndef SOFA_COMPONENT_FEM_NEOHOOKEAN_H
#define SOFA_COMPONENT_FEM_NEOHOOKEAN_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif
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
class NeoHookean : public HyperelasticMaterial<DataTypes>{

  typedef typename DataTypes::Coord::value_type Real;
  typedef defaulttype::Mat<3,3,Real> Matrix3;
  typedef defaulttype::Mat<6,6,Real> Matrix6;
  typedef defaulttype::MatSym<3,Real> MatrixSym;
 
  virtual Real getStrainEnergy(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param) {
		Real mu=param.parameterArray[0];
		Real k=param.parameterArray[1];
		Real I1=sinfo->trC;
		return (Real)mu*(Real)(1.0/2.0)*(I1-3)-mu*log(sinfo->J)+k*log(sinfo->J)*log(sinfo->J)/2;
  }

	virtual void deriveSPKTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,MatrixSym &SPKTensorGeneral){
		MatrixSym inversematrix;
		invertMatrix(inversematrix,sinfo->deformationTensor);
		Real mu=param.parameterArray[0];
		Real k=param.parameterArray[1];
		MatrixSym ID;
		ID.identity();
		SPKTensorGeneral=mu*ID+inversematrix*(-mu+k*log(sinfo->J));
	}
	

    virtual void applyElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,const MatrixSym& inputTensor, MatrixSym &outputTensor)  {
		Real mu=param.parameterArray[0];
		Real k=param.parameterArray[1];
		MatrixSym inversematrix;
		invertMatrix(inversematrix,sinfo->deformationTensor);
		MatrixSym ID;
		ID.identity();
		Real trHC=inputTensor[0]*inversematrix[0]+inputTensor[2]*inversematrix[2]+inputTensor[5]*inversematrix[5]
		+2*inputTensor[1]*inversematrix[1]+2*inputTensor[3]*inversematrix[3]+2*inputTensor[4]*inversematrix[4];
		MatrixSym Firstmatrix;
		Firstmatrix.Mat2Sym(inversematrix.SymMatMultiply(inputTensor.SymSymMultiply(inversematrix)),Firstmatrix);		
		outputTensor=Firstmatrix*(mu-k*log(sinfo->J))+inversematrix*(k*trHC/2);
	
	}

	virtual void ElasticityTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,Matrix6 &outputTensor)  {
		Real mu=param.parameterArray[0];
		Real k=param.parameterArray[1];
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
		

		outputTensor=(C_H_C*(mu-k*log(sinfo->J))+trC_HC_*k/(Real)2.0)*2.0;
	
	}


};


} // namespace fem

} // namespace component

} // namespace sofa

#endif
