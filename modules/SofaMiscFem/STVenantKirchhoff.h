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
#ifndef SOFA_COMPONENT_FEM_STVENANTKIRCHHOFF_H
#define SOFA_COMPONENT_FEM_STVENANTKIRCHHOFF_H
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
class STVenantKirchhoff : public HyperelasticMaterial<DataTypes>{

  typedef typename DataTypes::Coord::value_type Real;
  typedef defaulttype::Mat<3,3,Real> Matrix3;
  typedef defaulttype::Mat<6,6,Real> Matrix6;
  typedef defaulttype::MatSym<3,Real> MatrixSym;

public:

	virtual Real getStrainEnergy(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param) {
		Real I1=sinfo->trC;
			MatrixSym C=sinfo->deformationTensor;
			Real I1square=(Real)(C[0]*C[0] + C[2]*C[2]+ C[5]*C[5]+2*(C[1]*C[1] + C[3]*C[3] + C[4]*C[4]));
			Real I2=(Real)((pow(I1,(Real)2)- I1square)/2);
			Real mu=param.parameterArray[0];
			Real lambda=param.parameterArray[1];
			return (Real)(-mu*I2/2+(mu/4+lambda/8)*pow(I1,(Real)2)-I1*(3*lambda/4+mu/2));


	}
	void deriveSPKTensor(StrainInformation<DataTypes> *sinfo, const MaterialParameters<DataTypes> &param,MatrixSym &SPKTensorGeneral){
		MatrixSym C=sinfo->deformationTensor;
		Real I1=sinfo->trC;
		Real mu=param.parameterArray[0];
        Real lambda=param.parameterArray[1];
		MatrixSym ID;
		ID.identity();
        SPKTensorGeneral=(C-ID)*mu+ID*lambda/(Real)2.0*(I1-(Real)3.0);
	}

   void applyElasticityTensor(StrainInformation<DataTypes> *, const MaterialParameters<DataTypes> &param,const MatrixSym& inputTensor, MatrixSym &outputTensor)  {
		Real mu=param.parameterArray[0];
        Real lambda=param.parameterArray[1];
		MatrixSym ID;
		ID.identity();
		Real trH=inputTensor[0]+inputTensor[2]+inputTensor[5];
        outputTensor=ID*trH*lambda/2.0+inputTensor*mu;
	
	}
   virtual void ElasticityTensor(StrainInformation<DataTypes> *, const MaterialParameters<DataTypes> &param,Matrix6 &outputTensor)  {
		Real mu=param.parameterArray[0];
        Real lambda=param.parameterArray[1];
		MatrixSym ID;
		ID.identity();
		Matrix6 IDHID;
		IDHID.identity();
		Matrix6 trIDHID;
		trIDHID[0]=ID[0]*ID;
		trIDHID[1]=ID[1]*ID;
		trIDHID[2]=ID[2]*ID;
		trIDHID[3]=ID[3]*ID;
		trIDHID[4]=ID[4]*ID;
		trIDHID[5]=ID[5]*ID;
        outputTensor=(IDHID*mu+trIDHID*lambda/(Real)2.0)*2.0;
	
	}

	

	


};


} // namespace fem

} // namespace component

} // namespace sofa

#endif
