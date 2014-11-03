/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_FEM_HYPERELASTICMATERIAL_H
#define SOFA_COMPONENT_FEM_HYPERELASTICMATERIAL_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/MatSym.h>
#include <string>

#ifdef SOFA_HAVE_EIGEN2
//#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#endif
namespace sofa
{

namespace component
{

namespace fem
{

template<typename Real>
class StrainInformation;

template<typename DataTypes>
struct MaterialParameters;

/** a Class that describe a generic hyperelastic material .
The material is described based on continuum mechanics and the description is independent
to any discretization method like the finite element method. 
A material is generically described by a strain energy function and its first and second derivatives.
*/
template<class DataTypes>
class HyperelasticMaterial
{
public:

  typedef typename DataTypes::Coord Coord;
  typedef typename Coord::value_type Real;
  typedef defaulttype::MatSym<3,Real> MatrixSym;
  typedef defaulttype::Mat<3,3,Real> Matrix3;
  typedef defaulttype::Mat<6,6,Real> Matrix6;

   virtual ~HyperelasticMaterial(){}

	
	/** returns the strain energy of the current configuration */
	virtual Real getStrainEnergy(StrainInformation<DataTypes> *, const  MaterialParameters<DataTypes> &) {
			return 0;
	}


	/** computes the second Piola Kirchhoff stress tensor of the current configuration */
    virtual void deriveSPKTensor(StrainInformation<DataTypes> *, const  MaterialParameters<DataTypes> &,MatrixSym &)  {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

	}
	/** computes the Elasticity Tensor of the current configuration */

	virtual void applyElasticityTensor(StrainInformation<DataTypes> *, const  MaterialParameters<DataTypes> &,const MatrixSym , MatrixSym &)  {
		
	}

	virtual void ElasticityTensor(StrainInformation<DataTypes> *, const  MaterialParameters<DataTypes> &, Matrix6&) {;}
			

};

/** structure that store the parameters required to that are necessary to compute the strain energy
The material parameters might be constant in space (homogeneous material) or not */
template<typename DataTypes>
struct MaterialParameters {
  typedef typename DataTypes::Coord Coord;
  typedef typename Coord::value_type Real;

  /** an array of Real values that correspond to the material parameters : the size depends on the material,
  e.g. 2 Lam� coefficients for St-Venant Kirchhoff materials */
  std::vector<Real> parameterArray;
  /** the direction of anisotropy in the rest configuration  : the size of the array is 0 if the material is
  isotropic, 1 if it is transversely isotropic and 2 for orthotropic materials (assumed to be orthogonal to each other)*/
  std::vector<Coord> anisotropyDirection;
  /** for viscous part, give the real alphai and taui such as alpha(t)= alpha0+sum(1,N)alphaiexp(-t/taui)*/
  std::vector<Real> parameterAlpha;
  std::vector<Real> parameterTau;//starting with delta t the time step
};

template<typename DataTypes>
class StrainInformation
{
public:


  typedef typename DataTypes::Coord Coord;
  typedef typename Coord::value_type Real;
  typedef defaulttype::MatSym<3,Real> MatrixSym;
#ifdef SOFA_HAVE_EIGEN2
  typedef typename Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real,3,3> >::MatrixType EigenMatrix;
  typedef typename Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real,3,3> >::RealVectorType CoordEigen;
#endif
  /// Trace of C = I1
  Real trC;
  Real J;
  Real lambda;
  /// Trace of C^2 : I2 = (trCSquare - trC^2)/2
  Real trCsquare;

  /// boolean indicating whether the invariants have been computed
  bool hasBeenInitialized;
  /// right Cauchy-Green deformation tensor C (gradPhi^T gradPhi)
  MatrixSym deformationTensor;
#ifdef SOFA_HAVE_EIGEN2
  EigenMatrix Evect;
  CoordEigen Evalue;
#endif
  Real logJ;
  MatrixSym E;


  StrainInformation() : hasBeenInitialized(false) {}
  virtual ~StrainInformation() {}
};

} // namespace fem

} // namespace component

} // namespace sofa

#endif


