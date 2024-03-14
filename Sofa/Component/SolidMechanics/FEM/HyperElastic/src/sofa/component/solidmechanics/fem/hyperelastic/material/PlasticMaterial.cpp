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

#include <sofa/component/solidmechanics/fem/hyperelastic/material/PlasticMaterial.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <Eigen/SVD>
namespace sofa::component::solidmechanics::fem::hyperelastic::material
{

PlasticMaterial::PlasticMaterial()
: _poissonRatio(initData(&_poissonRatio,(SReal)0.45,"poissonRatio","Poisson ratio in Hooke's law"))
, _youngModulus(initData(&_youngModulus,(SReal)3000.,"youngModulus","Young modulus in Hooke's law"))
{
	// we give the inclination here
	_E.push_back(_youngModulus.getValue());
	_E.push_back(0.5*_youngModulus.getValue());

	// we give the limits that separates the sections
	_epsilon.push_back(10000);

	// contains the value of the stress at the upper extremity of each section
	_sigma.push_back(Vec3(0, 0, 0));
	const Vec3 Stress;
//	computeStressOnSection(Stress, _epsilon[0], 0);
	_sigma.push_back(Stress);
}

void PlasticMaterial::computeStress(Vec3& Stress, Vec3& Strain, unsigned int& elementIndex)
{
	// Computes the Von Mises strain
	const SReal vonMisesStrain = computeVonMisesStrain(Strain);

	// Seeks the section of the piecewise function where we are on
	int section = 0;
	while ( vonMisesStrain > _epsilon[section] )
	{
		section++;
	}

	// If strain increases, we are on the loading curve
	if ( vonMisesStrain - _previousVonMisesStrain[elementIndex] > 0 )
	{
		// we compute the stress according to the section of the curve
		computeStressOnSection(Stress, Strain, section);
	}
	// Otherwise we are on the unloading curve
	else
	{
//		 Stress *= (_E[section] / (12 * (1 - _poissonRatio.getValue()*_poissonRatio.getValue())));
	}


	// Stores the strain for use in the next iteration
	_previousVonMisesStrain.push_back(vonMisesStrain);
}

void PlasticMaterial::computeDStress(Vec3& dStress, Vec3& dStrain)
{
	dStress[0] = dStrain[0] + _poissonRatio.getValue() * dStrain[1];
	dStress[1] = _poissonRatio.getValue() * dStrain[0] + dStrain[1];
	dStress[2] = 0.5f * (1-_poissonRatio.getValue()) * dStrain[2];

	dStress *= (_youngModulus.getValue() / (12 * (1 - _poissonRatio.getValue()*_poissonRatio.getValue())));
}

SReal PlasticMaterial::computeVonMisesStrain(Vec3 &strain)
{
	Eigen::Matrix<SReal, -1, -1> e;
	e.resize(2, 2);

	e(0,0) = strain[0];
	e(0,1) = strain[2];
	e(1,0) = strain[2];
	e(1,1) = strain[1];

	//compute eigenvalues and eigenvectors
	const Eigen::JacobiSVD svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);

	const auto& S = svd.singularValues();

	return 1/(1+_poissonRatio.getValue())*sqrt( 0.5*( (S(0)-S(1))*(S(0)-S(1)) + (S(1)-S(2))*(S(1)-S(2)) + (S(2)-S(0))*(S(2)-S(0)) ));
}

void PlasticMaterial::computeStressOnSection(Vec3& Stress, Vec3 Strain, int section)
{
	Stress[0] = Strain[0] + _poissonRatio.getValue() * Strain[1];
	Stress[1] = _poissonRatio.getValue() * Strain[0] + Strain[1];
	Stress[2] = 0.5f * (1-_poissonRatio.getValue()) * Strain[2];

	Stress *= (_E[section] / (12 * (1 - _poissonRatio.getValue()*_poissonRatio.getValue())));
	Stress += _sigma[section];

}

int PlasticMaterialClass = core::RegisterObject("Plastic material")
.add< PlasticMaterial >()
;

} // namespace sofa::component::solidmechanics::fem::hyperelastic::material
