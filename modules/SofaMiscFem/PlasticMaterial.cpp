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
#include <SofaMiscFem/initMiscFEM.h>
#include <SofaMiscFem/PlasticMaterial.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace fem
{

namespace material
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
	_sigma.push_back(Vector3(0, 0, 0));
	Vector3 Stress;
//	computeStressOnSection(Stress, _epsilon[0], 0);
	_sigma.push_back(Stress);
}

void PlasticMaterial::computeStress(Vector3& Stress, Vector3& Strain, unsigned int& elementIndex)
{
	// Computes the Von Mises strain
    SReal vonMisesStrain = computeVonMisesStrain(Strain);

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

void PlasticMaterial::computeDStress(Vector3& dStress, Vector3& dStrain)
{
	dStress[0] = dStrain[0] + _poissonRatio.getValue() * dStrain[1];
	dStress[1] = _poissonRatio.getValue() * dStrain[0] + dStrain[1];
	dStress[2] = 0.5f * (1-_poissonRatio.getValue()) * dStrain[2];

	dStress *= (_youngModulus.getValue() / (12 * (1 - _poissonRatio.getValue()*_poissonRatio.getValue())));
}

SReal PlasticMaterial::computeVonMisesStrain(Vector3 &strain)
{
	NEWMAT::SymmetricMatrix e(2);
	e = 0.0;

	NEWMAT::DiagonalMatrix D(2);
	D = 0.0;

	NEWMAT::Matrix V(2,2);
	V = 0.0;

	e(1,1) = strain[0];
	e(1,2) = strain[2];
	e(2,1) = strain[2];
	e(2,2) = strain[1];

	NEWMAT::Jacobi(e, D, V);

	return 1/(1+_poissonRatio.getValue())*sqrt( 0.5*( (D(1,1)-D(2,2))*(D(1,1)-D(2,2)) + (D(2,2)-D(3,3))*(D(2,2)-D(3,3)) + (D(3,3)-D(1,1))*(D(3,3)-D(1,1)) ));
}

void PlasticMaterial::computeStressOnSection(Vector3& Stress, Vector3 Strain, int section)
{
	Stress[0] = Strain[0] + _poissonRatio.getValue() * Strain[1];
	Stress[1] = _poissonRatio.getValue() * Strain[0] + Strain[1];
	Stress[2] = 0.5f * (1-_poissonRatio.getValue()) * Strain[2];

	Stress *= (_E[section] / (12 * (1 - _poissonRatio.getValue()*_poissonRatio.getValue())));
	Stress += _sigma[section];

}

SOFA_DECL_CLASS(PlasticMaterial)

int PlasticMaterialClass = core::RegisterObject("Plastic material")
.add< PlasticMaterial >()
;

} // namespace material

} // namespace fem

} // namespace component

} // namespace sofa
