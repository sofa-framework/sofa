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

#include <sofa/component/solidmechanics/fem/hyperelastic/material/BaseMaterial.h>
#include <sofa/type/Vec.h>

namespace sofa::component::solidmechanics::fem::hyperelastic::material
{

namespace
{
    using sofa::type::Vec3;
}

/**
 * Plastic material (proof of principle)
 */
class PlasticMaterial : public BaseMaterial
{

public:
    SOFA_CLASS(PlasticMaterial, BaseMaterial);

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);

    typedef sofa::type::vector<double> VecDouble;
    typedef sofa::type::vector<Vec3> VecStress;

	// Material properties
    Data<SReal> _poissonRatio; ///< Poisson ratio in Hooke's law
    Data<SReal> _youngModulus; ///< Young modulus in Hooke's law

    // Stress-strain curve description
    VecDouble _E;
    VecDouble _epsilon;
    VecStress _sigma;

    // Strain of the previous iteration
    VecDouble _previousVonMisesStrain;

    PlasticMaterial();
    void computeStress (Vec3& stress, Vec3& strain, unsigned int& elementIndex) override;
    void computeDStress (Vec3& dstress, Vec3& dstrain) override;

    SReal computeVonMisesStrain(Vec3 &strain);
    void computeStressOnSection(Vec3& Stress, Vec3 Strain, int section);	// computes the stress on a given section of the piecewise function

    void computeStress (unsigned int /*iElement*/) override {}

};


} // namespace sofa::component::solidmechanics::fem::hyperelastic::material
