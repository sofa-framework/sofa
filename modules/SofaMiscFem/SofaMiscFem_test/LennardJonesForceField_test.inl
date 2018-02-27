/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_LENNARDJONESFORCEFIELD_TEST_INL
#define SOFA_LENNARDJONESFORCEFIELD_TEST_INL

#include "LennardJonesForceField_test.h"

// Solvers
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaImplicitOdeSolver/StaticSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>

// Box roi
#include <SofaGeneralEngine/PairBoxRoi.h>
#include <SofaEngine/BoxROI.h>
#include <SofaGeneralEngine/GenerateCylinder.h>

// Constraint
#include <SofaBoundaryCondition/ProjectToLineConstraint.h>
#include <SofaBoundaryCondition/FixedConstraint.h>
#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include <SofaBoundaryCondition/FixedPlaneConstraint.h>

// ForceField
#include <SofaBoundaryCondition/TrianglePressureForceField.h>
#include <SofaMiscForceField/LennardJonesForceField.h>

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <SofaGeneralDeformable/RegularGridSpringForceField.h>
#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaMiscForceField/MeshMatrixMass.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaRigid/RigidMapping.h>

namespace sofa
{

	typedef component::container::MechanicalObject<defaulttype::Vec3Types> MechanicalObject3;
	typedef component::mass::UniformMass<defaulttype::Vec3Types, SReal> UniformMass3;
	typedef component::projectiveconstraintset::FixedConstraint<defaulttype::Vec3Types> FixedConstraint3;

	/// Create a scene with a regular grid and an affine constraint for patch test

		template<class DataTypes>
		simulation::Node::SPtr LennardJonesForceField_test<DataTypes>::createSunPlanetSystem(
			simulation::Node::SPtr root,
			double mSun,
			double mPlanet,
			double g,
			Coord xSun,
			Deriv vSun,
			Coord xPlanet,
			Deriv vPlanet)
		{

			// Mechanical object with 2 dofs: first sun, second planet
			MechanicalObject3::SPtr sunPlanet_dof = modeling::addNew<MechanicalObject3>(root, "sunPlanet_MO");

			// Set position and velocity
			sunPlanet_dof->resize(2);
			// Position
			MechanicalObject3::VecCoord positions(2);
			positions[0] = xSun;
			positions[1] = xPlanet;
			MechanicalObject3::WriteVecCoord xdof = sunPlanet_dof->writePositions();
			copyToData(xdof, positions);
			// Velocity
			MechanicalObject3::VecDeriv velocities(2);
			velocities[0] = vSun;
			velocities[1] = vPlanet;
			MechanicalObject3::WriteVecDeriv vdof = sunPlanet_dof->writeVelocities();
			copyToData(vdof, velocities);

			// Fix sun
			FixedConstraint3::SPtr fixed = modeling::addNew<FixedConstraint3>(root, "FixedSun");
			fixed->addConstraint(0);

			// Uniform Mass
			UniformMass3::SPtr massPtr = modeling::addNew<UniformMass3>(root, "mass");
			massPtr->d_totalMass.setValue(mPlanet + mSun);

			// Lennard Jones Force Field
			typename component::forcefield::LennardJonesForceField<DataTypes>::SPtr ff =
				modeling::addNew<typename component::forcefield::LennardJonesForceField<DataTypes> >(root);
			// Set froce field parameters
			ff->setAlpha(1);
			ff->setBeta(-1);
			ff->setAInit(mPlanet*mSun*g);

			return root;

		}


}// sofa

#endif // SOFA_LENNARDJONESFORCEFIELD_TEST_INL

