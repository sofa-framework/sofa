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
#include <sofa/testing/BaseSimulationTest.h>
#include <sofa/testing/NumericTest.h>

#include <sofa/type/Quat.h>

//Including Simulation
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

// Including constraint, force and mass
#include <sofa/component/constraint/projective/AffineMovementProjectiveConstraint.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/solidmechanics/spring/MeshSpringForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>
#include <sofa/core/MechanicalParams.h>

#include <sofa/defaulttype/VecTypes.h>
#include <SceneCreator/SceneCreator.h>

#include <sofa/component/topology/testing/RegularGridNodeCreation.h>

namespace sofa {

using namespace component;
using namespace type;
using namespace defaulttype;
using namespace modeling;

/**  AFfine Patch test.
An affine movement (rotation and translation) is applied to the borders of a mesh. Test if the points inside have the same affine movement.*/

template <typename _DataTypes>
struct AffinePatch_sofa_test : public sofa::testing::BaseSimulationTest, sofa::testing::NumericTest<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef constraint::projective::AffineMovementProjectiveConstraint<DataTypes> AffineMovementProjectiveConstraint;
    typedef statecontainer::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename component::solidmechanics::spring::MeshSpringForceField<DataTypes> MeshSpringForceField;
    typedef typename component::solidmechanics::fem::elastic::TetrahedronFEMForceField<DataTypes> TetraForceField;
    typedef type::Quat<SReal> Quat;
    typedef type::Vec3 Vec3;

    /// Root of the scene graph
    simulation::Node::SPtr root;
    /// Tested simulation
    simulation::Simulation* simulation;
    /// Structure which contains current node and pointers to the mechanical object and the affine constraint
    PatchTestStruct<DataTypes> patchStruct;
    /// Tested Rotation: random rotation matrix
    type::Mat<3,3,Real> testedRotation;
    /// Tested Translation: random translation
    Coord testedTranslation;

    /// Create the context for the scene
    void doSetUp() override
    {
        // Init simulation
        simulation = sofa::simulation::getSimulation();

        root = simulation::getSimulation()->createNewGraph("root");

    }

    /// Create a scene with a 2D regular grid and an affine constraint
    void createScene2DRegularGrid(bool randomRotation = true, bool randomTranslation=true)
    {
        // Create a scene with a regular grid
        patchStruct = sofa::createRegularGridScene<DataTypes>(
                        root,  // attached to the root node
                        Vec<3,SReal>(0,0,0), // Start point of regular grid
                        Vec<3,SReal>(1,1,0), // End point of regular grid
                        5,5,1,  // Resolution of the regular grid
                        Vec<6,SReal>(-0.1,-0.1,0,1.1,1.1,0), // BoxRoi to find all mesh points
                        Vec<6,SReal>(-0.1,-0.1,0,1.1,1.1,0), // inclusive box of pair box roi
                        Vec<6,SReal>(0.1,0.1,0,0.9,0.9,0)); // included box of pair box roi

        simulation::Node::SPtr SquareNode = patchStruct.SquareNode;

        //Force field for 2D Grid
        typename MeshSpringForceField::SPtr meshSpringForceField = addNew<MeshSpringForceField> (SquareNode,"forceField");
        meshSpringForceField->setStiffness(10);

        if(randomRotation)
        // Random Rotation
        {

            SReal x,y,z,w;
            x = 0;
            y = 0;
            z = 1;
            w = SReal(helper::drand()*360.0);
            Quat quat(x,y,z,w);
            quat.normalize();
            quat.toMatrix(testedRotation);
        }
        patchStruct.affineConstraint->d_rotation.setValue(testedRotation);

        // Random Translation
        if(randomTranslation)
        {
            for(size_t i=0;i<Coord::total_size;++i)
            {
                testedTranslation[i]=helper::drand(2);
                if(i==2)
                    testedTranslation[i]=0;
            }
        }
        patchStruct.affineConstraint->d_translation.setValue(testedTranslation);

    }

    /// Create a scene with a 3D regular grid and an affine constraint
    void createScene3DRegularGrid(bool randomRotation = true, bool randomTranslation=true)
    {
        // Create a scene with a regular grid
        patchStruct = sofa::createRegularGridScene<DataTypes>(
                        root,  // attached to the root node
                        Vec<3,SReal>(0,0,0), // Start point of regular grid
                        Vec<3,SReal>(1,1,1), // End point of regular grid
                        5,5,5,  // Resolution of the regular grid
                        Vec<6,SReal> (-0.1,-0.1,-0.1,1.1,1.1,1.1), // BoxRoi to find all mesh points
                        Vec<6,SReal>(-0.1,-0.1,-0.1,1.1,1.1,1.1), // inclusive box of pair box roi
                        Vec<6,SReal>(0.1,0.1,0.1,0.9,0.9,0.9)); // included box of pair box roi

        simulation::Node::SPtr SquareNode = patchStruct.SquareNode;

        // Force field for 3D Grid
        typename TetraForceField::SPtr tetraFEM = addNew<TetraForceField>(SquareNode,"forceField");
        tetraFEM->setMethod("small");
        tetraFEM->setYoungModulus(20);
        tetraFEM->setPoissonRatio(0.4);

        // Random Rotation
        if(randomRotation)
        {
            SReal x,y,z,w;
            x = SReal(helper::drand(1));
            y = SReal(helper::drand(1));
            z = SReal(helper::drand(1));
            // If the rotation axis is null
            Vec3 rotationAxis(x,y,z);
            if(rotationAxis.norm() < 1e-7)
            {
                rotationAxis = Vec3(0,0,1);
            }
            w = SReal(helper::drand()*360.0);
            Quat quat(x,y,z,w);
            quat.normalize();
            quat.toMatrix(testedRotation);
        }
        patchStruct.affineConstraint->d_rotation.setValue(testedRotation);

        // Random Translation
        if(randomTranslation)
        {
            for(size_t i=0;i<Coord::total_size;++i)
            {
                testedTranslation[i] = helper::drand(2);
            }
        }
        patchStruct.affineConstraint->d_translation.setValue(testedTranslation);

    }

    void setRotation(type::Mat<3,3,Real> rotationMatrix)
    {
        testedRotation = rotationMatrix;
    }

    void setTranslation(int x,int y,int z)
    {
        testedTranslation = Coord(x,y,z);
    }

    /// After simulation compare the positions of points to the theoretical positions.
    bool compareSimulatedToTheoreticalPositions(double convergenceAccuracy, double diffMaxBetweenSimulatedAndTheoreticalPosition)
    {
        // Init simulation
        sofa::simulation::node::initRoot(root.get());

        // Compute the theoretical final positions
        VecCoord finalPos;
        patchStruct.affineConstraint->getFinalPositions( finalPos,*patchStruct.dofs->write(core::vec_id::write_access::position) );


        // Initialize
        size_t numNodes = finalPos.size();
        VecCoord xprev(numNodes);
        VecDeriv dx(numNodes);
        bool hasConverged = true;

        for (size_t i=0; i<numNodes; i++)
        {
            xprev[i] = CPos(0,0,0);
        }

        // Animate
        do
        {
            hasConverged = true;
            sofa::simulation::node::animate(root.get(), 0.5_sreal);
            typename MechanicalObject::ReadVecCoord x = patchStruct.dofs->readPositions();

            // Compute dx
            for (size_t i=0; i<x.size(); i++)
            {
                dx[i] = x[i]-xprev[i];
                // Test convergence
                if(dx[i].norm()>convergenceAccuracy) hasConverged = false;
            }

            // xprev = x
            for (size_t i=0; i<numNodes; i++)
            {
                xprev[i]=x[i];
            }
        }
        while(!hasConverged); // not converged

        // Get simulated positions
        typename MechanicalObject::WriteVecCoord x = patchStruct.dofs->writePositions();

        // Compare the theoretical positions and the simulated positions
        bool succeed=true;
        for(size_t i=0; i<finalPos.size(); i++ )
        {
            if((finalPos[i]-x[i]).norm()>diffMaxBetweenSimulatedAndTheoreticalPosition)
            {
                succeed = false;
                ADD_FAILURE() << "final Position of point " << i << " is wrong: " << x[i] << std::endl <<"the expected Position is " << finalPos[i] << std::endl
                    << "difference = " <<(finalPos[i]-x[i]).norm() << std::endl <<"rotation = " << testedRotation << std::endl << " translation = " << testedTranslation << std::endl
                    << "Failed seed number =" << sofa::testing::NumericTest<typename _DataTypes::Real>::seed;

            }
        }
        return succeed;
    }

};

// Define the list of DataTypes to instantiate
using ::testing::Types;
typedef Types<
    Vec3Types
> DataTypes; // the types to instantiate.

// Test suite for all the instantiations
TYPED_TEST_SUITE(AffinePatch_sofa_test, DataTypes);

// first test case
TYPED_TEST( AffinePatch_sofa_test , patchTest2D )
{
   this->createScene2DRegularGrid();
   ASSERT_TRUE( this->compareSimulatedToTheoreticalPositions(5e-6,5e-5));
}

// second test case
TYPED_TEST( AffinePatch_sofa_test , patchTest3D )
{
    this->createScene3DRegularGrid();
    ASSERT_TRUE( this->compareSimulatedToTheoreticalPositions(1e-5,1.1e-4));
}

} // namespace sofa
