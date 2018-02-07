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
#include <SofaTest/Elasticity_test.h>
#include <SceneCreator/SceneCreator.h>
#include <sofa/defaulttype/VecTypes.h>

//Including Simulation
#include <SofaSimulationGraph/DAGSimulation.h>

#include <SofaMiscFem/TetrahedralTensorMassForceField.h>
#include <SofaGeneralSimpleFem/TetrahedralCorotationalFEMForceField.h>
#include <SofaBaseTopology/TopologySparseData.inl>
#include <SofaBoundaryCondition/TrianglePressureForceField.h>
#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaGeneralEngine/PairBoxRoi.h>
#include <SofaImplicitOdeSolver/StaticSolver.h>
#include <SofaBoundaryCondition/ProjectToLineConstraint.h>
#include <SofaMiscForceField/MeshMatrixMass.h>

namespace sofa {

using namespace component;
using namespace defaulttype;
using namespace modeling;

const double pressureArray[] = {0.6, 0.2, -0.2, -0.6};
const size_t sizePressureArray = sizeof(pressureArray)/sizeof(pressureArray[0]);

const double youngModulusArray[] = {1.0,2.0};
const size_t sizeYoungModulusArray = sizeof(youngModulusArray)/sizeof(youngModulusArray[0]);

const double poissonRatioArray[] = {0.0,0.3,0.49};
const size_t sizePoissonRatioArray = sizeof(poissonRatioArray)/sizeof(poissonRatioArray[0]);


/**  Test force fields implementing linear elasticity on tetrahedral mesh.
Implement traction applied on the top part of a cylinder and test that the deformation
is simply related with the Young Modulus and Poisson Ratio of the isotropc linear elastic material */

template <typename _DataTypes>
struct LinearElasticity_test : public Elasticity_test<_DataTypes>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename container::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename sofa::component::mass::MeshMatrixMass<DataTypes,Real>  MeshMatrixMass;
    typedef typename sofa::component::forcefield::TetrahedralTensorMassForceField<DataTypes> TetrahedralTensorMassForceField;
    typedef typename sofa::core::behavior::ForceField<DataTypes>::SPtr ForceFieldSPtr;
    typedef ForceFieldSPtr (LinearElasticity_test<_DataTypes>::*LinearElasticityFF)(simulation::Node::SPtr,double,double);
    /// Simulation
    simulation::Simulation* simulation;
    /// struct with the pointer of the main components
    CylinderTractionStruct<DataTypes> tractionStruct;
    /// index of the vertex used to compute the compute the deformation
    size_t vIndex;

    // Create the context for the scene
    void SetUp()
    {
        // Init simulation
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
        size_t resolutionCircumferential=7;
        size_t  resolutionRadial=3;
        size_t  resolutionHeight=7;
        size_t maxIteration=3000; // maximum iteration for the CG.

        tractionStruct= this->createCylinderTractionScene(resolutionCircumferential,resolutionRadial,
            resolutionHeight,maxIteration);
        /// take the vertex at mid height and on the surface of the cylinder
        vIndex=(resolutionCircumferential*(resolutionRadial-1)+1)*resolutionHeight/2;
    }
    ForceFieldSPtr addTetrahedralLinearElastic(simulation::Node::SPtr root,
        double youngModulus,double poissonRatio)
    {
        typename TetrahedralTensorMassForceField::SPtr ff=addNew<TetrahedralTensorMassForceField>(root);
        ff->setYoungModulus(youngModulus);
        ff->setPoissonRatio(poissonRatio);
        return (ForceFieldSPtr )ff;
    }
    ForceFieldSPtr addTetrahedralCorotationalFEMLinearElastic(simulation::Node::SPtr root,
        double youngModulus,double poissonRatio)
    {
        typename sofa::component::forcefield::TetrahedralCorotationalFEMForceField<DataTypes>::SPtr ff=addNew<sofa::component::forcefield::TetrahedralCorotationalFEMForceField<DataTypes> >(root);
        ff->setYoungModulus(youngModulus);
        ff->setPoissonRatio(poissonRatio);
        ff->setMethod(0); // small method
        return (ForceFieldSPtr )ff;
    }
    bool testLinearElasticityInTraction(LinearElasticityFF createForceField){

        sofa::simulation::getSimulation()->init(tractionStruct.root.get());

        size_t i,j,k;
        for (k=0;k<sizeYoungModulusArray;++k) {
            Real youngModulus=youngModulusArray[k];
            for (j=0;j<sizePoissonRatioArray;++j) {
                Real poissonRatio=poissonRatioArray[j];
                // create the linear elasticity force field
                ForceFieldSPtr ff=(this->*createForceField)(tractionStruct.root,youngModulus,poissonRatio);
                ff->init();

                for (i=0;i<sizePressureArray;++i) {
                    // set the pressure on the top part
                    Real pressure= pressureArray[i];
                    tractionStruct.forceField.get()->pressure=Coord(0,0,pressure);

                    // reset simulation and init the triangle pressure forcefield
                    sofa::simulation::getSimulation()->reset(tractionStruct.root.get());
                    // sofa::simulation::getSimulation()->init(tractionStruct.root.get());
                    tractionStruct.forceField.get()->init();
                    // record the initial point of a given vertex
                    Coord p0=tractionStruct.dofs.get()->read(sofa::core::ConstVecCoordId::position())->getValue()[vIndex];

                    //  do one step of the static solver
                    sofa::simulation::getSimulation()->animate(tractionStruct.root.get(),0.5);

                    // Get the simulated final position of that vertex
                    Coord p1=tractionStruct.dofs.get()->read(sofa::core::ConstVecCoordId::position())->getValue()[vIndex];
                    // test the young modulus
                    Real longitudinalDeformation=(p1[2]-p0[2])/p0[2];
                    if (fabs(longitudinalDeformation-pressure/youngModulus)>1e-4) {
                        ADD_FAILURE() << "Wrong longitudinal deformation for Young Modulus = " << youngModulus << " Poisson Ratio = "<<
                            poissonRatio << " pressure= "<<pressure<< std::endl <<
                            "Got "<<longitudinalDeformation<< " instead of "<< pressure/youngModulus<< std::endl;
                        return false;
                    }
                    // compute radial deformation
                    p0[2]=0;
                    p1[2]=0;
                    Real radius=p0.norm2();
                    Real radialDeformation= dot(p0,p1)/radius-1 ;
                    // test the Poisson Ratio
                    if (fabs(radialDeformation+pressure*poissonRatio/youngModulus)>2e-4) {
                        ADD_FAILURE() << "Wrong radial deformation for Young Modulus = " << youngModulus << " Poisson Ratio = "<<
                            poissonRatio << " pressure= "<<pressure<< std::endl <<
                            "Got "<<radialDeformation<< " instead of "<< -pressure*poissonRatio/youngModulus<< std::endl;
                        return false;
                    }
                }
                tractionStruct.root->removeObject(ff);
            }
        }
        return true;
    }
    void TearDown()
    {
        if (tractionStruct.root!=NULL)
            sofa::simulation::getSimulation()->unload(tractionStruct.root);
    }

};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(LinearElasticity_test, DataTypes);

// first test topology
TYPED_TEST( LinearElasticity_test , testTractionTensorMass )
{
    //	this->loadScene( "tests/SofaTest/LinearElasticity.scn");
    ASSERT_TRUE( this->testLinearElasticityInTraction(&sofa::LinearElasticity_test<TypeParam>::addTetrahedralLinearElastic));
}

TYPED_TEST( LinearElasticity_test , testTractionCorotational )
{
//	this->loadScene( "tests/SofaTest/LinearElasticity.scn");
    ASSERT_TRUE( this->testLinearElasticityInTraction(&sofa::LinearElasticity_test<TypeParam>::addTetrahedralCorotationalFEMLinearElastic));
}

} // namespace sofa
