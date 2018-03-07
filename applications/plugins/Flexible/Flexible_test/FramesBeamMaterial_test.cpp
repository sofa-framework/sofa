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
#include "stdafx.h"
#include <SofaTest/Elasticity_test.h>
#include <sofa/defaulttype/VecTypes.h>
#include "../types/AffineTypes.h"

//Including Simulation
#include <SofaSimulationGraph/DAGSimulation.h>

#include <SofaBoundaryCondition/SurfacePressureForceField.h>
#include "../material/HookeForceField.h"
#include <SofaBaseMechanics/MechanicalObject.h>

namespace sofa {

using std::cout;
using std::cerr;
using std::endl;
using namespace component;
using namespace defaulttype;
using namespace modeling;
using helper::vector;

const double pressureArray[] = {0.01, 0.05};
const size_t sizePressureArray = sizeof(pressureArray)/sizeof(pressureArray[0]);

const double youngModulusArray[] = {1.0,2.0};
const size_t sizeYoungModulusArray = sizeof(youngModulusArray)/sizeof(youngModulusArray[0]);

const double poissonRatioArray[] = {0.1,0.3};
const size_t sizePoissonRatioArray = sizeof(poissonRatioArray)/sizeof(poissonRatioArray[0]);


/**  Test flexible material. Apply a traction on a beam with affine dofs and
test that the longitudinal deformation is related with the material law.
 */

template <typename _DataTypes>
struct BeamMaterial_test : public Sofa_test<typename Vec3Types::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::EType EType;
	typedef typename Vec3Types::Coord Coord;
	typedef typename Vec3Types::Real Real;
    typedef component::container::MechanicalObject<Affine3Types> AffineMechanicalObject;
    typedef sofa::component::forcefield::HookeForceField<EType> HookeForceField;
    typedef typename sofa::component::forcefield::HookeForceField<EType>::SPtr HookeForceFieldSPtr;
    typedef HookeForceFieldSPtr (BeamMaterial_test<DataTypes>::*LinearElasticityFF)(simulation::Node::SPtr,double,double,double);
    typename component::forcefield::SurfacePressureForceField<Vec3Types>* bottomForceField;
    typename component::forcefield::SurfacePressureForceField<Vec3Types>* topForceField;

    /// Simulation
    simulation::Simulation* simulation;
    /// Affine dofs
    AffineMechanicalObject::SPtr affineDofs;
    /// Root node of the tested scene
    simulation::Node::SPtr root;
	/// index of the vertex used to compute the deformation
	size_t vIndex;
    // Strain node for the force field
    simulation::Node::SPtr strainNode;
   

     // Define the path for the scenes directory
    #define ADD_SOFA_TEST_SCENES_PATH( x ) sofa_tostring(SOFA_TEST_SCENES_PATH)sofa_tostring(x) 

    // Create the context for the scene
    void SetUp()
    { 
        // Init simulation
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
		
        vIndex=1;
 
       //Load the scene
       std::string sceneName = (DataTypes::sceneName);
       std::string fileName = std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + sceneName;
       root = simulation->createNewGraph("root");
       root = down_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()).get() );

       // Get child nodes
       simulation::Node::SPtr flexibleNode = root->getChild("Flexible");
       simulation::Node::SPtr collisionNode = flexibleNode->getChild("collision");
       simulation::Node::SPtr behaviorNode = flexibleNode->getChild("behavior");
       strainNode = behaviorNode->getChild("Strain");

       // Get force field for bottom face
       typedef component::forcefield::SurfacePressureForceField<Vec3Types> SurfacePressureForceField;
       collisionNode->get<SurfacePressureForceField>(bottomForceField,"bottomFF");

       // Get force field for up face
       typedef component::forcefield::SurfacePressureForceField<Vec3Types> SurfacePressureForceField;
       collisionNode->get<SurfacePressureForceField>(topForceField,"upFF");
    
       // Get mechanical object
       affineDofs = flexibleNode->get<AffineMechanicalObject>( root->SearchDown);

    }

	HookeForceFieldSPtr addHookeForceField(simulation::Node::SPtr node,
        double youngModulus,double poissonRatio, double viscosity)
	{
        // Hooke Force Field
        HookeForceFieldSPtr hookeFf = addNew<HookeForceField>(node,"strainMapping");

        // Set young modulus, poisson ratio and viscosity
        vector<Real> youngModulusVec; vector<Real> poissonRatioVec;vector<Real> viscosityVec;
        youngModulusVec.push_back(youngModulus); poissonRatioVec.push_back(poissonRatio);viscosityVec.push_back(viscosity);
        hookeFf->_youngModulus.setValue(youngModulusVec);
        hookeFf->_poissonRatio.setValue(poissonRatioVec);
        hookeFf->_viscosity.setValue(viscosityVec);
        return (HookeForceFieldSPtr )hookeFf;

	}

	bool testHexahedraInTraction(LinearElasticityFF createForceField)
    {
        // Init
		sofa::simulation::getSimulation()->init(root.get());
		size_t i,j,k,l;
        Real viscosity = 1;

        for (k=0;k<sizeYoungModulusArray;++k) 
        {
            // Set young modulus
            Real youngModulus=youngModulusArray[k];
  
            for (j=0;j<sizePoissonRatioArray;++j) 
            {
                // Set Poisson ratio
                Real poissonRatio=poissonRatioArray[j];

                // Create the force field
                HookeForceFieldSPtr ff=(this->*createForceField)(strainNode,youngModulus,poissonRatio,viscosity);

                ff->init();

                for (i=0;i<sizePressureArray;++i) 
                {

                    // Set the pressure on the bottom and top part
                    Real pressure= pressureArray[i];

                    //forceField.get()->pressure=Coord(pressure,0,0);
                    bottomForceField->setPressure(pressure);
                    topForceField->setPressure(pressure);

                    // Reset simulation
                    sofa::simulation::getSimulation()->reset(root.get());
                    
                    // Init the top and bottom pressure forcefield
                    bottomForceField->init();
                    topForceField->init();
                   
                    // Record the initial point of a given vertex
                    typename  AffineMechanicalObject::ReadVecCoord xelasticityDofs0 = affineDofs->readPositions();
                    Vec<3,Real> p0Center = xelasticityDofs0[vIndex].getCenter();
//                    Vec<3,Real> p0CenterRadial = xelasticityDofs0[9].getCenter();

                    //  do several steps of the static solver
                    for(l=0;l<2;++l) 
                    {
                        sofa::simulation::getSimulation()->animate(root.get(),4);
                    }

                    // Get the simulated final position of that vertex
                    typename  AffineMechanicalObject::ReadVecCoord xelasticityDofs = affineDofs->readPositions();
                    Vec<3,Real> p1Center = xelasticityDofs[vIndex].getCenter();
//                    Vec<3,Real> p1CenterRadial = xelasticityDofs[9].getCenter();

                    // Compute longitudinal deformation
                    Real longitudinalDeformation=(p1Center[2]-p0Center[2])/p0Center[2];

                    // test the longitudinal deformation
                    if (fabs((longitudinalDeformation-pressure/youngModulus)/(pressure/youngModulus))>2.7e-1) 
                    {
                        ADD_FAILURE() << "Wrong longitudinal deformation for Young Modulus = " << youngModulus << " Poisson Ratio = "<<
                            poissonRatio << " pressure= "<<pressure<< std::endl <<
                            "Got "<<longitudinalDeformation<< " instead of "<< pressure/youngModulus<< std::endl;
                        return false;
                    }

                }
                root->removeObject(ff);
                if (root!=NULL)
                    sofa::simulation::getSimulation()->unload(root);
                this->SetUp();
                sofa::simulation::getSimulation()->init(root.get());

            }
        }
		return true;
	}
    void TearDown()
    {
        if (root!=NULL)
            sofa::simulation::getSimulation()->unload(root);
    }

};

// Define the types for the test
struct BeamType{
    typedef E332Types EType;
    static const std::string sceneName; 
};
const std::string BeamType::sceneName= "FramesBeamTractionTest.scn";

// Define the list of DataTypes to instanciate
using testing::Types;
typedef testing::Types<
    BeamType
> DataTypes; 

// Test suite for all the instanciations
TYPED_TEST_CASE(BeamMaterial_test, DataTypes);

// Test traction cylinder
TYPED_TEST( BeamMaterial_test , test_Hooke_Beam_InTraction )
{
    ASSERT_TRUE( this->testHexahedraInTraction(&sofa::BeamMaterial_test<TypeParam>::addHookeForceField));
}

} // namespace sofa

