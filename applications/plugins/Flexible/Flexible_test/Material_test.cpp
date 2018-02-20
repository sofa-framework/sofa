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
#include <SceneCreator/SceneCreator.h>
#include <sofa/defaulttype/VecTypes.h>

//Including Simulation
#include <SofaSimulationGraph/DAGSimulation.h>

#include <SofaBoundaryCondition/TrianglePressureForceField.h>
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

const double pressureArray[] = {0.6, 0.2,-0.3};
const size_t sizePressureArray = sizeof(pressureArray)/sizeof(pressureArray[0]);

const double youngModulusArray[] = {1.0,2.0};
const size_t sizeYoungModulusArray = sizeof(youngModulusArray)/sizeof(youngModulusArray[0]);

const double poissonRatioArray[] = {0.1,0.3};
const size_t sizePoissonRatioArray = sizeof(poissonRatioArray)/sizeof(poissonRatioArray[0]);


/**  Test flexible material. Apply a traction on the top part of a discretized cylinder and
test that the longitudinal and radial deformation are related with the material law.
 */

template <typename _DataTypes>
struct Material_test : public Sofa_test<typename Vec3Types::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::EType EType;
	typedef typename Vec3Types::Coord Coord;
	typedef typename Vec3Types::Real Real;
    typedef typename container::MechanicalObject<Vec3Types> MechanicalObject;
    typedef sofa::component::forcefield::HookeForceField<EType> HookeForceField;
    typedef typename sofa::component::forcefield::HookeForceField<EType>::SPtr HookeForceFieldSPtr;
    typedef HookeForceFieldSPtr (Material_test<DataTypes>::*LinearElasticityFF)(simulation::Node::SPtr,double,double,double);
    
    /// Simulation
    simulation::Simulation* simulation;
	/// struct with the pointer of the main components 
	CylinderTractionStruct<Vec3Types> tractionStruct;
	/// index of the vertex used to compute the compute the deformation
	size_t vIndex;
    // Strain node for the force field
    simulation::Node::SPtr strainNode;

    // Create the context for the scene
    void SetUp()
    { 
        // Init simulation
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
		size_t resolutionCircumferential=7;
		size_t  resolutionRadial=3;
		size_t  resolutionHeight=7;
//		size_t maxIteration=3000; // maximum iteration for the CG.
      
       vIndex=(resolutionCircumferential*(resolutionRadial-1)+1)*resolutionHeight/2;
    
       //Load the scene
       std::string sceneName = (DataTypes::sceneName);
       std::string fileName = std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + sceneName;
       tractionStruct.root = simulation->createNewGraph("root");
       tractionStruct.root = down_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()).get() );

       // Get force field
       typedef component::forcefield::TrianglePressureForceField<Vec3Types> TrianglePressureForceField;
       tractionStruct.forceField = tractionStruct.root->get<TrianglePressureForceField>( tractionStruct.root->SearchDown);

       // Get mechanical object
       typedef component::container::MechanicalObject<Vec3Types> MechanicalObject;
       tractionStruct.dofs = tractionStruct.root->get<MechanicalObject>( tractionStruct.root->SearchDown);

       // Get child nodes
       simulation::Node::SPtr behaviorNode = tractionStruct.root->getChild("behavior");
       strainNode = behaviorNode->getChild("Strain");
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

	bool testCylinderInTraction(LinearElasticityFF createForceField)
    {
        // Init
		sofa::simulation::getSimulation()->init(tractionStruct.root.get());
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

                    // Set the pressure on the top part
                    Real pressure= pressureArray[i];
                    tractionStruct.forceField.get()->pressure=Coord(0,0,pressure);

                    // Reset simulation
                    sofa::simulation::getSimulation()->reset(tractionStruct.root.get());
                    // Init the triangle pressure forcefield
                    tractionStruct.forceField.get()->init();
                    // Record the initial point of a given vertex
                    Coord p0=tractionStruct.dofs.get()->read(core::ConstVecCoordId::position())->getValue()[vIndex];

                    //  do several steps of the static solver
                    for(l=0;l<10;++l) 
                    {
                        sofa::simulation::getSimulation()->animate(tractionStruct.root.get(),0.5);
                    }

                    // Get the simulated final position of that vertex
                    Coord p1=tractionStruct.dofs.get()->read(core::ConstVecCoordId::position())->getValue()[vIndex];
                     Real longitudinalDeformation=(p1[2]-p0[2])/p0[2];
                    // Test if longitudinal stretch is a nan value
                    if(longitudinalDeformation != longitudinalDeformation)
                    {
                        ADD_FAILURE() << "Error longitudinal stretch is NAN" << std::endl;
                        return false;
                    }

                    // test the longitudinal deformation
                    if (fabs((longitudinalDeformation-pressure/youngModulus)/(pressure/youngModulus))>5.2e-3) {
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

                    // Test if radial stretch is a nan value
                    if(radialDeformation != radialDeformation)
                    {
                        ADD_FAILURE() << "Error radial stretch is NAN" << std::endl;
                        return false;
                    }

                    // test the radial deformation
                    if (fabs((radialDeformation+pressure*poissonRatio/youngModulus)/(pressure*poissonRatio/youngModulus))>5.2e-3)
                    {
                        ADD_FAILURE() << "Wrong radial deformation for Young Modulus = " << youngModulus << " Poisson Ratio = "<<
                            poissonRatio << " pressure= "<<pressure<< std::endl <<
                            "Got "<<radialDeformation<< " instead of "<< -pressure*poissonRatio/youngModulus<< std::endl;
                        return false;
                    }
                }
                tractionStruct.root->removeObject(ff);
                if (tractionStruct.root!=NULL)
                    sofa::simulation::getSimulation()->unload(tractionStruct.root);
                this->SetUp();
                sofa::simulation::getSimulation()->init(tractionStruct.root.get());

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

// Define the types for the test
struct Type1{
    typedef E331Types EType;
    static const std::string sceneName ; 
};
const std::string Type1::sceneName= "CylinderTraction.scn";

struct Type2{
    typedef E332Types EType;
    static const std::string sceneName; 
};
const std::string Type2::sceneName= "CylinderTractionElaston.scn";

// Define the list of DataTypes to instanciate
using testing::Types;
typedef testing::Types<
    Type1,
    Type2
> DataTypes; 

// Test suite for all the instanciations
TYPED_TEST_CASE(Material_test, DataTypes);

// Test traction cylinder
TYPED_TEST( Material_test , testTractionCylinder )
{
    ASSERT_TRUE( this->testCylinderInTraction(&sofa::Material_test<TypeParam>::addHookeForceField));
}

} // namespace sofa

