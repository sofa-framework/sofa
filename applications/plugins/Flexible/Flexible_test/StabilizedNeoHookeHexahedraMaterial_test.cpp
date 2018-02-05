/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

//Including Simulation
#include <SofaSimulationGraph/DAGSimulation.h>

#include <SofaBoundaryCondition/QuadPressureForceField.h>
#include "../strainMapping/InvariantMapping.h"
#include "../strainMapping/PrincipalStretchesMapping.h"
#include "../material/StabilizedNeoHookeanForceField.h"
#include <SofaBaseMechanics/MechanicalObject.h>

namespace sofa {

using std::cout;
using std::cerr;
using std::endl;
using namespace component;
using namespace defaulttype;
using namespace modeling;
using helper::vector;

const size_t sizePressureArray = 8;

const double poissonRatioArray[] = {0.1,0.33,0.49};
const size_t sizePoissonRatioArray = sizeof(poissonRatioArray)/sizeof(poissonRatioArray[0]);

const double pressureNHArray[3][8]={{-0.0889597892, -0.0438050358, 0.0635053964, 0.1045737959, 0.1447734054,  0.2230181491, 0.2612702806, 0.2990693371},{  0.0908043726, 0.1442067021, 0.1972023146, 0.2501382493, 0.3033829162,  0.3573334637, 0.4124247244, 0.4691402652},{0.0506740890, 0.1015527192, 0.1529835922, 0.2053969789,    0.2593326498, 0.3154780834, 0.3747238572, 0.4382459743}};
const double s1NHArray[3][8]={{0.9253724143, 0.9621812232, 1.058685691, 1.099159142, 1.140749790, 1.227419011, 1.272569570, 1.318981460},{1.095547169, 1.158492292, 1.226210815, 1.299242414, 1.378211471, 1.463844371, 1.556991211, 1.658653303},{1.053287121, 1.112386732, 1.178304553, 1.252292527, 1.335929089, 1.431233010, 1.540828471, 1.668190396}};
const double s2NHArray[3][8]={{1.018538248, 1.009217097, 0.9863591796, 0.9773820557, 0.9684934549, 0.9509661681, 0.9423200997, 0.9337477636},{0.9672670466, 0.9474800960, 0.9275635172, 0.9075003349, 0.8872723319, 0.8668598514, 0.8462415656, 0.8253942001},{0.9748631443, 0.9490868600, 0.9226175014, 0.8953935507, 0.8673438984, 0.8383856153, 0.8084210028, 0.7773336312}};

/**  Test flexible material. Apply a traction on the top part of an hexahedra and
test that the longitudinal and radial deformation are related with the material law.
 */

template <typename _DataTypes>
struct StabilizedNeoHookeHexahedraMaterial_test : public Sofa_test<typename Vec3Types::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::StrainType StrainType;
    typedef typename DataTypes::StrainMapping StrainMapping;
	typedef typename Vec3Types::Coord Coord;
	typedef typename Vec3Types::Real Real;
    typedef const double dataArray[3][8];
    typedef typename container::MechanicalObject<Vec3Types> MechanicalObject;
    typedef container::MechanicalObject<StrainType> StrainDOFs;
    typedef typename container::MechanicalObject<StrainType>::SPtr strainDOFsSPtr;
    typedef sofa::component::forcefield::StabilizedNeoHookeanForceField<StrainType> StabilizedNeoHookeanForceField;
    typedef typename sofa::component::forcefield::StabilizedNeoHookeanForceField<StrainType>::SPtr StabilizedNeoHookeanForceFieldSPtr;
    typedef typename sofa::core::behavior::ForceField<StrainType>::SPtr ForceFieldSPtr;
    typedef ForceFieldSPtr (StabilizedNeoHookeHexahedraMaterial_test<DataTypes>::*LinearElasticityFF)(simulation::Node::SPtr,double,double);
    typename component::forcefield::QuadPressureForceField<Vec3Types>::SPtr pressureForceField;

    /// Simulation
    simulation::Simulation* simulation;
	/// struct with the pointer of the main components 
	CylinderTractionStruct<Vec3Types> tractionStruct;
	/// index of the vertex used to compute the deformation
	size_t vIndex;
    // Strain node for the force field
    simulation::Node::SPtr strainNode;

    // Create the context for the scene
    void SetUp()
    { 
        // Init simulation
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
		
        vIndex=25;
 
       //Load the scene
       std::string sceneName = (DataTypes::sceneName);
       std::string fileName = std::string(FLEXIBLE_TEST_SCENES_DIR) + "/" + sceneName;
       tractionStruct.root = simulation->createNewGraph("root");
       tractionStruct.root = down_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()).get() );

       // Get child nodes
       simulation::Node::SPtr quadNode = tractionStruct.root->getChild("Quads");
       simulation::Node::SPtr behaviorNode = tractionStruct.root->getChild("behavior");
       strainNode = behaviorNode->getChild("Strain");

       // Get force field
       typedef component::forcefield::QuadPressureForceField<Vec3Types> QuadPressureForceField;
       pressureForceField = quadNode->get<QuadPressureForceField>( tractionStruct.root->SearchDown);
    
       // Get mechanical object
       typedef component::container::MechanicalObject<Vec3Types> MechanicalObject;
       tractionStruct.dofs = tractionStruct.root->get<MechanicalObject>( tractionStruct.root->SearchDown);

       // Get behavior mechanical object
       typedef component::container::MechanicalObject<F331Types> BehaviorMechanicalObject;
       BehaviorMechanicalObject::SPtr behaviorDofs = behaviorNode->get<BehaviorMechanicalObject>( behaviorNode->SearchDown);

       // Add strain mechanical object
       strainDOFsSPtr strainDOFs = addNew<StrainDOFs>(strainNode);
       // Add strain mapping
       typename StrainMapping::SPtr strainMapping = addNew<StrainMapping>(strainNode);
       strainMapping->setModels(behaviorDofs.get(),strainDOFs.get());

    }

    ForceFieldSPtr addStabilizedNeoHookeanForceField(simulation::Node::SPtr node,
        double youngModulus,double poissonRatio)
    {
        // Hooke Force Field
        StabilizedNeoHookeanForceFieldSPtr ff = addNew<StabilizedNeoHookeanForceField>(node,"strainMapping");

        // Set young modulus and poisson ratio
        vector<Real> youngModulusVec; vector<Real> poissonRatioVec;
        youngModulusVec.push_back(youngModulus); poissonRatioVec.push_back(poissonRatio);
        ff->_youngModulus.setValue(youngModulusVec);
        ff->_poissonRatio.setValue(poissonRatioVec);
        return (ForceFieldSPtr )ff;

    }

    bool testHexahedraInTraction(LinearElasticityFF createForceField, dataArray pressureArray, 
            dataArray s1Array,dataArray s2Array,double longitudinalStretchAccuracy,double radialStretchAccuracy,bool debug)
    {
        // Init
		sofa::simulation::getSimulation()->init(tractionStruct.root.get());
		size_t i,j,l;

        Real youngModulus=1.0;
  
        for (j=0;j<sizePoissonRatioArray;++j) 
        {
            // Set Poisson ratio
            Real poissonRatio=poissonRatioArray[j];

            // Create the force field
            ForceFieldSPtr ff = (this->*createForceField)(strainNode,youngModulus,poissonRatio);

            ff->init();

            for (i=0;i<sizePressureArray;++i) 
            {

                // Set the pressure on the top part
                Real pressure= pressureArray[j][i];

                pressureForceField.get()->pressure=Coord(pressure,0,0);

                // Reset simulation
                sofa::simulation::getSimulation()->reset(tractionStruct.root.get());
                    
                // Init the triangle pressure forcefield
                pressureForceField.get()->init();
                   
                // Record the initial point of a given vertex
                Coord p0=tractionStruct.dofs.get()->read(core::ConstVecCoordId::position())->getValue()[vIndex];

                //  do several steps of the implicit solver
                for(l=0;l<8;++l) 
                {
                    sofa::simulation::getSimulation()->animate(tractionStruct.root.get(),0.5);
                }

                // Get the simulated final position of that vertex
                Coord p1=tractionStruct.dofs.get()->read(core::ConstVecCoordId::position())->getValue()[vIndex];

                // Compute longitudinal deformation
                Real longitudinalStretch=p1[0]/p0[0];
                    
                // Test if longitudinal stretch is a nan value
                if(longitudinalStretch != longitudinalStretch)
                {
                    ADD_FAILURE() << "Error longitudinal stretch is NAN" << std::endl;
                    return false;
                }

                // test the longitudinal deformation
                if(debug)
                std::cout << "precision longitudinal deformation = " << fabs((longitudinalStretch-s1Array[j][i])/(s1Array[j][i])) << std::endl;
                if (fabs((longitudinalStretch-s1Array[j][i])/(s1Array[j][i]))>longitudinalStretchAccuracy) 
                {
                    ADD_FAILURE() << "Wrong longitudinal stretch for Poisson Ratio = "<<
                        poissonRatio << " pressure= "<<pressure<< std::endl <<
                        "Got "<<longitudinalStretch<< " instead of "<< s1Array[j][i]<< std::endl;
                    return false;
                }

                // compute radial deformation
                p0[0]=0;
                p1[0]=0;
                p0[1]=0;
                p1[1]=0;
                Real radius=p0.norm2();
                Real radialStretch= dot(p0,p1)/radius;//-1 ;

                // Test if radial stretch is a nan value
                if(radialStretch != radialStretch)
                {
                    ADD_FAILURE() << "Error radial stretch is NAN" << std::endl;
                    return false;
                }

                // test the radial deformation
                if(debug)
                std::cout << "precision radial deformation = " << fabs((radialStretch-s2Array[j][i])/(s2Array[j][i])) << std::endl;
                if (fabs((radialStretch-s2Array[j][i])/(s2Array[j][i]))>radialStretchAccuracy) 
                {
                    ADD_FAILURE() << "Wrong radial stretch for  Poisson Ratio = "<<
                        poissonRatio << " pressure= "<<pressure<< std::endl <<
                        "Got "<<radialStretch<< " instead of "<< s2Array[j][i]<< std::endl;
                    return false;
                }
            }
            tractionStruct.root->removeObject(ff);
            if (tractionStruct.root!=NULL)
                sofa::simulation::getSimulation()->unload(tractionStruct.root);
            this->SetUp();
            sofa::simulation::getSimulation()->init(tractionStruct.root.get());

        }
        
		return true;
	}

    void TearDown()
    {
        if (tractionStruct.root!=NULL)
            sofa::simulation::getSimulation()->unload(tractionStruct.root);
    }

};

// Principal Stretches mapping U331
struct TypePrincipalStretchesStabilizedNHHexaTest{
    typedef U331Types StrainType;
    typedef mapping::PrincipalStretchesMapping<defaulttype::F331Types,defaulttype::U331Types> StrainMapping;
    static const double longitudinalStretchAccuracy;
    static const double radialStretchAccuracy; 
    static const std::string sceneName; 
};
const double TypePrincipalStretchesStabilizedNHHexaTest::longitudinalStretchAccuracy= 1.5e-1; // Accuracy of longitudinal stretch
const double TypePrincipalStretchesStabilizedNHHexaTest::radialStretchAccuracy= 5.4e-2; // Accuracy of radial stretch
const std::string TypePrincipalStretchesStabilizedNHHexaTest::sceneName= "AssembledSolverMrNhHexahedraTractionTest.scn"; // Scene to test


// Define the list of DataTypes to instanciate
using testing::Types;
typedef testing::Types<
    TypePrincipalStretchesStabilizedNHHexaTest
> DataTypes; 


// Test suite for all the instanciations
TYPED_TEST_CASE(StabilizedNeoHookeHexahedraMaterial_test, DataTypes);

// Test NeoHooke with principal stretches mapping
TYPED_TEST( StabilizedNeoHookeHexahedraMaterial_test , test_StabilizedNeoHooke_Hexahedra_InTraction )
{
    ASSERT_TRUE( this->testHexahedraInTraction(&sofa::StabilizedNeoHookeHexahedraMaterial_test<TypeParam>::addStabilizedNeoHookeanForceField,pressureNHArray,s1NHArray, s2NHArray,
                                                TypeParam::longitudinalStretchAccuracy,TypeParam::radialStretchAccuracy,false));
}

} // namespace sofa


