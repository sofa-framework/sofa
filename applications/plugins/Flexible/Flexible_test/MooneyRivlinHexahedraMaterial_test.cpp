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

//Including Simulation
#include <SofaSimulationGraph/DAGSimulation.h>

#include <SofaBoundaryCondition/QuadPressureForceField.h>
#include "../strainMapping/InvariantMapping.h"
#include "../strainMapping/PrincipalStretchesMapping.h"
#include "../material/MooneyRivlinForceField.h"
#include <SofaBaseMechanics/MechanicalObject.h>

namespace sofa {

using std::cout;
using std::cerr;
using std::endl;
using namespace component;
using namespace defaulttype;
using namespace modeling;
using helper::vector;

const size_t sizePressureArray = 11;

const double poissonRatioArray[] = {0.1,0.33,0.49};
const size_t sizePoissonRatioArray = sizeof(poissonRatioArray)/sizeof(poissonRatioArray[0]);

const double pressureMRArray[3][11]={{ -.132595933819742, -0.758890223788378e-1, -0.258991562820281e-1, 0.122841742266160e-1, 0.520772676750806e-1, 0.873930706748338e-1, .118816127431302, .146849902919337, .171927899808457, .194423344562865, .214657633906771}, {-.154930950668924, -0.979576615372482e-1, -0.464710012871213e-1, 0.284218582658924e-1, 0.914766994502278e-1, .144633398240619, .189615987963914, .227903290162599, .260727571353401, .289093880485712, .313810447718268}, { -.174294795457842, -.110585160631557, -0.525879542151690e-1, 0.475417922161212e-1, 0.904460240483829e-1, .129148389744539, .164092090469086, .195709909049912, -0.525879542151690e-1, 0.475417922161212e-1, 0.904460240483829e-1}};
const double s1MRArray[3][11]={{ .889079674146271, .931794051853002, .975065910151025, 1.01251520167355, 1.05652828731312, 1.10078865847721, 1.14519812021085, 1.18966661875371, 1.23411306966208, 1.27846567737549, 1.32266186026694}, {.874088410079668, .914536984204839, .956550854415787, 1.02968913936779, 1.10608130132324, 1.18492093513811, 1.26540562751559, 1.34680469638069, 1.42849860927677, 1.50999193035351, 1.59090769085560}, {.861904401269021, .905185951795058, .951263797501961, 1.05118074082972, 1.10452763290780, 1.15971595999524, 1.21639661336373, 1.27421792985636, .951263797501961, 1.05118074082972, 1.10452763290780}};
const double s2MRArray[3][11]={{1.01169675110348, 1.00706137054476, 1.00252693873968, .998757171436292, .994529295679769, .990512372245131, .986727246300965, .983187350362974, .979899868402788, .976866865097434, .974086327685361}, { 1.04525921482584, 1.02987669475421, 1.01476173719551, .990393133930705, .967334438704431, .945819462262607, .925976549680711, .907839716156012, .891370013246562, .876479335741178, .863051445950385},{1.07551885873374, 1.05001709087481, 1.02478381944566, .975839382171100, .952457416859653, .929982228608163, .908509044567281, .888098578545382, 1.02478381944566, .975839382171100, .952457416859653}};

/**  Test flexible material. Apply a traction on the top part of an hexahedra and
test that the longitudinal and radial deformation are related with the material law.
 */

template <typename _DataTypes>
struct MooneyRivlinHexahedraMaterial_test : public Sofa_test<typename Vec3Types::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::StrainType StrainType;
    typedef typename DataTypes::StrainMapping StrainMapping;
	typedef typename Vec3Types::Coord Coord;
	typedef typename Vec3Types::Real Real;
    typedef const double dataArray[3][11];
    typedef typename container::MechanicalObject<Vec3Types> MechanicalObject;
    typedef container::MechanicalObject<StrainType> StrainDOFs;
    typedef typename container::MechanicalObject<StrainType>::SPtr strainDOFsSPtr;
    typedef sofa::component::forcefield::MooneyRivlinForceField<StrainType> MooneyRivlinForceField;
    typedef typename sofa::component::forcefield::MooneyRivlinForceField<StrainType>::SPtr MooneyRivlinForceFieldSPtr;
    typedef typename sofa::core::behavior::ForceField<StrainType>::SPtr ForceFieldSPtr;
    typedef ForceFieldSPtr (MooneyRivlinHexahedraMaterial_test<DataTypes>::*LinearElasticityFF)(simulation::Node::SPtr,double,double);
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
		
        /// index of the vertex used to compute the deformation
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

    ForceFieldSPtr addMooneyRivlinForceField(simulation::Node::SPtr node,
        double youngModulus,double poissonRatio)
    {
        // Mooney Rivlin Force Field
        MooneyRivlinForceFieldSPtr hookeFf = addNew<MooneyRivlinForceField>(node,"strainMapping");
        Real lambda=youngModulus*poissonRatio/((1+poissonRatio)*(1-2*poissonRatio));
        Real mu=youngModulus/(2*(1+poissonRatio));
        Real bulkModulus=lambda+2*mu/3;
        vector<Real> c1Vec; vector<Real> c2Vec;vector<Real> bulkModulusVec;
        c1Vec.push_back(mu/4); c2Vec.push_back(mu/4);bulkModulusVec.push_back(bulkModulus);
        hookeFf->f_C1.setValue(c1Vec);
        hookeFf->f_C2.setValue(c2Vec);
        hookeFf->f_bulk.setValue(bulkModulusVec);
        return (ForceFieldSPtr )hookeFf;
    }

    bool testHexahedraInTraction(LinearElasticityFF createForceField, dataArray pressureArray, 
            dataArray s1Array,dataArray s2Array,double longitudinalStretchAccuracy,double radialStretchAccuracy, bool debug)
    {
        // Init
		sofa::simulation::getSimulation()->init(tractionStruct.root.get());
		size_t i,j,l;

        // Set young modulus
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
                std::cout << "precision longitudinal stretch = " << fabs((longitudinalStretch-s1Array[j][i])/(s1Array[j][i])) << std::endl;
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
                Real radialStretch= dot(p0,p1)/radius;

                // Test if radial stretch is a nan value
                if(radialStretch != radialStretch)
                {
                    ADD_FAILURE() << "Error radial stretch is NAN" << std::endl;
                    return false;
                }

                // test the radial deformation
                if(debug)
                std::cout << "precision radial stretch = " << fabs((radialStretch-s2Array[j][i])/(s2Array[j][i])) << std::endl;
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

// Define the types for the test

// Invariant mapping
struct TypeInvariantMRHexaTest{
    typedef I331Types StrainType;
    typedef mapping::InvariantMapping<defaulttype::F331Types,defaulttype::I331Types> StrainMapping;
    static const double longitudinalStretchAccuracy; 
    static const double radialStretchAccuracy;
    static const std::string sceneName; 
};
const double TypeInvariantMRHexaTest::longitudinalStretchAccuracy= 3.5e-2; // Accuracy of longitudinal stretch
const double TypeInvariantMRHexaTest::radialStretchAccuracy= 5e-2; // Accuracy of radial stretch
const std::string TypeInvariantMRHexaTest::sceneName= "StaticSolverMrNhHexahedraTractionTest.scn"; // Scene to test

// Principal Stretches mapping
struct TypePrincipalStretchesMRHexaTest{
    typedef U331Types StrainType;
    typedef mapping::PrincipalStretchesMapping<defaulttype::F331Types,defaulttype::U331Types> StrainMapping;
    static const double longitudinalStretchAccuracy;
    static const double radialStretchAccuracy; 
    static const std::string sceneName; 
};
const double TypePrincipalStretchesMRHexaTest::longitudinalStretchAccuracy= 2.1e-1; // Accuracy of longitudinal stretch
const double TypePrincipalStretchesMRHexaTest::radialStretchAccuracy= 2.3e-1; // Accuracy of radial stretch
const std::string TypePrincipalStretchesMRHexaTest::sceneName= "AssembledSolverMrNhHexahedraTractionTest.scn"; // Scene to test


// Define the list of DataTypes to instanciate
using testing::Types;
typedef testing::Types<
    TypeInvariantMRHexaTest,
    TypePrincipalStretchesMRHexaTest
> DataTypes; 

// Test suite for all the instanciations
TYPED_TEST_CASE(MooneyRivlinHexahedraMaterial_test, DataTypes);

TYPED_TEST( MooneyRivlinHexahedraMaterial_test , test_MR_Hexahedra_InTraction )
{
    ASSERT_TRUE( this->testHexahedraInTraction(&sofa::MooneyRivlinHexahedraMaterial_test<TypeParam>::addMooneyRivlinForceField,pressureMRArray,s1MRArray, s2MRArray,
                                                TypeParam::longitudinalStretchAccuracy,TypeParam::radialStretchAccuracy,false));
}


} // namespace sofa


