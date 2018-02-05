/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTest/Sofa_test.h>
#include<sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

// Including constraint, force and mass
#include <SofaBaseTopology/BezierTetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/BezierTetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <SceneCreator/SceneCreator.h>
#include <SofaMiscForceField/MeshMatrixMass.h>
#include <SofaEngine/GenerateCylinder.h>
#include <SofaTopologyMapping/Mesh2BezierTopologicalMapping.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa {

using std::cout;
using std::cerr;
using std::endl;
using namespace component;
using namespace defaulttype;

/**  Test the topology and geometry of Bezier Tetrahedral meshes. 
No deformation is applied but only the init() function is called to create a Bezier Tetrahedral mesh from a a regular tetraheral mesh */

template <typename _DataTypes>
struct BezierTetrahedronTopology_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
	typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::Real Real;
    typedef typename sofa::component::topology::BezierTetrahedronSetTopologyContainer BezierTetrahedronSetTopologyContainer;
	typedef typename sofa::component::topology::BezierTetrahedronSetGeometryAlgorithms<DataTypes> BezierTetrahedronSetGeometryAlgorithms;
	typedef typename sofa::component::topology::BezierDegreeType BezierDegreeType;
	typedef typename sofa::component::topology::TetrahedronBezierIndex TetrahedronBezierIndex;
	typedef typename BezierTetrahedronSetTopologyContainer::BezierTetrahedronPointLocation BezierTetrahedronPointLocation;
    typedef typename container::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename sofa::component::mass::MeshMatrixMass<DataTypes,Real>  MeshMatrixMass;

    
    /// Root of the scene graph
    simulation::Node::SPtr root;      
    /// Simulation
    simulation::Simulation* simulation;  

    // Create the context for the scene
    void SetUp()
    { 
        // Init simulation
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

         root = simulation::getSimulation()->createNewGraph("root");
    }
	 // Load the scene BezierTetrahedronTopology.scn from the scenes directory
    void loadScene(std::string sceneName)
    {
        // Load the scene from the xml file
	std::string fileName = std::string(SOFABASETOPOLOGY_TEST_SCENES_DIR) + "/" + sceneName;
        root = down_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()).get() );
    }
	 // Load the scene BezierTetrahedronTopology.sc from the Scenes directory
    void createScene()
    {
		// GenerateCylinder object
        typename sofa::component::engine::GenerateCylinder<DataTypes>::SPtr eng= sofa::modeling::addNew<sofa::component::engine::GenerateCylinder<DataTypes> >(root,"cylinder");
		eng->f_radius=0.2;
		eng->f_height=1.0;
		// TetrahedronSetTopologyContainer object
		sofa::component::topology::TetrahedronSetTopologyContainer::SPtr container1= sofa::modeling::addNew<sofa::component::topology::TetrahedronSetTopologyContainer>(root,"Container1");
		sofa::modeling::setDataLink(&eng->f_tetrahedra,&container1->d_tetrahedron);
		sofa::modeling::setDataLink(&eng->f_outputTetrahedraPositions,&container1->d_initPoints);
		// TetrahedronSetGeometryAlgorithms object
        typename sofa::component::topology::TetrahedronSetGeometryAlgorithms<DataTypes>::SPtr geo1= sofa::modeling::addNew<sofa::component::topology::TetrahedronSetGeometryAlgorithms<DataTypes> >(root);
		// mechanicalObject object
        typename MechanicalObject::SPtr meca1= sofa::modeling::addNew<MechanicalObject>(root);
		sofa::modeling::setDataLink(&eng->f_outputTetrahedraPositions,&meca1->x);
		// subnode
	    simulation::Node::SPtr bezierNode = root->createChild("BezierTetrahedronTopology");
		// BezierTetrahedronSetTopologyContainer
		sofa::component::topology::BezierTetrahedronSetTopologyContainer::SPtr container2= sofa::modeling::addNew<sofa::component::topology::BezierTetrahedronSetTopologyContainer>(bezierNode,"Container2");
		// Mesh2BezierTopologicalMapping
		sofa::component::topology::Mesh2BezierTopologicalMapping::SPtr mapping= sofa::modeling::addNew<sofa::component::topology::Mesh2BezierTopologicalMapping>(bezierNode,"Mapping");
		mapping->setTopologies(container1.get(),container2.get());
		mapping->bezierTetrahedronDegree=3;
		// mechanicalObject object
        typename MechanicalObject::SPtr meca2= sofa::modeling::addNew<MechanicalObject>(bezierNode,"BezierMechanicalObject");
		// BezierTetrahedronSetGeometryAlgorithms
        typename sofa::component::topology::BezierTetrahedronSetGeometryAlgorithms<DataTypes>::SPtr geo2= sofa::modeling::addNew<sofa::component::topology::BezierTetrahedronSetGeometryAlgorithms<DataTypes> >(bezierNode);
		// MeshMatrixMass
        typename MeshMatrixMass::SPtr mass= sofa::modeling::addNew<MeshMatrixMass >(bezierNode,"BezierMass");
		mass->m_massDensity=1.0;
		mass->d_integrationMethod.setValue(std::string("analytical"));
	}
	bool testBezierTetrahedronTopology()
	{
		// Init simulation
		sofa::simulation::getSimulation()->init(root.get());
		BezierTetrahedronSetTopologyContainer *container=root->get<BezierTetrahedronSetTopologyContainer>(root->SearchDown);
		size_t nTetras,elem;
		BezierDegreeType degree=container->getDegree();
		// check the total number of vertices.
		size_t nbPoints=container->getNumberOfTetrahedralPoints()+container->getNumberOfEdges()*(degree-1)+container->getNumberOfTriangles()*(degree-1)*(degree-2)/2+container->getNumberOfTetrahedra()*((degree-1)*(degree-2)*(degree-3)/6);
        if((size_t)container->getNbPoints()!=nbPoints) {
			ADD_FAILURE() << "wrong number of points " <<container->getNbPoints() << " is wrong. It should be  " <<nbPoints  << std::endl;
			return false;
		}

		sofa::helper::vector<TetrahedronBezierIndex> tbiArray=container->getTetrahedronBezierIndexArray();
		
		BezierTetrahedronPointLocation location; 
		size_t elementIndex, elementOffset/*,localIndex*/;
		for (nTetras=0;nTetras<container->getNumberOfTetrahedra();++nTetras) {
			
			const BezierTetrahedronSetTopologyContainer::VecPointID &indexArray=container->getGlobalIndexArrayOfBezierPoints(nTetras);
			// check the number of control points per tetrahedron is correct
			nbPoints=(4+6*(degree-1)+2*(degree-1)*(degree-2)+(degree-1)*(degree-2)*(degree-3)/6);
			if(indexArray.size()!=nbPoints) {
				ADD_FAILURE() << "wrong number of control points in tetrahedron " <<nTetras<< ". It is "<<indexArray.size() <<" and should be "<<nbPoints  << std::endl;
				return false;
			}
			for(elem=0;elem<indexArray.size();++elem) {
				size_t globalIndex=container->getGlobalIndexOfBezierPoint(nTetras,tbiArray[elem]);
				// check that getGlobalIndexOfBezierPoint and getGlobalIndexArrayOfBezierPointsInTetrahedron give the same answer
				if(globalIndex!=indexArray[elem]) {
					ADD_FAILURE() << "wrong global index given by  getGlobalIndexOfBezierPoint(). It is : "<<globalIndex <<" and should be "<<indexArray[elem]  << std::endl;
					return false;
				}
				TetrahedronBezierIndex tbi=container->getTetrahedronBezierIndex(elem);
				if(elem!=container->getLocalIndexFromTetrahedronBezierIndex(tbi)) {
					ADD_FAILURE() << "wrong local index given by  getLocalIndexFromTetrahedronBezierIndex(). It is : "<<container->getLocalIndexFromTetrahedronBezierIndex(tbi) <<" and should be "<<elem  << std::endl;
					return false;
				}
				// check that getTetrahedronBezierIndex is consistant with getTetrahedronBezierIndexArray
				if ((tbiArray[elem][0]!=tbi[0]) || (tbiArray[elem][1]!=tbi[1]) || (tbiArray[elem][2]!=tbi[2]) || (tbiArray[elem][3]!=tbi[3])) {
					ADD_FAILURE() << "non consistent indices between getTetrahedronBezierIndexArray() and getTetrahedronBezierIndex(). Got  : "<<tbiArray[elem] <<" versus  "<<tbi  << std::endl;
					return false;
				}
				// check that getLocationFromGlobalIndex is consistent with 
				container->getLocationFromGlobalIndex(globalIndex,location,elementIndex,elementOffset);
				if (elem<4) {
					if ((location!=BezierTetrahedronSetTopologyContainer::POINT) || (elementIndex!=container->getTetrahedron(nTetras)[elem]) || (elementOffset!=0)) {
						ADD_FAILURE() << "non consistent indices given by  getLocationFromGlobalIndex() for global index : "<<globalIndex <<std::endl;
						return false;
					}
				}
				else if (elem<(size_t)(4+6*(degree-1))){
					if ((location!=BezierTetrahedronSetTopologyContainer::EDGE) || (elementIndex!=container->getEdgesInTetrahedron(nTetras)[(elem-4)/(degree-1)])) {
						ADD_FAILURE() << "non consistent indices given by  getLocationFromGlobalIndex() for global index : "<<globalIndex <<std::endl;
						return false;
					}

				}
				else if (elem<(size_t)(4+6*(degree-1)+2*(degree-1)*(degree-2))){
					size_t nbPointPerEdge=(degree-1)*(degree-2)/2;
					size_t val=(elem-4-6*(degree-1))/(nbPointPerEdge);
					if ((location!=BezierTetrahedronSetTopologyContainer::TRIANGLE) || (elementIndex!=container->getTrianglesInTetrahedron(nTetras)[val])) {
						ADD_FAILURE() << "non consistent indices given by  getLocationFromGlobalIndex() for global index : "<<globalIndex <<std::endl;
						return false;
					}
				}
			}

		}
		return( true);
	}
	bool testBezierTetrahedronGeometry()
	{
		BezierTetrahedronSetTopologyContainer *container=root->get<BezierTetrahedronSetTopologyContainer>(root->SearchDown);
		typename MechanicalObject::SPtr dofs = root->get<MechanicalObject>(std::string("BezierTetrahedronTopology/"));
		typename MechanicalObject::WriteVecCoord coords = dofs->writePositions();
		size_t i,j;	
		BezierDegreeType degree=container->getDegree();

		sofa::helper::vector<TetrahedronBezierIndex> tbiArray=container->getTetrahedronBezierIndexArray();
		for ( i = 0; i<container->getNumberOfTetrahedra(); i++)
		{
			
			const BezierTetrahedronSetTopologyContainer::VecPointID &indexArray=container->getGlobalIndexArrayOfBezierPoints(i);
			

			for (j=0;j<tbiArray.size();++j) {

				if (j>=4) {
					// test if the position is correct
					Coord pos=coords[indexArray[0]]*(Real)tbiArray[j][0]/degree+coords[indexArray[1]]*(Real)tbiArray[j][1]/degree+coords[indexArray[2]]*(Real)tbiArray[j][2]/degree+coords[indexArray[3]]*(Real)tbiArray[j][3]/degree;
					if ((pos-coords[indexArray[j]]).norm()>1e-5) {
						ADD_FAILURE() << "Wrong control point position in tetrahedron no  : "<<i <<" for point of local index " <<j
						<< " Got point position="<<coords[indexArray[j]]<<" instead of "<<pos<<std::endl;
						return false;
					}
				}

			}
		}
		return true;
	}
	bool testBezierTetrahedronMass()
	{
		BezierTetrahedronSetTopologyContainer *container=root->get<BezierTetrahedronSetTopologyContainer>(root->SearchDown);
		BezierTetrahedronSetGeometryAlgorithms *geo=root->get<BezierTetrahedronSetGeometryAlgorithms>(root->SearchDown);
	
		typename MechanicalObject::SPtr dofs = root->get<MechanicalObject>(std::string("BezierTetrahedronTopology/"));
		typename MechanicalObject::WriteVecCoord coords = dofs->writePositions();
		MeshMatrixMass *mass=root->get<MeshMatrixMass>(root->SearchDown);
		const sofa::helper::vector<typename MeshMatrixMass::MassVector> & mv=mass->tetrahedronMassInfo.getValue();
		const sofa::helper::vector<typename MeshMatrixMass::MassType> &ma =mass->vertexMassInfo.getValue();

		size_t i,j,k,rank;	
		BezierDegreeType degree=container->getDegree();
		Real tetraVol1,tetraVol2,totalVol1,totalVol2;
		
		sofa::helper::vector<TetrahedronBezierIndex> tbiArray=container->getTetrahedronBezierIndexArray();
		size_t nbControlPoints=(degree+1)*(degree+2)*(degree+3)/6;
		totalVol1=0;
		for ( i = 0; i<container->getNumberOfTetrahedra(); i++)
		{
			
//				const BezierTetrahedronSetTopologyContainer::VecPointID &indexArray=container->getGlobalIndexArrayOfBezierPoints(i);
			
			
			/// get the volume of the tetrahedron
			tetraVol1=geo->computeTetrahedronVolume(i);
			tetraVol2=0;
			// compute the total volume
			totalVol1+=tetraVol1;
			/// check that the sum of the volume matrix elements is equal to the volume of the tetrahedron
			for (rank=0,j=0;j<nbControlPoints;j++) {
				for (k=j;k<nbControlPoints;k++,rank++) {
					if (k==j) 
						// add diagonal term
						tetraVol2+=mv[i][rank]; 
					else 
						// add 2 times off-diagonal term
						tetraVol2+=2*mv[i][rank]; 
				}
			}
			if (fabs(tetraVol1-tetraVol2)>1e-5) {
				ADD_FAILURE() << "Wrong mass matrix in tetrahedron no  : "<<i
				<< " Got total mass="<<tetraVol2<<" instead of "<<tetraVol1<<std::endl;
				return false;
			}
		}
		// compute totalVol2 as the total of the lumped volume
		totalVol2=0;
		for ( i = 0; i<ma.size(); i++)
		{
			totalVol2+=ma[i];
		}
		if (fabs(totalVol1-totalVol2)>1e-5) {
			ADD_FAILURE() << "Wrong total vertex mass value."
			 << " Got total vertex mass="<<totalVol2<<" instead of "<<totalVol1<<std::endl;
			return false;
		}
		return true;
	}
    void TearDown()
    {
        if (root!=NULL)
            sofa::simulation::getSimulation()->unload(root);
//        cerr<<"tearing down"<<endl;
    }

};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(BezierTetrahedronTopology_test, DataTypes);

// first test topology
TYPED_TEST( BezierTetrahedronTopology_test , testTopology )
{
//	this->loadScene( "tests/SofaTest/BezierTetrahedronTopology.scn");
	this->createScene();
	ASSERT_TRUE( this->testBezierTetrahedronTopology());
	ASSERT_TRUE( this->testBezierTetrahedronGeometry());
	ASSERT_TRUE( this->testBezierTetrahedronMass());
}



} // namespace sofa
