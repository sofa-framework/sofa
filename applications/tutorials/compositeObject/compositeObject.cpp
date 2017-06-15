/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <iostream>
#include <sstream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/UnitTest.h>
#include <sofa/helper/vector_algebra.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/PluginManager.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationTree/init.h>
#include <SofaSimulationTree/TreeSimulation.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaGeneralTopology/CubeTopology.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>

#include <utility>



using namespace sofa;
using namespace sofa::helper;
using helper::vector;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using namespace sofa::component::container;
using namespace sofa::component::topology;
using namespace sofa::component::collision;
using namespace sofa::component::visualmodel;
using namespace sofa::component::mapping;
using namespace sofa::component::forcefield;

typedef SReal Scalar;
typedef Vec<3,SReal> Vec3;
typedef Vec<1,SReal> Vec1;
typedef component::odesolver::EulerImplicitSolver EulerImplicitSolver;
typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;


bool startAnim = true;
bool verbose = false;
SReal complianceValue = 0.1;
Vec3 gravity(0,-1,0);
SReal dt = 0.01;

/// helper for more compact component creation
template<class Component>
typename Component::SPtr addNew( Node::SPtr parentNode, std::string name="" )
{
    typename Component::SPtr component = New<Component>();
    parentNode->addObject(component);
    component->setName(parentNode->getName()+"_"+name);
    return component;
}


/// Create an assembly of a siff hexahedral grid with other objects
simulation::Node::SPtr createGridScene(Vec3 startPoint, Vec3 endPoint, unsigned numX, unsigned numY, unsigned numZ, double totalMass/*, double stiffnessValue, double dampingRatio=0.0*/ )
{
    using helper::vector;

    // The graph root node
    Node::SPtr  root = simulation::getSimulation()->createNewGraph("root");
    root->setGravity( Coord3(0,-10,0) );
    root->setAnimate(false);
    root->setDt(0.01);
    addVisualStyle(root)->setShowVisual(false).setShowCollision(false).setShowMapping(true).setShowBehavior(true);

    Node::SPtr simulatedScene = root->createChild("simulatedScene");

    EulerImplicitSolver::SPtr eulerImplicitSolver = New<EulerImplicitSolver>();
    simulatedScene->addObject( eulerImplicitSolver );
    CGLinearSolver::SPtr cgLinearSolver = New<CGLinearSolver>();
    simulatedScene->addObject(cgLinearSolver);

    // The rigid object
    Node::SPtr rigidNode = simulatedScene->createChild("rigidNode");
    MechanicalObjectRigid3::SPtr rigid_dof = addNew<MechanicalObjectRigid3>(rigidNode, "dof");
    UniformMassRigid3::SPtr rigid_mass = addNew<UniformMassRigid3>(rigidNode,"mass");
    FixedConstraintRigid3::SPtr rigid_fixedConstraint = addNew<FixedConstraintRigid3>(rigidNode,"fixedConstraint");

    // Particles mapped to the rigid object
    Node::SPtr mappedParticles = rigidNode->createChild("mappedParticles");
    MechanicalObject3::SPtr mappedParticles_dof = addNew< MechanicalObject3>(mappedParticles,"dof");
    RigidMappingRigid3_to_3::SPtr mappedParticles_mapping = addNew<RigidMappingRigid3_to_3>(mappedParticles,"mapping");
    mappedParticles_mapping->setModels( rigid_dof.get(), mappedParticles_dof.get() );

    // The independent particles
    Node::SPtr independentParticles = simulatedScene->createChild("independentParticles");
    MechanicalObject3::SPtr independentParticles_dof = addNew< MechanicalObject3>(independentParticles,"dof");

    // The deformable grid, connected to its 2 parents using a MultiMapping
    Node::SPtr deformableGrid = independentParticles->createChild("deformableGrid"); // first parent
    mappedParticles->addChild(deformableGrid);                                       // second parent

    RegularGridTopology::SPtr deformableGrid_grid = addNew<RegularGridTopology>( deformableGrid, "grid" );
    deformableGrid_grid->setSize(numX,numY,numZ);
    deformableGrid_grid->setPos(startPoint[0],endPoint[0],startPoint[1],endPoint[1],startPoint[2],endPoint[2]);

    MechanicalObject3::SPtr deformableGrid_dof = addNew< MechanicalObject3>(deformableGrid,"dof");

    SubsetMultiMapping3_to_3::SPtr deformableGrid_mapping = addNew<SubsetMultiMapping3_to_3>(deformableGrid,"mapping");
    deformableGrid_mapping->addInputModel(independentParticles_dof.get()); // first parent
    deformableGrid_mapping->addInputModel(mappedParticles_dof.get());      // second parent
    deformableGrid_mapping->addOutputModel(deformableGrid_dof.get());

    UniformMass3::SPtr mass = addNew<UniformMass3>(deformableGrid,"mass" );
    mass->d_mass.setValue( totalMass/(numX*numY*numZ) );

    HexahedronFEMForceField3::SPtr hexaFem = addNew<HexahedronFEMForceField3>(deformableGrid, "hexaFEM");
    hexaFem->f_youngModulus.setValue(1000);
    hexaFem->f_poissonRatio.setValue(0.4);


    // ======  Set up the multimapping and its parents, based on its child
    deformableGrid_grid->init();  // initialize the grid, so that the particles are located in space
    deformableGrid_dof->init();   // create the state vectors
    MechanicalObject3::ReadVecCoord  xgrid = deformableGrid_dof->readPositions(); //    cerr<<"xgrid = " << xgrid << endl;


    // create the rigid frames and their bounding boxes
    unsigned numRigid = 2;
    vector<sofa::defaulttype::BoundingBox> boxes(numRigid);
    vector< vector<unsigned> > indices(numRigid); // indices of the particles in each box
    double eps = (endPoint[0]-startPoint[0])/(numX*2);

    // first box, x=xmin
    boxes[0] = sofa::defaulttype::BoundingBox(sofa::defaulttype::Vec3d(startPoint[0]-eps, startPoint[1]-eps, startPoint[2]-eps),
                           sofa::defaulttype::Vec3d(startPoint[0]+eps,   endPoint[1]+eps,   endPoint[2]+eps));

    // second box, x=xmax
    boxes[1] = sofa::defaulttype::BoundingBox(sofa::defaulttype::Vec3d(endPoint[0]-eps, startPoint[1]-eps, startPoint[2]-eps),
                           sofa::defaulttype::Vec3d(endPoint[0]+eps,   endPoint[1]+eps,   endPoint[2]+eps));
    rigid_dof->resize(numRigid);
    MechanicalObjectRigid3::WriteVecCoord xrigid = rigid_dof->writePositions();
    xrigid[0].getCenter()=sofa::defaulttype::Vec3d(startPoint[0], 0.5*(startPoint[1]+endPoint[1]), 0.5*(startPoint[2]+endPoint[2]));
    xrigid[1].getCenter()=sofa::defaulttype::Vec3d(  endPoint[0], 0.5*(startPoint[1]+endPoint[1]), 0.5*(startPoint[2]+endPoint[2]));

    // find the particles in each box
    vector<bool> isFree(xgrid.size(),true);
    unsigned numMapped = 0;
    for(unsigned i=0; i<xgrid.size(); i++){
        for(unsigned b=0; b<numRigid; b++ )
        {
            if( isFree[i] && boxes[b].contains(xgrid[i]) )
            {
                indices[b].push_back(i); // associate the particle with the box
                isFree[i] = false;
                numMapped++;
            }
        }
    }

    // distribution of the grid particles to the different parents (independent particle or solids.
    vector< std::pair<MechanicalObject3*,unsigned> > parentParticles(xgrid.size());

    // Copy the independent particles to their parent DOF
    independentParticles_dof->resize( numX*numY*numZ - numMapped );
    MechanicalObject3::WriteVecCoord xindependent = independentParticles_dof->writePositions(); // parent positions
    unsigned independentIndex=0;
    for( unsigned i=0; i<xgrid.size(); i++ ){
        if( isFree[i] ){
            parentParticles[i]=std::make_pair(independentParticles_dof.get(),independentIndex);
            xindependent[independentIndex] = xgrid[i];
            independentIndex++;
        }
    }

    // Mapped particles. The RigidMapping requires to cluster the particles based on their parent frame.
    mappedParticles_dof->resize(numMapped);
    MechanicalObject3::WriteVecCoord xmapped = mappedParticles_dof->writePositions(); // parent positions
    mappedParticles_mapping->globalToLocalCoords.setValue(true);                      // to define the mapped positions in world coordinates

    vector<unsigned>& rigidIndexPerPoint = *mappedParticles_mapping->rigidIndexPerPoint.beginEdit(); // to set to which rigid frame is attached each mapped particle
    rigidIndexPerPoint.clear();
    rigidIndexPerPoint.reserve( numMapped );
    unsigned mappedIndex=0;
    for( unsigned b=0; b<numRigid; b++ )
    {
        const vector<unsigned>& ind = indices[b];
        for(unsigned i=0; i<ind.size(); i++)
        {
            rigidIndexPerPoint.push_back( b );
            parentParticles[ind[i]]=std::make_pair(mappedParticles_dof.get(),mappedIndex);
            xmapped[mappedIndex] = xgrid[ ind[i] ];
            mappedIndex++;
        }
    }
    mappedParticles_mapping->rigidIndexPerPoint.endEdit();

    // Declare all the particles to the multimapping
    for( unsigned i=0; i<xgrid.size(); i++ )
    {
        deformableGrid_mapping->addPoint( parentParticles[i].first, parentParticles[i].second );
    }

    return root;
}

int main(int argc, char** argv)
{
    sofa::simulation::tree::init();
    sofa::helper::BackTrace::autodump();
    sofa::core::ExecParams::defaultInstance()->setAspectID(0);

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&startAnim,'a',"start","start the animation loop")
    .option(&verbose,'v',"verbose","print debug info")
    (argc,argv);

    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();
    sofa::gui::initMain();

    if (int err = sofa::gui::GUIManager::Init(argv[0],"")) return err;
    if (int err=sofa::gui::GUIManager::createGUI(NULL)) return err;
    sofa::gui::GUIManager::SetDimension(800,600);

    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());

    //=================================================
    sofa::simulation::Node::SPtr groot = createGridScene(Vec3(0,0,0), Vec3(5,1,1), 6,2,2, 1.0 );
    //=================================================

    sofa::simulation::getSimulation()->init(groot.get());
    sofa::gui::GUIManager::SetScene(groot);

#ifdef PS3
    groot->setAnimate(true);
#endif

    // Run the main loop
    if (int err = sofa::gui::GUIManager::MainLoop(groot))
        return err;

    sofa::simulation::getSimulation()->unload(groot);
    sofa::gui::GUIManager::closeGUI();

    sofa::simulation::tree::cleanup();
    return 0;
}



