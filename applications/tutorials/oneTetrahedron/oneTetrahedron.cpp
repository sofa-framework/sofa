/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/VecId.h>
using sofa::core::VecId;
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
using sofa::defaulttype::Vec3Types;
using Coord3 = sofa::type::Vec3;
using VecCoord3 = sofa::type::vector<Coord3>;
#include <sofa/gui/common/GUIManager.h>
#include <sofa/gui/init.h>
#include <sofa/gui/common/ArgumentParser.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/Node.h>

#include <sofa/component/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/graph/init.h>

#include <sofa/component/io/mesh/MeshOBJLoader.h>
#include <sofa/simulation/DefaultAnimationLoop.h>
#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>
using CGLinearSolver = sofa::component::linearsolver::iterative::CGLinearSolver<sofa::component::linearsolver::GraphScatteredMatrix, sofa::component::linearsolver::GraphScatteredVector>;
#include <sofa/component/mapping/linear/BarycentricMapping.h>
using BarycentricMapping3 = sofa::component::mapping::linear::BarycentricMapping<Vec3Types, Vec3Types>;
#include <sofa/component/statecontainer/MechanicalObject.h>
using MechanicalObject3 = sofa::component::statecontainer::MechanicalObject<Vec3Types>;
#include <sofa/component/mass/UniformMass.h>
using UniformMass3 = sofa::component::mass::UniformMass<Vec3Types>;
#include <sofa/component/topology/container/constant/MeshTopology.h>
using sofa::component::topology::container::constant::MeshTopology;
#include <sofa/component/visual/VisualStyle.h>
using sofa::component::visual::VisualStyle;
#include <sofa/component/constraint/projective/FixedConstraint.h>
using FixedConstraint3 = sofa::component::constraint::projective::FixedConstraint<Vec3Types>;
#include <sofa/component/odesolver/backward/EulerImplicitSolver.h>
using sofa::component::odesolver::backward::EulerImplicitSolver;
#include <sofa/gl/component/rendering3d/OglModel.h>
using sofa::gl::component::rendering3d::OglModel;
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>
using TetrahedronFEMForceField3 = sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField<Vec3Types>;

using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::New;
using sofa::helper::ReadAccessor;
using sofa::helper::WriteAccessor;
using sofa::simulation::Node;


// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    sofa::gui::common::ArgumentParser argParser(argc, argv);
    sofa::gui::common::GUIManager::RegisterParameters(&argParser);
    argParser.parse();

    //force load all components
    sofa::component::init();
    sofa::simulation::graph::init();
    //force load SofaGui (registering guis)
    sofa::gui::init();

    sofa::gui::common::GUIManager::Init(argv[0]);

    const auto groot = sofa::simulation::getSimulation()->createNewGraph("root");
    groot->setGravity( sofa::type::Vec3(0,-10,0) );

    groot->addObject(sofa::core::objectmodel::New<sofa::simulation::DefaultAnimationLoop>());

    // One solver for all the graph
    const EulerImplicitSolver::SPtr solver = sofa::core::objectmodel::New<EulerImplicitSolver>();
    solver->setName("solver");
    solver->f_printLog.setValue(false);
    groot->addObject(solver);

    const CGLinearSolver::SPtr linearSolver = New<CGLinearSolver>();
    linearSolver->setName("linearSolver");
    groot->addObject(linearSolver);
    linearSolver->d_maxIter.setValue(25);
    linearSolver->d_tolerance.setValue(1e-5);
    linearSolver->d_smallDenominatorThreshold.setValue(1e-5);


    // Tetrahedron degrees of freedom
    const MechanicalObject3::SPtr DOF = sofa::core::objectmodel::New<MechanicalObject3>();
    groot->addObject(DOF);
    DOF->resize(4);
    DOF->setName("DOF");
    //get write access to the position vector of mechanical object DOF
    WriteAccessor<Data<VecCoord3> > x = *DOF->write(VecId::position());
    x[0] = Coord3(0,10,0);
    x[1] = Coord3(10,0,0);
    x[2] = Coord3(-10*0.5,0,10*0.866);
    x[3] = Coord3(-10*0.5,0,-10*0.866);
    DOF->showObject.setValue(true);
    DOF->showObjectScale.setValue(10.);


    // Tetrahedron uniform mass
    const UniformMass3::SPtr mass = sofa::core::objectmodel::New<UniformMass3>();
    groot->addObject(mass);
    mass->setMass(2);
    mass->setName("mass");

    // Tetrahedron topology
    const MeshTopology::SPtr topology = sofa::core::objectmodel::New<MeshTopology>();
    topology->setName("mesh topology");
    groot->addObject( topology );
    topology->addTetra(0,1,2,3);

    // Tetrahedron constraints
    const FixedConstraint3::SPtr constraints = sofa::core::objectmodel::New<FixedConstraint3>();
    constraints->setName("constraints");
    groot->addObject(constraints);
    constraints->addConstraint(0);

    // Tetrahedron force field
    const TetrahedronFEMForceField3::SPtr fem = sofa::core::objectmodel::New<TetrahedronFEMForceField3>();
    fem->setName("FEM");
    groot->addObject(fem);
    fem->setMethod("polar");
    fem->setUpdateStiffnessMatrix(true);
    fem->setYoungModulus(6);
    fem->setPoissonRatio(0.45);

    // Tetrahedron skin
    const Node::SPtr skin = groot.get()->createChild("skin");

    // Load the visual mesh
    const auto meshLoader = sofa::core::objectmodel::New<sofa::component::io::mesh::MeshOBJLoader>();
    meshLoader->setName( "meshLoader" );
    meshLoader->setFilename(sofa::helper::system::DataRepository.getFile("mesh/liver-smooth.obj"));
    meshLoader->setScale(0.7, 0.7, 0.7);
    meshLoader->setTranslation(1.2, 0.8, 0);
    skin->addObject(meshLoader);

    // The visual model
    const OglModel::SPtr visual = sofa::core::objectmodel::New<OglModel>();
    visual->setName( "visual" );
    visual->setColor("red");
    visual->setSrc("@meshLoader", meshLoader.get());
    skin->addObject(visual);

    // The mapping between the tetrahedron (DOF) and the liver (visual)
    const BarycentricMapping3::SPtr mapping = sofa::core::objectmodel::New<BarycentricMapping3>();
    mapping->setModels(DOF.get(), visual.get());
    mapping->setName( "mapping" );
    skin->addObject(mapping);

    // Display Flags
    const VisualStyle::SPtr style = sofa::core::objectmodel::New<VisualStyle>();
    groot->addObject(style);
    sofa::core::visual::DisplayFlags& flags = *style->displayFlags.beginEdit();
    flags.setShowNormals(false);
    flags.setShowInteractionForceFields(false);
    flags.setShowMechanicalMappings(false);
    flags.setShowCollisionModels(false);
    flags.setShowBoundingCollisionModels(false);
    flags.setShowMappings(false);
    flags.setShowForceFields(true);
    flags.setShowWireFrame(true);
    flags.setShowVisualModels(true);
    flags.setShowBehaviorModels(true);
    style->displayFlags.endEdit();

    sofa::gui::common::GUIManager::SetScene(groot);

    // Init the scene
    sofa::simulation::node::initRoot(groot.get());
    groot->setAnimate(false);

    //=======================================
    // Run the main loop
    sofa::gui::common::GUIManager::MainLoop(groot);
    sofa::simulation::graph::cleanup();
    return 0;
}
