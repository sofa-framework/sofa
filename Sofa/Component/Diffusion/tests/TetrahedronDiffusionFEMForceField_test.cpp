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
using sofa::testing::BaseSimulationTest;

#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/topology/container/grid/RegularGridTopology.h>
#include <sofa/component/diffusion/TetrahedronDiffusionFEMForceField.h>
#include <sofa/component/mass/DiagonalMass.h>

#include <sofa/type/Vec.h>

#include <iostream>
#include <fstream>
#include <climits>
#include <gtest/gtest.h>

namespace sofa {

/** @brief Comparison of the result of simulation with theoretical values in Diffusion case
 *
 * @author Talbot Hugo, 2016
 *
 */
template <typename _ForceFieldType>
struct TetrahedronDiffusionFEMForceField_test : public BaseSimulationTest
{
    typedef _ForceFieldType ForceField;
    typedef typename ForceField::DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef component::statecontainer::MechanicalObject<DataTypes> DOF;
    typedef typename component::topology::container::grid::RegularGridTopology RegularGridTopology;
    typedef typename component::diffusion::TetrahedronDiffusionFEMForceField<DataTypes> TetrahedronDiffusionFEMForceField;
    typedef typename component::mass::DiagonalMass<DataTypes> DiagonalMass;

    /// @name Scene elements
    /// {
    typename DOF::SPtr dof;
    simulation::Node::SPtr root, tetraNode, temperatureNode;
    std::string sceneFilename;
    Coord theorX;
    /// }

    /// @name Precision and control parameters
    /// {
    Real diffusionCoefficient;
    SReal massDensity;
    type::Vec<3,SReal> beamDimension;
    type::Vec3i beamResolution;
    SReal errorMax;
    SReal timeEvaluation;
    SReal timeStep;
    unsigned long idMiddlePoint;
    bool debug;
    /// }

    /// @name Tested API
    /// {
    static const unsigned char TEST_ALL = UCHAR_MAX; ///< testing everything
    unsigned char flags; ///< testing options. (all by default). To be used with precaution.
    /// }


    TetrahedronDiffusionFEMForceField_test()
        : sceneFilename(std::string(SOFA_COMPONENT_DIFFUSION_TEST_SCENES_DIR) + "/" + "TetrahedronDiffusionFEMForceField.scn")
        , diffusionCoefficient( 1.0 )
        , massDensity( 1.0 )
        , beamDimension(1.0, 0.5, 0.5)
        , beamResolution( type::Vec3i(21,11,11) )
        , errorMax( 1e-3 )
        , timeEvaluation( 0.05 )
    {
        timeEvaluation = 0.05;
        timeStep = 0.0001;
        idMiddlePoint = 1270;

        simulation::Simulation* simu = sofa::simulation::getSimulation();
        assert(simu);

        /// Load the scene
        root = simu->createNewGraph("root");
        root = sofa::simulation::node::load(sceneFilename.c_str());

    }


    void init_scene()
    {        
        tetraNode = root->getChild("Tetra");
        temperatureNode = tetraNode->getChild("Temperature");


        if(!tetraNode || !temperatureNode)
        {
          msg_error("TetrahedronDiffusionFEMForceField_test") << "Node not found in TetrahedronDiffusionFEMForceField_test.scn, test will break" ;
          return;
        }

        // Beam setup
        const typename RegularGridTopology::SPtr grid = root->get<RegularGridTopology>(root->SearchDown);
        if(grid)
        {
            grid->setSize(beamResolution);
            grid->setPos(0.0, beamDimension[0], 0.0, beamDimension[1], 0.0, beamDimension[2]);
        }

        // Init simulation
        sofa::simulation::node::initRoot(this->root.get());

        // Mass parameters
        typename DiagonalMass::SPtr mass = temperatureNode->get<DiagonalMass>(temperatureNode->SearchDown);
        if(mass)
        {
            mass->setMassDensity(massDensity);
        }

        ///  Get mechanical object of tracked points
        dof = temperatureNode->get<DOF>(temperatureNode->SearchDown);

        // Diffusion parameters
        typename TetrahedronDiffusionFEMForceField::SPtr diffusionFF = temperatureNode->get<TetrahedronDiffusionFEMForceField>(temperatureNode->SearchDown);
        if(diffusionFF)
        {
            diffusionFF->setDiffusionCoefficient(diffusionCoefficient);
        }
    }

    void animate_scene()
    {
        //Animate simulation
        const unsigned int nbSteps = timeEvaluation/timeStep;
        unsigned int stepId;
        for (stepId = 0; stepId < nbSteps; ++stepId)
        {
            sofa::simulation::node::animate(root.get(), timeStep);
        }
    }


    void compute_theory()
    {
        // For a Dirac heat of T=1 and a fixed BC T=0, the temperature at time = TTTT in the middle of the beam is:
        const SReal temp = 1.0 / (4.0 * sqrt(timeEvaluation));
        theorX[0] = std::erfc(temp);
    }


    void run_test_theoretical_diffusion()
    {
        typename DOF::ReadVecCoord sofaX= this->dof->readPositions();
        SReal value_SOFA = 0.0;
        if(sofaX.size() > idMiddlePoint)
            value_SOFA = sofaX[idMiddlePoint][0];

        if( fabs(value_SOFA - theorX[0]) > errorMax)
            ADD_FAILURE()<<"Wrong computation for diffusion effect" << std::endl;
    }
};


// ========= Define the list of types to instantiate.
//using ::testing::Types;
typedef ::testing::Types<component::diffusion::TetrahedronDiffusionFEMForceField<defaulttype::Vec1Types> > TestTypes; // the types to instantiate.


// ========= Tests to run for each instantiated type
TYPED_TEST_SUITE(TetrahedronDiffusionFEMForceField_test, TestTypes);

// test case
TYPED_TEST( TetrahedronDiffusionFEMForceField_test , extension )
{
    EXPECT_MSG_NOEMIT(Error) ;
    this->debug = false;


    // run test
    this->init_scene();
    this->animate_scene();
    this->compute_theory();

    this->run_test_theoretical_diffusion();
}


} // namespace sofa

#undef ERFC
