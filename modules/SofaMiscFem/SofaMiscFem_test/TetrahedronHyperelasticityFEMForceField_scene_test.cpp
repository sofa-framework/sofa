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
#ifndef SOFA_STANDARDTEST_TETRAHEDRONHYPERELASTICITYFEMFORCEFIELD__TEST_CPP
#define SOFA_STANDARDTEST_TETRAHEDRONHYPERELASTICITYFEMFORCEFIELD__TEST_CPP

#include <SofaSimulationGraph/DAGSimulation.h>
#include <SofaTest/Sofa_test.h>
#include <SofaTest/TestMessageHandler.h>

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaMiscFem/TetrahedronHyperelasticityFEMForceField.h>

#include <sofa/defaulttype/Vec.h>

#include <iostream>
#include <fstream>



namespace sofa {

/** @brief Comparison of the result of simulation with theoretical values with hyperelastic material
 *
 * @author Talbot Hugo, 2017
 *
 */
template <typename _ForceFieldType>
struct TetrahedronHyperelasticityFEMForceField_scene_test : public Sofa_test<typename _ForceFieldType::DataTypes::Real>
{
    typedef _ForceFieldType ForceField;
    typedef typename ForceField::DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef component::container::MechanicalObject<DataTypes> DOF;
    typedef typename component::forcefield::TetrahedronHyperelasticityFEMForceField<DataTypes> TetrahedronHyperelasticityFEMForceField;

    /// @name Scene elements
    /// {
    typename DOF::SPtr dof;
    simulation::Node::SPtr root, mooneyNode;
    std::string sceneFilename;
    Coord refY;
    /// }

    /// @name Precision and control parameters
    /// {
    SReal epsilon, timeEvaluation, timeStep, tolerance;
    unsigned long tipPoint;
    bool debug;
    /// }

    /// @name Tested API
    /// {
    static const unsigned char TEST_ALL = UCHAR_MAX; ///< testing everything
    unsigned char flags; ///< testing options. (all by default). To be used with precaution.
    /// }


    TetrahedronHyperelasticityFEMForceField_scene_test()
        : sceneFilename(std::string(SOFAMISCFEM_TEST_SCENES_DIR) + "/" + "TetrahedronHyperelasticityFEMForceField_test.scn")
          , epsilon( 1e-5 )

        //Mooney-Rivlin
        //Tip position practice : -0.000373234, -0.106832, 0.141393

    {
        timeEvaluation = 3.60;
        timeStep = 0.02;
        tipPoint = 2677;
        // Experimental data gave a deflexion of y=-0.11625 with the following parameters (C01 = 151065.460 ; C10 = 101709.668 1e07 ; D0 = 1e07)
        // Simulation gives a slightly different reference:
        refY[0] = -0.106832;

        simulation::Simulation* simu;
        sofa::simulation::setSimulation(simu = new sofa::simulation::graph::DAGSimulation());

        /// Load the scene
        root = simu->createNewGraph("root");
        root = sofa::simulation::getSimulation()->load(sceneFilename.c_str());

    }


    void init_scene()
    {
        mooneyNode = root->getChild("MooneyRivlin-Model");

        if(!mooneyNode)
        {
            msg_error("TetrahedronHyperelasticityFEMForceField_scene_test") << "Node mooneyNode not found in TetrahedronHyperelasticityFEMForceField_test.scn, test will break" ;
            return;
        }

        // Init simulation
        sofa::simulation::getSimulation()->init(this->root.get());

        ///  Get mechanical object of tracked points
        dof = mooneyNode->get<DOF>(mooneyNode->SearchDown);
    }


    void animate_scene()
    {
        //Animate simulation
        unsigned int nbSteps = timeEvaluation/timeStep;
        unsigned int stepId;
        for (stepId = 0; stepId < nbSteps; ++stepId)
            sofa::simulation::getSimulation()->animate(root.get(),timeStep);
    }


    void run_test_compare_reference()
    {
        typename DOF::ReadVecCoord sofaX= this->dof->readPositions();
        SReal value_SOFA = 0.0;
        if(sofaX.size() > tipPoint)
            value_SOFA = sofaX[tipPoint][1]; // should be close to -0.109578

        if( fabs(value_SOFA - refY[0]) > epsilon)
            ADD_FAILURE()<<"Wrong computation for hyperelastic material" << std::endl;
    }
};


// ========= Define the list of types to instanciate.
//using testing::Types;
typedef testing::Types<component::forcefield::TetrahedronHyperelasticityFEMForceField<defaulttype::Vec3Types> > TestTypes; // the types to instanciate.


// ========= Tests to run for each instanciated type
TYPED_TEST_CASE(TetrahedronHyperelasticityFEMForceField_scene_test, TestTypes);

// test case
TYPED_TEST( TetrahedronHyperelasticityFEMForceField_scene_test , extension )
{
    EXPECT_MSG_NOEMIT(Error) ;
    this->debug = false;

    // run test
    this->init_scene();
    this->animate_scene();
    this->run_test_compare_reference();
}


} // namespace sofa


#endif /* SOFA_STANDARDTEST_TETRAHEDRONHYPERELASTICITYFEMFORCEFIELD__TEST_CPP */
