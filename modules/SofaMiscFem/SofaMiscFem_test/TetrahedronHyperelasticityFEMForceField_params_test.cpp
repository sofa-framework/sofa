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
struct TetrahedronHyperelasticityFEMForceField_params_test : public Sofa_test<typename _ForceFieldType::DataTypes::Real>
{
    typedef _ForceFieldType ForceField;
    typedef typename ForceField::DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef sofa::component::container::MechanicalObject<DataTypes> DOF;
    typedef typename sofa::component::forcefield::TetrahedronHyperelasticityFEMForceField<DataTypes> TetrahedronHyperelasticityFEMForceField;

    /// @name Scene elements
    /// {
    typename DOF::SPtr dof;
    simulation::Node::SPtr root, hyperelasticNode;
    std::string sceneFilename;
    /// }

    /// @name Precision and control parameters
    /// {
    SReal timeStep;
    bool debug;
    /// }

    /// @name Tested API
    /// {
    static const unsigned char TEST_ALL = UCHAR_MAX; ///< testing everything
    unsigned char flags; ///< testing options. (all by default). To be used with precaution.
    /// }


    TetrahedronHyperelasticityFEMForceField_params_test()
        : sceneFilename(std::string(SOFAMISCFEM_TEST_SCENES_DIR) + "/" + "TetrahedronHyperelasticityFEMForceField_base.scn")
    {
        timeStep = 0.02;

        simulation::Simulation* simu;
        sofa::simulation::setSimulation(simu = new sofa::simulation::graph::DAGSimulation());

        /// Load the scene
        root = simu->createNewGraph("root");
    }

    void scene_load()
    {
        root = sofa::simulation::getSimulation()->load(sceneFilename.c_str());

        hyperelasticNode = root->getChild("Hyperelastic-Liver");

        if(!hyperelasticNode)
        {
            msg_error("TetrahedronHyperelasticityFEMForceField_params_test") << "Node hyperelasticNode not found in TetrahedronHyperelasticityFEMForceField_params_test.scn, test will break" ;
            return;
        }
    }

    void run_test_params_mooney_case()
    {
        this->scene_load();

        // Add a Mooney-Rivlin constitutive law
        typename sofa::component::forcefield::TetrahedronHyperelasticityFEMForceField<DataTypes>::SPtr FF = sofa::core::objectmodel::New< sofa::component::forcefield::TetrahedronHyperelasticityFEMForceField<DataTypes> >();
        sofa::helper::vector<Real> param_vector;
        param_vector.resize(3);
        // Experimental data gave a deflexion of y=-0.11625 with the following parameters (C01 = 151065.460 ; C10 = 101709.668 1e07 ; D0 = 1e07)
        param_vector[0] = 151065.460;   // Monney parameter C01
        param_vector[1] = 101709.668;   // Monney parameter C10
        param_vector[2] = 1e07;         // Monney parameter K0

        hyperelasticNode->addObject(FF);
        FF->setName("FEM");
        FF->setMaterialName("MooneyRivlin");
        FF->setparameter(param_vector);

        // Init simulation
        sofa::simulation::getSimulation()->init(this->root.get());

        //Check component creation
        sofa::core::objectmodel::BaseObject* hefem = root->getTreeNode("Hyperelastic-Liver")->getObject("FEM") ;
        EXPECT_NE(hefem, nullptr) ;
    }
};


// ========= Define the list of types to instanciate.
//using testing::Types;
typedef testing::Types<component::forcefield::TetrahedronHyperelasticityFEMForceField<defaulttype::Vec3Types> > TestTypes; // the types to instanciate.


// ========= Tests to run for each instanciated type
TYPED_TEST_CASE(TetrahedronHyperelasticityFEMForceField_params_test, TestTypes);

// test case
TYPED_TEST( TetrahedronHyperelasticityFEMForceField_params_test , extension )
{
    EXPECT_MSG_NOEMIT(Error) ;
    this->debug = false;

    // run test : Mooney
    this->run_test_params_mooney_case();
}


} // namespace sofa


#endif /* SOFA_STANDARDTEST_TETRAHEDRONHYPERELASTICITYFEMFORCEFIELD__TEST_CPP */
