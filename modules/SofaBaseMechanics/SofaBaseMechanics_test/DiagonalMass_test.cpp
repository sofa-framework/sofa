/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define MY_PARAM_TYPE(name, DType, MType) \
    struct name { \
        typedef DType DataType; \
        typedef MType MassType; \
    }; \

#include <gtest/gtest.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>
#include <SceneCreator/SceneCreator.h>

#include <SofaBaseMechanics/DiagonalMass.h>
#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>
#include <SofaBaseMechanics/MechanicalObject.h>

//TODO : Perform smart tests :) Infrastructure for multi templated tests is ok.

namespace sofa {

template <class T>
class DiagonalMass_test : public ::testing::Test
{
public :

    typedef typename T::DataType dt;
    typedef typename T::MassType mt;
    typedef typename dt::Coord       Coord;
    typedef typename dt::VecCoord    VecCoord;
    typedef sofa::component::mass::DiagonalMass<dt,mt> DiagonalMassType;
    typename DiagonalMassType::SPtr m;

    /// Root of the scene graph
    sofa::simulation::Node::SPtr root;
    /// Simulation
    sofa::simulation::Simulation* simulation;

    /**
     * Constructor call for each test
     */
    virtual void SetUp()
    {
        // Init simulation
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        root = simulation::getSimulation()->createNewGraph("root");
    }

    void createSceneHexa()
    {
        simulation::Node::SPtr oneHexa = root->createChild("oneHexa");

        sofa::component::topology::HexahedronSetTopologyContainer::SPtr container = sofa::modeling::addNew<sofa::component::topology::HexahedronSetTopologyContainer>(oneHexa, "container");
        container->addHexa(0,1,2,3,4,5,6,7);
        typename sofa::component::topology::HexahedronSetGeometryAlgorithms<dt>::SPtr algo = sofa::modeling::addNew<sofa::component::topology::HexahedronSetGeometryAlgorithms<dt> >(oneHexa);
        typename sofa::component::container::MechanicalObject<dt>::SPtr meca= sofa::modeling::addNew<sofa::component::container::MechanicalObject<dt> >(oneHexa);
        VecCoord pos;
        pos.push_back(Coord(0.0, 0.0, 0.0));
        pos.push_back(Coord(1.0, 0.0, 0.0));
        pos.push_back(Coord(1.0, 1.0, 0.0));
        pos.push_back(Coord(0.0, 1.0, 0.0));
        pos.push_back(Coord(0.0, 0.0, 1.0));
        pos.push_back(Coord(1.0, 0.0, 1.0));
        pos.push_back(Coord(1.0, 1.0, 1.0));
        pos.push_back(Coord(0.0, 1.0, 1.0));
        meca->x = pos;

        typename sofa::component::mass::DiagonalMass<dt,mt>::SPtr mass = sofa::modeling::addNew<sofa::component::mass::DiagonalMass<dt,mt> >(oneHexa);
    }


    bool testHexaMass()
    {
        //initialize the simulation
        sofa::simulation::getSimulation()->init(root.get());

        //get the node called "oneHexa"
        simulation::Node *node = root->getChild("oneHexa");

        //the node is supposed to be found
        if (!node)
        {
            ADD_FAILURE() << "Cannot find first node" << std::endl;
            return false;
        }

        //get the mass
        sofa::component::mass::DiagonalMass<dt,mt> *mass = node->get<sofa::component::mass::DiagonalMass<dt,mt> >(node->SearchDown);
        if (!mass)
        {
            ADD_FAILURE() << "Cannot find mass" << std::endl;
            return false;
        }

        //get the mechanical object
        typename sofa::component::container::MechanicalObject<dt> *meca = node->get<sofa::component::container::MechanicalObject<dt> >(node->SearchDown);
        if (!meca)
        {
            ADD_FAILURE() << "Cannot find mechanical object" << std::endl;
            return false;
        }

        //test the number of mass points
        if (meca->x.getValue().size() != mass->f_mass.getValue().size())
        {
            ADD_FAILURE() << "Mass vector has not the same size as the number of points (" << mass->f_mass.getValue().size() << "!="<< meca->x.getValue().size() << ")" << std::endl;
            return false;
        }

        //check if the total mass is correct
        if ( fabs(1.0 - mass->m_totalMass.getValue()) > 1e-6)
        {
            ADD_FAILURE() << "Not the expected total mass: " << mass->m_totalMass << " and should be 1" << std::endl;
            return false;
        }

        return true;
    }

    void TearDown()
    {
        if (root!=NULL)
            sofa::simulation::getSimulation()->unload(root);
    }
};


TYPED_TEST_CASE_P(DiagonalMass_test);


TYPED_TEST_P(DiagonalMass_test, testHexahedra)
{
    this->createSceneHexa();
    ASSERT_TRUE(this->testHexaMass());
}

REGISTER_TYPED_TEST_CASE_P(DiagonalMass_test, testHexahedra);

#ifndef SOFA_FLOAT

MY_PARAM_TYPE(Vec3dd, sofa::defaulttype::Vec3dTypes, double)
MY_PARAM_TYPE(Vec2dd, sofa::defaulttype::Vec2dTypes, double)
MY_PARAM_TYPE(Vec1dd, sofa::defaulttype::Vec1dTypes, double)

INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_hexa_test_case3d, DiagonalMass_test, Vec3dd);

//INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_test_case1d, DiagonalMass_test, Vec3dd);
//INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_test_case2d, DiagonalMass_test, Vec2dd);
//INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_test_case3d, DiagonalMass_test, Vec1dd);

#endif

#ifndef SOFA_DOUBLE

MY_PARAM_TYPE(Vec3ff, sofa::defaulttype::Vec3fTypes, float)
MY_PARAM_TYPE(Vec2ff, sofa::defaulttype::Vec2fTypes, float)
MY_PARAM_TYPE(Vec1ff, sofa::defaulttype::Vec1fTypes, float)

INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_hexa_test_case3f, DiagonalMass_test, Vec3ff);

//INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_test_case1f, DiagonalMass_test, Vec3ff);
//INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_test_case2f, DiagonalMass_test, Vec2ff);
//INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_test_case3f, DiagonalMass_test, Vec1ff);

#endif

} // namespace sofa
