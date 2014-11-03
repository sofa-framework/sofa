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


#include "Sofa_test.h"
#include <SofaComponentMain/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <SofaConstraint/BilateralInteractionConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/simulation/common/SceneLoaderXML.h>


namespace sofa {

namespace {


using std::cout;
using std::cerr;
using std::endl;
using namespace component;
using namespace defaulttype;


template <typename _DataTypes>
struct BilateralInteractionConstraint_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos CPos;
    typedef typename Coord::value_type Real;
    typedef constraintset::BilateralInteractionConstraint<DataTypes> BilateralInteractionConstraint;
    typedef component::topology::PointSetTopologyContainer PointSetTopologyContainer;
    typedef container::MechanicalObject<DataTypes> MechanicalObject;

    simulation::Node::SPtr root;                 ///< Root of the scene graph, created by the constructor an re-used in the tests
    simulation::Simulation* simulation;          ///< created by the constructor an re-used in the tests

    /// Create the context for the tests.
    void SetUp()
    {
        sofa::component::init();
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        /// Load the scene
        std::string sceneName = "BilateralInteractionConstraint.scn";
        std::string fileName  = std::string(SOFATEST_SCENES_DIR) + "/" + sceneName;
        root = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()));

        // Test if load has succeeded
        sofa::simulation::SceneLoaderXML scene;

        if(!root || !scene.loadSucceed)
        {
            ADD_FAILURE() << "Error while loading the scene: " << sceneName << std::endl;
        }
    }

    void init_Setup()
    {
        /// Init
        sofa::simulation::getSimulation()->init(root.get());
    }

    bool test_constrainedPositions()
    {
        std::vector<MechanicalObject*> meca;
        root->get<MechanicalObject>(&meca,std::string("mecaConstraint"),root->SearchDown);

        std::vector<Coord> points;
        points.resize(2);

        if(meca.size()==2)
        {
            for(int i=0; i<meca.size(); i++)
                points[i] = meca[i]->read(core::ConstVecCoordId::position())->getValue()[0];
        }
        else
        {
            ADD_FAILURE() << "Error while searching mechanical object" << std::endl;
        }

        for(int i=0; i<10; i++)
            sofa::simulation::getSimulation()->animate(root.get(),(double)0.001);

        if(points[0] == points[1]) return true;
        else
        {
            ADD_FAILURE() << "Error while testing if two positions are correctly constrained" << std::endl;
        }

        return false;
    }


 };


// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<Vec3Types> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(BilateralInteractionConstraint_test, DataTypes);
TYPED_TEST( BilateralInteractionConstraint_test , constrainedPositions )
{
    this->init_Setup();
    ASSERT_TRUE(  this->test_constrainedPositions() );
}

}

} // namespace sofa

