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
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::Node ;
using sofa::core::objectmodel::BaseObject ;
using sofa::core::objectmodel::BaseData ;
using sofa::core::objectmodel::New ;
using sofa::core::execparams::defaultInstance; 

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/component/mechanicalload/PlaneForceField.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/odesolver/backward/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>
#include <sofa/component/mass/UniformMass.h>

#include <sofa/component/linearsolver/iterative/GraphScatteredTypes.h>

#include <map>

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace;

namespace sofa {

namespace {
using namespace component;
using namespace type;
using namespace defaulttype;
using namespace core::objectmodel;

using std::vector ;
using std::string ;
using std::map ;
using std::pair;

using sofa::component::linearsolver::GraphScatteredMatrix ;
using sofa::component::linearsolver::GraphScatteredVector ;
using sofa::component::linearsolver::iterative::CGLinearSolver ;

using sofa::simulation::graph::DAGSimulation ;

using sofa::component::mass::UniformMass ;
using sofa::component::mechanicalload::PlaneForceField ;

using sofa::component::statecontainer::MechanicalObject ;

using sofa::component::odesolver::backward::EulerImplicitSolver ;

template <typename TDataType, typename TMassType>
struct TypeTuple
{
    typedef TDataType DataType ;
    typedef TMassType MassType ;
} ;


template <typename TTypeTuple>
struct PlaneForceField_test : public BaseSimulationTest
{
    typedef typename TTypeTuple::DataType DataTypes ;
    typedef typename TTypeTuple::MassType MassType ;

    typedef typename DataTypes::VecCoord                                VecCoord;
    typedef typename DataTypes::Coord                                   Coord;
    typedef typename DataTypes::Deriv                                   Deriv;
    typedef typename DataTypes::CPos                                    CPos;
    typedef typename DataTypes::DPos                                    DPos;
    typedef typename Coord::value_type                                  Real;

    typedef CGLinearSolver<GraphScatteredMatrix, GraphScatteredVector>  CGLinearSolverType;
    typedef PlaneForceField<DataTypes>                                  PlaneForceFieldType;
    typedef MechanicalObject<DataTypes>                                 MechanicalObjectType;
    typedef EulerImplicitSolver                                         EulerImplicitSolverType;
    typedef UniformMass<DataTypes>                             TypedUniformMass;

    /// Root of the scene graph, created by the constructor and re-used in the tests
    simulation::Simulation*               m_simulation {nullptr};
    simulation::Node::SPtr                m_root;

    typename PlaneForceFieldType::SPtr      m_planeForceFieldSPtr;
    typename MechanicalObjectType::SPtr     m_mechanicalObj;

    /* Test if a mechanical object (point) subject only to the gravity is effectively stopped
     * by the plane force field.
     * In the special case where : stiffness = 500, damping = 5 and maxForce = 0 (default values)
    */
    void SetUp() override {}
    void TearDown() override {}

    void setupDefaultScene()
    {
        if(m_simulation==nullptr){
            BackTrace::autodump() ;
            m_simulation = sofa::simulation::getSimulation();
        }
        /// Create the scene
        m_root = m_simulation->createNewGraph("root");
        m_root->setGravity(Vec3d(-9.8, 0.0,0.0));

        const typename EulerImplicitSolverType::SPtr eulerImplicitSolver = New<EulerImplicitSolverType>();
        m_root->addObject(eulerImplicitSolver);

        const typename CGLinearSolverType::SPtr cgLinearSolver = New<CGLinearSolverType>();
        m_root->addObject(cgLinearSolver);
        cgLinearSolver->d_maxIter.setValue(25);
        cgLinearSolver->d_tolerance.setValue(1e-5);
        cgLinearSolver->d_smallDenominatorThreshold.setValue(1e-5);

        m_mechanicalObj = New<MechanicalObjectType>();
        m_root->addObject(m_mechanicalObj);

        //TODO(dmarchal): too much lines to just set a point... find a more concise way to do that
        Coord point;
        point[0]=1;
        VecCoord points;
        points.clear();
        points.push_back(point);

        m_mechanicalObj->x.setValue(points);

        typename TypedUniformMass::SPtr uniformMass = New<TypedUniformMass>();
        m_root->addObject(uniformMass);
        uniformMass->d_totalMass.setValue(1);

        /*Create the plane force field*/
        m_planeForceFieldSPtr = New<PlaneForceFieldType>();
        m_planeForceFieldSPtr->d_planeD.setValue(0);

        DPos normal;
        normal[0]=1;
        m_planeForceFieldSPtr->d_planeNormal.setValue(normal);

        m_root->addObject(m_planeForceFieldSPtr) ;
        sofa::simulation::node::initRoot(m_root.get());
    }

    void tearDownDefaultScene()
    {
        sofa::simulation::node::unload(m_root);
    }

    bool testBasicAttributes()
    {
        /*Create the plane force field*/
        m_planeForceFieldSPtr = New<PlaneForceFieldType>();
        m_planeForceFieldSPtr->d_planeD.setValue(0);

        DPos normal;
        normal[0]=1;
        m_planeForceFieldSPtr->d_planeNormal.setValue(normal);

        /// List of the supported attributes the user expect to find
        /// This list needs to be updated if you add an attribute.
        vector<string> attrnames = {
            "normal", "d", "stiffness", "damping", "maxForce", "bilateral", "localRange",
            "showPlane", "planeColor", "showPlaneSize"
        };

        for(auto& attrname : attrnames)
            EXPECT_NE( m_planeForceFieldSPtr->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

        return true;
    }

    bool testDefaultPlane()
    {
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "  <Node name='Level 1'>                                                        \n"
                 "   <MechanicalObject name='mstate' template='"<<  DataTypes::Name() << "'/>    \n"
                 "   <PlaneForceField name='myPlaneForceField'/>                                 \n"
                 "  </Node>                                                                      \n"
                 "</Node>                                                                        \n" ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
        EXPECT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        BaseObject* planeff = root->getTreeNode("Level 1")->getObject("myPlaneForceField") ;
        EXPECT_NE(planeff, nullptr) ;

        return true;
    }

    /// This kind of test is important as it enforce the developper to take a wider range of
    /// input values and check that they are gracefully handled.
    bool testMonkeyValuesForAttributes()
    {

        map<string, vector< pair<string, string> >> values ={
             {"damping",   {{"","5"}, {"-1.0","5"}, {"0.0","0"}, {"1.0", "1"}}},
             {"stiffness", {{"", "500"}, {"-1.0", "500"}, {"0.0", "0"}, {"1.0", "1"}}},
             {"maxForce",  {{"", "0"}, {"-1.0","0"}, {"0.5","0.5"}, {"1.5","1.5"}}},
             {"bilateral", {{"", "0"}, {"0","0"}, {"1","1"}, {"2","1"}, {"-1","1"}}},
             {"localRange", {{"","-1 -1"}, {"-2 -1", "-1 -1"}, {"-2 1", "-1 -1"}, {"0 0","0 0"}, {"1 -5","-1 -1"},
                             {"4 7","4 7"}, {"7 4","-1 -1"} }}
        };

        for(auto& kv : values){
          for(auto& v : kv.second){
            std::stringstream scene ;
            scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "  <Node name='Level 1'>                                                        \n"
                 "   <MechanicalObject name='mstate' template='"<<  DataTypes::Name() << "' position='1 2 3 4 5 6 7 8 9'/>    \n"
                 "   <PlaneForceField name='myPlaneForceField' "<< kv.first << "='"<< v.first << "' />\n"
                 "  </Node>                                                                      \n"
                 "</Node>                                                                        \n" ;

            Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
            EXPECT_NE(root.get(), nullptr) ;
            root->init(sofa::core::execparams::defaultInstance()) ;

            BaseObject* planeff = root->getTreeNode("Level 1")->getObject("myPlaneForceField") ;
            EXPECT_NE(planeff, nullptr) ;

            EXPECT_STREQ( planeff->findData(kv.first)->getValueString().c_str(), v.second.c_str() )
                  << "When the attribute '"<<kv.first<< "' is set to the value '" << v.first.c_str()
                  << "' it should be corrected during the component init to the valid value '" << v.second.c_str() << "'."
                  << " If this is not the case this means that the init function is not working properly (or the default "
                  << "value have changed and thus the test need to be fixed)";
            }
        }
        return true;
    }


    ///
    /// In this test we are verifying that a plane that have been reinited has the same
    /// value as one that have been inited.
    ///
    bool testInitReinitBehavior()
    {
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "  <Node name='Level 1'>                                                        \n"
                 "   <MechanicalObject name='mstate' template='"<<  DataTypes::Name() << "'/>    \n"
                 "   <PlaneForceField name='myPlaneForceField'/>                                 \n"
                 "  </Node>                                                                      \n"
                 "</Node>                                                                        \n" ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
        EXPECT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        BaseObject* planeff = root->getTreeNode("Level 1")->getObject("myPlaneForceField") ;
        EXPECT_NE(planeff, nullptr) ;

        {
            EXPECT_MSG_EMIT(Warning) ;
            planeff->init() ;
        }

        return true;
    }

    ///
    /// In this test we are verifying that a plane without a MechanicalObject is handled
    /// gracefully
    ///
    bool testBehaviorWhenMissingMechanicalObject()
    {
        EXPECT_MSG_EMIT(Error) ;

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "  <Node name='Level 1'>                                                        \n"
                 "   <PlaneForceField name='myPlaneForceField'/>                                 \n"
                 "  </Node>                                                                      \n"
                 "</Node>                                                                        \n" ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());

        EXPECT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        return true;
    }


    bool testPlaneForceField()
    {
        for(int i=0; i<100; i++)
        {
            sofa::simulation::node::animate(m_root.get(), 0.01_sreal);
        }
        Real x = m_mechanicalObj->x.getValue()[0][0];

        /// The point passed through the plane but is still too low.
        /// The value depend on the repulsion force generated and the mass of the point.
        if(x<-0.1)
        {
            ADD_FAILURE() << "Error while testing planeForceField. The mechnical point passed across the plane force field.";
            return false;
        }
        else
            return true;
    }

};

// Define the list of DataTypes to instanciate
using ::testing::Types;
typedef Types<
              TypeTuple<Rigid3Types, Rigid3Mass>
              ,TypeTuple<Vec1Types, SReal>
              ,TypeTuple<Vec2Types, SReal>
              ,TypeTuple<Vec3Types, SReal>
              ,TypeTuple<Vec6Types, SReal>
> DataTypes;

// Test suite for all the instanciations
TYPED_TEST_SUITE(PlaneForceField_test, DataTypes);// first test case
TYPED_TEST( PlaneForceField_test , testPlaneForceField )
{
    this->setupDefaultScene();
    ASSERT_TRUE (this->testPlaneForceField());
    this->tearDownDefaultScene();
}

TYPED_TEST( PlaneForceField_test , testBasicAttributes )
{
    ASSERT_TRUE (this->testBasicAttributes());
}

TYPED_TEST( PlaneForceField_test , testMonkeyValuesForAttributes )
{
    ASSERT_TRUE (this->testMonkeyValuesForAttributes());
}


TYPED_TEST( PlaneForceField_test , testInitReinitBehavior )
{
    ASSERT_TRUE (this->testInitReinitBehavior());
}


TYPED_TEST( PlaneForceField_test , testBehaviorWhenMissingMechanicalObject )
{
    ASSERT_TRUE (this->testBehaviorWhenMissingMechanicalObject());
}

TYPED_TEST( PlaneForceField_test , testDefaultPlane )
{
    ASSERT_TRUE (this->testDefaultPlane());
}


}
}// namespace sofa







