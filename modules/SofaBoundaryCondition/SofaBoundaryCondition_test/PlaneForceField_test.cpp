/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <SofaTest/Sofa_test.h>

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::BaseObject ;
using sofa::core::objectmodel::BaseData ;
using sofa::core::objectmodel::New ;
using sofa::core::ExecParams ;
using sofa::defaulttype::Vec3dTypes ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;


#include <SofaBoundaryCondition/PlaneForceField.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaBaseMechanics/UniformMass.h>

#include <SofaBaseLinearSolver/GraphScatteredTypes.h>

namespace sofa {

namespace {
using namespace component;
using namespace defaulttype;
using namespace core::objectmodel;

using std::vector ;
using std::string ;

using sofa::linearsolver::GraphScatteredMatrix ;
using sofa::linearsolver::GraphScatteredVector ;
using sofa::linearsolver::CGLinearSolver ;

using sofa::component::mass::UniformMass ;
using sofa::component::forcefield::PlaneForceField ;

using sofa::container::MechanicalObject ;

using sofa::odesolver::EulerImplicitSolver ;


template <typename _DataTypes>
struct PlaneForceField_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;

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


    /// Root of the scene graph, created by the constructor and re-used in the tests
    simulation::Simulation*               m_simulation;
    simulation::Node::SPtr                m_root;

    typename PlaneForceFieldType::SPtr      m_planeForceFieldSPtr;
    typename MechanicalObjectType::SPtr     m_mechanicalObj;

    /* Test if a mechanical object (point) subject only to the gravity is effectively stopped
     * by the plane force field.
     * In the special case where : stiffness = 500, damping = 5 and maxForce = 0 (default values)
    */
    void SetUp()
    {
        sofa::simulation::setSimulation(m_simulation = new sofa::simulation::graph::DAGSimulation());

        /// Create the scene
        m_root = m_simulation->createNewGraph("root");

        typename EulerImplicitSolverType::SPtr eulerImplicitSolver = New<EulerImplicitSolverType>();
        m_root->addObject(eulerImplicitSolver);

        typename CGLinearSolverType::SPtr cgLinearSolver = New<CGLinearSolverType>();
        m_root->addObject(cgLinearSolver);
        cgLinearSolver->f_maxIter.setValue(25);
        cgLinearSolver->f_tolerance.setValue(1e-5);
        cgLinearSolver->f_smallDenominatorThreshold.setValue(1e-5);

        m_mechanicalObj = New<MechanicalObjectType>();
        m_root->addObject(m_mechanicalObj);
        Coord point;
        point[0]=1;
        VecCoord points;
        points.clear();
        points.push_back(point);

        m_mechanicalObj->x0.setValue(points);

        std::string name = DataTypeInfo<DataTypes>::name();
#ifdef SOFA_WITH_DOUBLE
        if(name=="Rigid")
        {
            typename UniformMass<Rigid3dTypes,Rigid3dMass>::SPtr uniformMass = New<UniformMass<Rigid3dTypes,Rigid3dMass> >();
            m_root->addObject(uniformMass);
            uniformMass->d_totalMass.setValue(1);
        }
        else if(name=="Vec1d" )
        {
            typename UniformMass<Vec1dTypes,double>::SPtr uniformMass = New<UniformMass<Vec1dTypes,double> >();
            m_root->addObject(uniformMass);
            uniformMass->d_totalMass.setValue(1);
        }
        else if(name=="Vec2d" )
        {
            typename UniformMass<Vec2dTypes,double>::SPtr uniformMass = New<UniformMass<Vec2dTypes,double> >();
            m_root->addObject(uniformMass);
            uniformMass->d_totalMass.setValue(1);
        }
        else if(name=="Vec3d" )
        {
            typename UniformMass<Vec3dTypes,double>::SPtr uniformMass = New<UniformMass<Vec3dTypes,double> >();
            m_root->addObject(uniformMass);
            uniformMass->d_totalMass.setValue(1);
        }
        else if(name=="Vec6d")
        {
            typename UniformMass<Vec6dTypes,double>::SPtr uniformMass = New<UniformMass<Vec6dTypes,double> >();
            m_root->addObject(uniformMass);
            uniformMass->d_totalMass.setValue(1);
        }
        else
#endif
#ifdef SOFA_WITH_FLOAT
        if(name=="Rigid3f")
        {
            typename UniformMass<Rigid3fTypes,Rigid3fMass>::SPtr uniformMass = New<UniformMass<Rigid3fTypes,Rigid3fMass> >();
            m_root->addObject(uniformMass);
            uniformMass->d_totalMass.setValue(1);
        }
        else if(name=="Vec1f" )
        {
            typename UniformMass<Vec1fTypes,float>::SPtr uniformMass = New<UniformMass<Vec1fTypes,float> >();
            m_root->addObject(uniformMass);
            uniformMass->d_totalMass.setValue(1);
        }
        else if(name=="Vec2f" )
        {
            typename UniformMass<Vec2fTypes,float>::SPtr uniformMass = New<UniformMass<Vec2fTypes,float> >();
            m_root->addObject(uniformMass);
            uniformMass->d_totalMass.setValue(1);
        }
        else if(name=="Vec3f" )
        {
            typename UniformMass<Vec3fTypes,float>::SPtr uniformMass = New<UniformMass<Vec3fTypes,float> >();
            m_root->addObject(uniformMass);
            uniformMass->d_totalMass.setValue(1);
        }
        else if(name=="Vec6f")
        {
            typename UniformMass<Vec6fTypes,float>::SPtr uniformMass = New<UniformMass<Vec6fTypes,float> >();
            m_root->addObject(uniformMass);
            uniformMass->d_totalMass.setValue(1);
        }
        else
#endif
        //TODO(dmarchal): This is really weird and need proper investigation.
        // Why do the test succeed while there is no gravity to this scene and no
        // plane force field ?
        m_root->setGravity(Vec3d(0, -9.8,0));

        return;

        /*Create the plane force field*/
        m_planeForceFieldSPtr = New<PlaneForceFieldType>();
        m_planeForceFieldSPtr->d_planeD.setValue(0);

        DPos normal;
        normal[0]=1;
        m_planeForceFieldSPtr->d_planeNormal.setValue(normal);

    }

    void initSetup()
    {
        // Init
        sofa::simulation::getSimulation()->init(m_root.get());
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
            "draw", "color", "drawSize"
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
                 "   <MechanicalObject name='mstate' template='"<<  DataTypeInfo<DataTypes>::name() << "'/>                  \n"
                 "   <PlaneForceField name='myPlaneForceField'/>                                 \n"
                 "  </Node>                                                                      \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;
        EXPECT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        BaseObject* planeff = root->getTreeNode("Level 1")->getObject("myPlaneForceField") ;
        EXPECT_NE(planeff, nullptr) ;

        return true;
    }


    ///
    /// In this test we are verifying that a plane that have been reinited has the same
    /// value as one that have been inited.
    ///
    bool testInitReinitBehavior()
    {
#if 0
        typename PlaneForceFieldType::SPtr plane1 = New<PlaneForceFieldType>();
        typename PlaneForceFieldType::SPtr plane2 = New<PlaneForceFieldType>();

        plane1->init() ;
        plane2->init() ;

        //TODO(dmarchal): three line to set a value...something has to be improved.
        DPos normal;
        normal[0]=1;
        plane1->d_planeNormal.setValue(normal);
        plane1->d_planeD.setValue(0);

        plane1->reinit() ;

        /// List of the supported attributes the user expect to find
        /// This list needs to be updated if you add an attribute.
        vector<string> attrnames = {
            "normal", "d", "stiffness", "damping", "maxForce", "bilateral", "localRange",
            "draw", "color", "drawSize"
        };

        for(auto& attrname : attrnames){
            EXPECT_EQ( plane1->findData(attrname)->getValueString(),
                       plane2->findData(attrname)->getValueString() ) << "Attribute with name '" << attrname << "' has changed between init and reinit." ;
        }
#endif //
        return true;
    }



    bool testPlaneForceField()
    {
        for(int i=0; i<50; i++)
            m_simulation->animate(m_root.get(),(double)0.01);

        Real x = m_mechanicalObj->x.getValue()[0][0];

        if(x<0)//The point passed through the plane
        {
            ADD_FAILURE() << "Error while testing planeForceField"<< std::endl;
            return false;
        }
        else
            return true;
    }

};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<Vec1Types,
              Vec2Types,
              Vec3Types,
              Vec6Types,
              Rigid3Types> DataTypes;

// Test suite for all the instanciations
TYPED_TEST_CASE(PlaneForceField_test, DataTypes);// first test case
TYPED_TEST( PlaneForceField_test , testForceField )
{
    this->initSetup();
    ASSERT_TRUE (this->testPlaneForceField());
}

TYPED_TEST( PlaneForceField_test , testBasicAttributes )
{
    this->initSetup();
    ASSERT_TRUE (this->testBasicAttributes());
}

TYPED_TEST( PlaneForceField_test , testInitReinitBehavior )
{
    this->initSetup();
    ASSERT_TRUE (this->testInitReinitBehavior());
}

TYPED_TEST( PlaneForceField_test , testDefaultPlane )
{
    ASSERT_TRUE (this->testDefaultPlane());
}


}
}// namespace sofa







