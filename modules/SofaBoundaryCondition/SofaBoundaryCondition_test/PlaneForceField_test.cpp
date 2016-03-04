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

#include <SofaBoundaryCondition/PlaneForceField.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/simulation/graph/DAGSimulation.h>
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

template <typename _DataTypes>
struct PlaneForceField_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::DPos DPos;
    typedef typename Coord::value_type Real;

    typedef sofa::component::forcefield::PlaneForceField<DataTypes> PlaneForceFieldType;
    typedef container::MechanicalObject<DataTypes> MechanicalObjectType;
    typedef odesolver::EulerImplicitSolver EulerImplicitSolverType;
    typedef linearsolver::CGLinearSolver< linearsolver::GraphScatteredMatrix,linearsolver::GraphScatteredVector > CGLinearSolverType;

    simulation::Node::SPtr root;         // Root of the scene graph, created by the constructor and re-used in the tests
    simulation::Simulation* simulation;
    typename PlaneForceFieldType::SPtr planeForceFieldSPtr;
    typename MechanicalObjectType::SPtr mechanicalObj;

    /* Test if a mechanical object (point) subject only to the gravity is effectively stopped
     * by the plane force field.
     * In the special case where : stiffness = 500, damping = 5 and maxForce = 0 (default values)
    */

    void SetUp()
    {
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        // Create the scene
        root = simulation->createNewGraph("root");

        typename EulerImplicitSolverType::SPtr eulerImplicitSolver = New<EulerImplicitSolverType>();
        root->addObject(eulerImplicitSolver);

        typename CGLinearSolverType::SPtr cgLinearSolver = New<CGLinearSolverType>();
        root->addObject(cgLinearSolver);
        cgLinearSolver->f_maxIter.setValue(25);
        cgLinearSolver->f_tolerance.setValue(1e-5);
        cgLinearSolver->f_smallDenominatorThreshold.setValue(1e-5);

        mechanicalObj = New<MechanicalObjectType>();
        root->addObject(mechanicalObj);
        Coord point;
        point[0]=1;
        VecCoord points;
        points.clear();
        points.push_back(point);

        mechanicalObj->x0.setValue(points);

        std::string name = DataTypeInfo<DataTypes>::name();
#ifndef SOFA_FLOAT
        if(name=="Rigid")
        {
            typename mass::UniformMass<Rigid3dTypes,Rigid3dMass>::SPtr uniformMass = New<mass::UniformMass<Rigid3dTypes,Rigid3dMass> >();
            root->addObject(uniformMass);
            uniformMass->totalMass.setValue(1);
        }
        else if(name=="Vec1d" )
        {
            typename mass::UniformMass<Vec1dTypes,double>::SPtr uniformMass = New<mass::UniformMass<Vec1dTypes,double> >();
            root->addObject(uniformMass);
            uniformMass->totalMass.setValue(1);
        }
        else if(name=="Vec2d" )
        {
            typename mass::UniformMass<Vec2dTypes,double>::SPtr uniformMass = New<mass::UniformMass<Vec2dTypes,double> >();
            root->addObject(uniformMass);
            uniformMass->totalMass.setValue(1);
        }
        else if(name=="Vec3d" )
        {
            typename mass::UniformMass<Vec3dTypes,double>::SPtr uniformMass = New<mass::UniformMass<Vec3dTypes,double> >();
            root->addObject(uniformMass);
            uniformMass->totalMass.setValue(1);
        }
        else if(name=="Vec6d")
        {
            typename mass::UniformMass<Vec6dTypes,double>::SPtr uniformMass = New<mass::UniformMass<Vec6dTypes,double> >();
            root->addObject(uniformMass);
            uniformMass->totalMass.setValue(1);
        }
        else
#endif
#ifndef SOFA_DOUBLE
        if(name=="Rigid3f")
        {
            typename mass::UniformMass<Rigid3fTypes,Rigid3fMass>::SPtr uniformMass = New<mass::UniformMass<Rigid3fTypes,Rigid3fMass> >();
            root->addObject(uniformMass);
            uniformMass->totalMass.setValue(1);
        }
        else if(name=="Vec1f" )
        {
            typename mass::UniformMass<Vec1fTypes,float>::SPtr uniformMass = New<mass::UniformMass<Vec1fTypes,float> >();
            root->addObject(uniformMass);
            uniformMass->totalMass.setValue(1);
        }
        else if(name=="Vec2f" )
        {
            typename mass::UniformMass<Vec2fTypes,float>::SPtr uniformMass = New<mass::UniformMass<Vec2fTypes,float> >();
            root->addObject(uniformMass);
            uniformMass->totalMass.setValue(1);
        }
        else if(name=="Vec3f" )
        {
            typename mass::UniformMass<Vec3fTypes,float>::SPtr uniformMass = New<mass::UniformMass<Vec3fTypes,float> >();
            root->addObject(uniformMass);
            uniformMass->totalMass.setValue(1);
        }
        else if(name=="Vec6f")
        {
            typename mass::UniformMass<Vec6fTypes,float>::SPtr uniformMass = New<mass::UniformMass<Vec6fTypes,float> >();
            root->addObject(uniformMass);
            uniformMass->totalMass.setValue(1);
        }
        else
#endif
        return;

        /*Create the plane force field*/
        planeForceFieldSPtr = New<PlaneForceFieldType>();
        planeForceFieldSPtr->planeD.setValue(0);

        DPos normal;
        normal[0]=1;
        planeForceFieldSPtr->planeNormal.setValue(normal);

        root->setGravity(Vec3d(-9.81,0,0));
    }

    void init_Setup()
    {
        // Init
        sofa::simulation::getSimulation()->init(root.get());
    }

    bool test_planeForceField()
    {
        for(int i=0; i<50; i++)
            simulation->animate(root.get(),(double)0.01);

        Real x = mechanicalObj->x.getValue()[0][0];

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
typedef Types<defaulttype::Vec1Types, defaulttype::Vec2Types, defaulttype::Vec3Types, defaulttype::Vec6Types, defaulttype::Rigid3Types> DataTypes;// the types to instanciate.
// Test suite for all the instanciations
TYPED_TEST_CASE(PlaneForceField_test, DataTypes);// first test case
TYPED_TEST( PlaneForceField_test , testForceField )
{
    this->init_Setup();
    ASSERT_TRUE (this->test_planeForceField());
}



}
}// namespace sofa







