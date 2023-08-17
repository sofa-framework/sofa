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
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/simulation/graph/SimpleApi.h>
using namespace sofa::simpleapi;

#include <sofa/component/solidmechanics/spring/RestShapeSpringsForceField.h>
#include <sofa/simulation/Node.h>

#include <sofa/component/statecontainer/MechanicalObject.h>
using sofa::component::statecontainer::MechanicalObject;

using sofa::defaulttype::Vec3Types;
using sofa::defaulttype::Rigid3dTypes;
using sofa::helper::ReadAccessor;
using sofa::Data;

/// Test suite for RestShapeSpringsForceField
class RestStiffSpringsForceField_test : public BaseTest
{
public:
    ~RestStiffSpringsForceField_test() override;
    sofa::simulation::Node::SPtr createScene(const std::string& type);

    template<class Type>
    void testDefaultBehavior(sofa::simulation::Node::SPtr root);

    template<class Type>
    void checkDifference(MechanicalObject<Type>& mo, bool isFixed);
};

RestStiffSpringsForceField_test::~RestStiffSpringsForceField_test()
{
}

sofa::simulation::Node::SPtr RestStiffSpringsForceField_test::createScene(const std::string& type)
{
    const auto theSimulation = createSimulation();
    auto theRoot = createRootNode(theSimulation, "root");
    sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Backward");
    sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Iterative");
    sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
    sofa::simpleapi::importPlugin("Sofa.Component.Mass");
    
    createObject(theRoot, "DefaultAnimationLoop");
    createObject(theRoot, "EulerImplicitSolver");
    createObject(theRoot, "CGLinearSolver", {{ "iterations", "25" }, { "tolerance", "1e-5" }, {"threshold", "1e-5"}});

    /// Create an object with a mass and use a rest shape spring ff so it stays
    /// at the initial position
    const auto fixedObject = createChild(theRoot, "fixedObject");
    auto fixedObject_dofs = createObject(fixedObject, "MechanicalObject", {{"name","dofs"},
                                                                           {"size","10"},
                                                                           {"template",type}});
    createObject(fixedObject, "UniformMass", {{"totalMass", "1"}});

    createObject(fixedObject, "RestShapeSpringsForceField", {{"stiffness","1000"}});

    const auto movingObject = createChild(theRoot, "movingObject");
    auto movingObject_dofs =createObject(movingObject, "MechanicalObject", {{"name","dofs"},
                                                                            {"size","10"},
                                                                            {"template",type}});
    createObject(movingObject, "UniformMass", {{"totalMass", "1"}});

    sofa::simulation::node::initRoot(theRoot.get());
    for(unsigned int i=0;i<20;i++)
    {
        sofa::simulation::node::animate(theRoot.get(), 0.01_sreal);
    }
    return theRoot;
}

template<class Type>
void RestStiffSpringsForceField_test::checkDifference(MechanicalObject<Type>& mo, bool isFixed)
{
    ReadAccessor< Data<typename Type::VecCoord> > positions = mo.x;
    ReadAccessor< Data<typename Type::VecCoord> > rest_positions = mo.x0;
    for(size_t i=0;i<positions.size();i++)
    {
        sofa::type::Vec3 pos = Type::getCPos(positions[i]) ;
        sofa::type::Vec3 rpos = Type::getCPos(rest_positions[i]) ;

        if(isFixed)
        {
            ASSERT_NEAR( pos.x(), rpos.x(), 0.1 );
            ASSERT_NEAR( pos.y(), rpos.y(), 0.1 );
            ASSERT_NEAR( pos.z(), rpos.z(), 0.1 );
        }
        else
        {
            ASSERT_TRUE( fabs(pos.x()-rpos.x()) < 1 );
            ASSERT_TRUE( fabs(pos.y()-rpos.y()) < 1 );
            ASSERT_TRUE( fabs(pos.z()-rpos.z()) < 1 );
        }
    }
}

template<class Type>
void RestStiffSpringsForceField_test::testDefaultBehavior(sofa::simulation::Node::SPtr root)
{
    auto fixedDofs = dynamic_cast<MechanicalObject<Type>*>(root->getChild("fixedObject")->getObject("dofs"));
    ASSERT_TRUE( fixedDofs != nullptr );

    auto movingDofs = dynamic_cast<MechanicalObject<Type>*>(root->getChild("movingObject")->getObject("dofs"));
    ASSERT_TRUE( movingDofs != nullptr );

    checkDifference(*fixedDofs, true);
    checkDifference(*movingDofs, false);
}


TEST_F(RestStiffSpringsForceField_test, defaultBehaviorVec3)
{
    this->testDefaultBehavior<Vec3Types>(this->createScene("Vec3"));
}

TEST_F(RestStiffSpringsForceField_test, defaultBehaviorRigid3)
{
    this->testDefaultBehavior<sofa::defaulttype::Rigid3Types>(this->createScene("Rigid3"));
}
