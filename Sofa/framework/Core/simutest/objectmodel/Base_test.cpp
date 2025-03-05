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
#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base ;
using sofa::core::objectmodel::ComponentState;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;

#include <sofa/defaulttype/RigidTypes.h>
using sofa::defaulttype::Rigid3Types;

#include <sofa/defaulttype/VecTypes.h>
using sofa::defaulttype::Vec3Types;

namespace customns
{
class CustomBaseObject : public BaseObject
{
public:
    SOFA_CLASS(CustomBaseObject, BaseObject);
};

template<class D>
class CustomBaseObjectT : public BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CustomBaseObjectT, D), BaseObject);

    static const std::string GetCustomClassName() { return "MyFakeClassName"; }
};

}

using customns::CustomBaseObject;
using customns::CustomBaseObjectT;

class Base_test: public BaseSimulationTest
{
public:
    ~Base_test() override {}
    void testComponentState()
    {
        EXPECT_MSG_NOEMIT(Error, Warning) ;
        const std::string scene = R"(
            <?xml version='1.0'?>
            <Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >
               <RequiredPlugin name='Sofa.Component.StateContainer'/>
               <DefaultAnimationLoop />
               <DefaultVisualManagerLoop />
               <Node name='child1'>
                  <MechanicalObject />
                  <Node name='child2'>
                  </Node>
               </Node>
            </Node>
        )";

        SceneInstance c("xml", scene) ;
        c.initScene() ;

        Node* root = c.root.get() ;
        ASSERT_NE(root, nullptr) ;

        ASSERT_NE(root->findData("componentState"), nullptr);
        root->d_componentState.setValue(ComponentState::Valid);
        ASSERT_EQ(root->d_componentState.getValue(), ComponentState::Valid);
        root->d_componentState.setValue(ComponentState::Loading);
        ASSERT_EQ(root->d_componentState.getValue(), ComponentState::Loading);
    }
};

TEST_F(Base_test , testComponentState )
{
    this->testComponentState();
}

TEST_F(Base_test , testBaseClass)
{
    EXPECT_EQ(CustomBaseObject::GetClass()->className, "CustomBaseObject");
    EXPECT_EQ(CustomBaseObject::GetClass()->templateName, "");
    EXPECT_EQ(CustomBaseObject::GetClass()->shortName, "customBaseObject");

    EXPECT_EQ(CustomBaseObjectT<Rigid3Types>::GetClass()->className, "MyFakeClassName");
    EXPECT_EQ(CustomBaseObjectT<Rigid3Types>::GetClass()->templateName, Rigid3Types::Name());
    EXPECT_EQ(CustomBaseObjectT<Rigid3Types>::GetClass()->shortName, "myFakeClassName");

    EXPECT_EQ(CustomBaseObjectT<Vec3Types>::GetClass()->className, "MyFakeClassName");
    EXPECT_EQ(CustomBaseObjectT<Vec3Types>::GetClass()->templateName, Vec3Types::Name());
    EXPECT_EQ(CustomBaseObjectT<Vec3Types>::GetClass()->shortName, "myFakeClassName");
}

TEST_F(Base_test , testGetClassName)
{
    const CustomBaseObject o;
    EXPECT_EQ(o.getClassName(), "CustomBaseObject");
    EXPECT_EQ(o.getTemplateName(), "");
    EXPECT_EQ(o.getTypeName(), "CustomBaseObject");
    EXPECT_EQ(o.getClass()->className, "CustomBaseObject");
}

