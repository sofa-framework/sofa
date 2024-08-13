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
#include <fstream>
#include <gtest/gtest.h>
#include <sofa/simulation/DefaultAnimationLoop.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/ObjectFactoryJson.h>
#include <sofa/simpleapi/SimpleApi.h>


namespace sofa
{
TEST(ObjectFactoryJson, noObject)
{
    core::ObjectFactory o;
    const auto dump = core::ObjectFactoryJson::dump(&o);
    EXPECT_EQ(dump, "[]");
}

TEST(ObjectFactoryJson, oneObject)
{
    core::ObjectFactory o;

    EXPECT_EQ(core::RegisterObject("foo")
        .add< simulation::DefaultAnimationLoop >().commitTo(&o), 1);

    const auto dump = core::ObjectFactoryJson::dump(&o);
    const std::string expectedDump = R"x([{"className":"DefaultAnimationLoop","creator":{"":{"class":{"categories":["AnimationLoop"],"className":"DefaultAnimationLoop","namespaceName":"sofa::simulation","parents":["BaseAnimationLoop"],"shortName":"defaultAnimationLoop","templateName":"","typeName":"DefaultAnimationLoop"},"object":{"data":[{"defaultValue":"unnamed","group":"","help":"object name","name":"name","type":"string"},{"defaultValue":"0","group":"","help":"if true, emits extra messages at runtime.","name":"printLog","type":"bool"},{"defaultValue":"","group":"","help":"list of the subsets the objet belongs to","name":"tags","type":"TagSet"},{"defaultValue":"","group":"","help":"this object bounding box","name":"bbox","type":"BoundingBox"},{"defaultValue":"Undefined","group":"","help":"The state of the component among (Dirty, Valid, Undefined, Loading, Invalid).","name":"componentState","type":"ComponentState"},{"defaultValue":"0","group":"","help":"if true, handle the events, otherwise ignore the events","name":"listening","type":"bool"},{"defaultValue":"1","group":"","help":"If true, compute the global bounding box of the scene at each time step. Used mostly for rendering.","name":"computeBoundingBox","type":"bool"},{"defaultValue":"0","group":"","help":"If true, solves all the ODEs in parallel","name":"parallelODESolving","type":"bool"}],"link":[{"destinationTypeName":"BaseContext","help":"Graph Node containing this object (or BaseContext::getDefault() if no graph is used)","name":"context"},{"destinationTypeName":"BaseObject","help":"Sub-objects used internally by this object","name":"slaves"},{"destinationTypeName":"BaseObject","help":"nullptr for regular objects, or master object for which this object is one sub-objects","name":"master"},{"destinationTypeName":"BaseNode","help":"Link to the scene's node that will be processed by the loop","name":"targetNode"}]},"target":""}},"description":"foo\n"}])x";
    EXPECT_EQ(dump, expectedDump);
}

template<class T>
class DummyComponent : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(DummyComponent<T>, BaseObject);
};

TEST(ObjectFactoryJson, oneTemplatedObject)
{
    core::ObjectFactory o;

    EXPECT_EQ(core::RegisterObject("foo")
        .add< DummyComponent<sofa::defaulttype::Vec3fTypes> >().commitTo(&o), 1);

    const auto dump = core::ObjectFactoryJson::dump(&o);
    const auto vec3name = core::objectmodel::BaseClassNameHelper::getTypeName<sofa::defaulttype::Vec3fTypes>();
    const std::string expectedDump = R"x([{"className":"DummyComponent","creator":{"Vec3f":{"class":{"categories":["_Miscellaneous"],"className":"DummyComponent","namespaceName":"sofa","parents":["BaseObject"],"shortName":"dummyComponent","templateName":"Vec3f","typeName":"DummyComponent<)x" + std::string{vec3name} + R"x(>"},"object":{"data":[{"defaultValue":"unnamed","group":"","help":"object name","name":"name","type":"string"},{"defaultValue":"0","group":"","help":"if true, emits extra messages at runtime.","name":"printLog","type":"bool"},{"defaultValue":"","group":"","help":"list of the subsets the objet belongs to","name":"tags","type":"TagSet"},{"defaultValue":"","group":"","help":"this object bounding box","name":"bbox","type":"BoundingBox"},{"defaultValue":"Undefined","group":"","help":"The state of the component among (Dirty, Valid, Undefined, Loading, Invalid).","name":"componentState","type":"ComponentState"},{"defaultValue":"0","group":"","help":"if true, handle the events, otherwise ignore the events","name":"listening","type":"bool"}],"link":[{"destinationTypeName":"BaseContext","help":"Graph Node containing this object (or BaseContext::getDefault() if no graph is used)","name":"context"},{"destinationTypeName":"BaseObject","help":"Sub-objects used internally by this object","name":"slaves"},{"destinationTypeName":"BaseObject","help":"nullptr for regular objects, or master object for which this object is one sub-objects","name":"master"}]},"target":""}},"description":"foo\n"}])x";
    EXPECT_EQ(dump, expectedDump);
}

TEST(ObjectFactoryJson, mainInstance)
{
    EXPECT_TRUE(sofa::simpleapi::importPlugin("Sofa.Component"));
    const auto dump = core::ObjectFactoryJson::dump(core::ObjectFactory::getInstance());
    EXPECT_NE(dump.find("MechanicalObject"), std::string::npos);
}
}
