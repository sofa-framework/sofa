/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <vector>
using std::vector;

#include <string>
using std::string;

#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test ;

#include<sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::Node ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::ExecParams ;

#include <SofaOpenglVisual/OglLabel.h>
using sofa::component::visualmodel::OglLabel ;

#include <sofa/defaulttype/RGBAColor.h>
using sofa::defaulttype::RGBAColor ;

class OglLabelTest : public Sofa_test<>
{
public:
    void checkExcludingAttributes()
    {
        EXPECT_MSG_EMIT(Warning) ;

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "  <OglLabel name='label1' color='0 0 0 0' selectContrastingColor='true' />     \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(nullptr, root.get()) ;
        root->init(ExecParams::defaultInstance()) ;


        BaseObject* lm = root->getObject("label1") ;
        ASSERT_NE(nullptr, lm) ;

        OglLabel* ogllabel = dynamic_cast<OglLabel*>(lm);
        ASSERT_NE(nullptr, ogllabel) ;


        EXPECT_TRUE(ogllabel->d_selectContrastingColor.getValue()) ;
        EXPECT_EQ(RGBAColor::fromFloat(1,1,1,1), ogllabel->d_color.getValue()) ;
    }


    void checkDeprecatedAttribute()
    {
        EXPECT_MSG_EMIT(Deprecated) ;

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "  <OglLabel name='label1' color='contrast' printLog='true'/>                   \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(nullptr, root.get()) ;
        root->init(ExecParams::defaultInstance()) ;

        BaseObject* lm = root->getObject("label1") ;
        ASSERT_NE(nullptr, lm) ;

        OglLabel* ogllabel = dynamic_cast<OglLabel*>(lm);
        ASSERT_NE(nullptr, ogllabel) ;


        EXPECT_TRUE(ogllabel->d_selectContrastingColor.getValue()) ;
        EXPECT_EQ(RGBAColor::fromFloat(1,1,1,1), ogllabel->d_color.getValue()) ;
    }

    void checkAttributes()
    {
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "  <OglLabel name='label1'/>                                                    \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        BaseObject* lm = root->getObject("label1") ;
        ASSERT_NE(lm, nullptr) ;

        /// List of the supported attributes the user expect to find
        /// This list needs to be updated if you add an attribute.
        vector<string> attrnames = {
            "prefix", "label", "suffix", "x", "y", "fontsize", "color",
            "selectContrastingColor", "updateLabelEveryNbSteps",
            "visible"};

        for(auto& attrname : attrnames)
            EXPECT_NE( lm->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;
    }
};

TEST_F(OglLabelTest, checkAttributes)
{
    this->checkAttributes();
}

TEST_F(OglLabelTest, checkDeprecatedAttribute)
{
    this->checkDeprecatedAttribute();
}

TEST_F(OglLabelTest, checkExcludingAttributes)
{
    this->checkExcludingAttributes();
}
