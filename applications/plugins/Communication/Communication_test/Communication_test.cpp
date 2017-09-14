/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test ;

#include <sofa/defaulttype/Vec.h>

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;
using sofa::core::ExecParams;
using sofa::core::objectmodel::BaseData ;
using sofa::core::objectmodel::BaseObject;



#include <sofa/helper/vectorData.h>
using sofa::helper::vectorData;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

namespace sofa
{
namespace component
{
namespace container
{
namespace _distancegrid_
{
using sofa::defaulttype::Vector3 ;

class MyComponent : public BaseObject
{
public:
    MyComponent() :
      //   , vf_inputs( this, "input", "Input vector", helper::DataEngineInput )
      d_positionsOut(this, "positionOut", "")
    , d_positionsIn(this, "positionIn", "")
    {
        f_listening = true ;
    }

    virtual void init() override
    {
        d_positionsOut.resize(100) ;
        d_positionsIn.resize(100) ;
    }

    virtual void handleEvent(sofa::core::objectmodel::Event *event) override
    {
        std::cout << "YOLO " << std::endl ;
    }

    vectorData<Vector3>  d_positionsOut ;
    vectorData<Vector3> d_positionsIn ;
} ;

int mclass = sofa::core::RegisterObject("").add<MyComponent>();


class Communication_test : public Sofa_test<>
{
public:
    void checkPerformances(int numstep)
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0'?>"
                  "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       \n"
                  "<RequiredPlugin name='Communication'/>                              \n"
                  "<DefaultAnimationLoop />                                            \n"
                  "<MyComponent name='aName'/>                                         \n"
                  "<ServerCommunicationOSC name='oscSend' job='sender' port='6000'  refreshRate='1000'/>                                          \n"
                  "<CommunicationSubscriber name='subSend' communication='@oscSend' subject='/sender' source='@light1' arguments='positionsOut'/> \n"
                  "<ServerCommunicationOSC name='oscRec' job='receiver' port='6010'  refreshRate='2'/>                                            \n"
                  "<CommunicationSubscriber name='subRec' communication='@oscRec' subject='/receive' source='@light1' arguments='positionsIn'/>   \n"
                  "</Node>" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene1.str().c_str(),
                                                          scene1.str().size()) ;

        root->init(ExecParams::defaultInstance()) ;

        for(unsigned int i=0;i<numstep;i++)
        {
            sofa::simulation::getSimulation()->animate(root.get(), 0.001);
        }
    }
};

TEST_F(Communication_test, checkPerformancs) {
    ASSERT_NO_THROW(this->checkPerformances(1000)) ;
}


} // __distance_grid__
} // container
} // component
} // sofa
