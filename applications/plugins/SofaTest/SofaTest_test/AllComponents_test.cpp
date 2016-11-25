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
* Contributors:                                                               *
*      - damien.machal@univ-lille1.fr                                         *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <SofaSimulationTree/init.h>
#include <SofaSimulationTree/TreeSimulation.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem ;

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

#include <SofaTest/Sofa_test.h>

/// A per-file namespace to avoid name clash for static variables.
namespace allcomponents_test
{

bool recursiveListDirectory(const std::string& path, std::map<std::string, bool>& fileMap)
{
    std::vector<std::string> dircontent ;
    if( FileSystem::listDirectory(path, dircontent) ){
        msg_warning("AllComponentTest") << "Unable to scan the directory: '"<< path <<"'." ;
        return true ;
    }

    for(auto& name : dircontent){
        std::string childname=path+"/"+name;
        if( FileSystem::isDirectory(childname) ){
            recursiveListDirectory(childname, fileMap);
        }else
        {
            fileMap[name] = true;
        }
    }

    return false;
}

class AllComponents_test: public testing::Test
{

    void SetUp(){
        sofa::component::initComponentBase();
        sofa::component::initComponentCommon();
        sofa::component::initComponentGeneral();
        sofa::component::initComponentAdvanced();
        sofa::component::initComponentMisc();
    }

    void TearDown(){

    }

public:
    void buildListOfExamples()
    {

    }

    void checkExampleFileScene()
    {
        unsigned int numberOfComponentWithAnExample = 0 ;
        std::map<std::string, bool> file2bool;

        ASSERT_FALSE(recursiveListDirectory(std::string(SOFA_SRC_DIR)+"/examples", file2bool));

        std::vector<ObjectFactory::ClassEntry::SPtr> components;

        std::stringstream tmp;

        ObjectFactory::getInstance()->getAllEntries(components);
        for(ObjectFactory::ClassEntry::SPtr& entry : components)
        {
            if(entry){
                if(file2bool.find(entry->className+".scn") != file2bool.end() ||
                   file2bool.find(entry->className+".xml") != file2bool.end()){

                   numberOfComponentWithAnExample++;
                }else{
                    tmp << " " << entry->className ;
                }
            }
        }

        EXPECT_EQ(components.size(), numberOfComponentWithAnExample) << " Only " << numberOfComponentWithAnExample
                                                                     << "/" << components.size()
                                                                     << " of sofa components have an example. If you have a good example for one of the following "
                                                                        "components: [" << tmp.str() <<  "] please commit it to the example directory" ;
    }
};

/// performing the regression test on every plugins/projects
TEST_F(AllComponents_test, checkExampleFileScene_OpenIssue)
{
    this->checkExampleFileScene();
}



} // namespace allcomponents_test

