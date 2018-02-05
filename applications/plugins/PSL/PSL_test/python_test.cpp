/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
/******************************************************************************
 * Contributors:                                                              *
 *    - damien.marchal@univ-lille1.fr                                         *
 *****************************************************************************/

#include <vector>

#include <SofaTest/Python_test.h>
using sofa::Python_test ;
using sofa::Python_test_list ;


#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

using std::vector;
using std::string;

namespace
{

/// Be sure that PSL & SofaPython plugin are loaded.
void anInit(){
    static bool _inited_ = false;
    if(!_inited_){
        PluginManager::getInstance().loadPlugin("PSL") ;
        PluginManager::getInstance().loadPlugin("SofaPython") ;
    }
}

/// static build of the test list
static struct Tests : public Python_test_list
{
    Tests()
    {
        static const std::string testPath = std::string(PYTHON_TESTFILES_DIR);

        addTest( "pslloader_test.py", testPath, {} );

    }
} python_tests;


/// run test list
INSTANTIATE_TEST_CASE_P(Batch,
                        Python_test,
                        ::testing::ValuesIn(python_tests.list));

TEST_P(Python_test, psl_python_tests)
{
    anInit();
    run(GetParam());
}

}
