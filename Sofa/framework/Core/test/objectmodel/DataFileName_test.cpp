/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/testing/config.h>

#include <sofa/core/objectmodel/BaseObjectDescription.h>
using sofa::core::objectmodel::BaseObjectDescription ;

#include <sofa/core/objectmodel/DataFileName.h>
using sofa::core::objectmodel::DataFileName ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository ;

#include <sofa/helper/system/SetDirectory.h>
using sofa::helper::system::SetDirectory ;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;

const std::string filename = "UtilsTest.ini";
const std::string fullpath = std::string(SOFA_TESTING_RESOURCES_DIR) + "/" + filename;

class DataFileName_test: public BaseTest
{
    DataFileName d_filename;
public:
    void doSetUp() override
    {
        DataRepository.addFirstPath( SOFA_TESTING_RESOURCES_DIR );
    }

    /// These two should return the same things
    void checkSetGetValues(){
        d_filename.setValue(filename) ;
        EXPECT_EQ( filename, d_filename.getValue()) ;

        d_filename.setValue(fullpath) ;
        EXPECT_EQ(fullpath, d_filename.getValue());
    }

    /// I see no reason why asking for the full path return a short path.
    void checkSetGetFullPath(){
        d_filename.setValue(filename) ;
        EXPECT_EQ(fullpath, d_filename.getFullPath() )  ;

        d_filename.setValue(fullpath) ;
        EXPECT_EQ(fullpath, d_filename.getFullPath() ) ;
    }

    void checkSetGetRelativePath(){
        d_filename.setValue(filename) ;
        EXPECT_EQ( filename, d_filename.getRelativePath() ) ;

        d_filename.setValue(fullpath) ;
        EXPECT_EQ( filename, d_filename.getRelativePath() ) ;
    }
};

TEST_F(DataFileName_test, checkSetGetValues)
{
    this->checkSetGetValues();
}

TEST_F(DataFileName_test, checkSetGetFullPath)
{
    this->checkSetGetFullPath();
}

TEST_F(DataFileName_test, checkSetGetRelativePath)
{
    this->checkSetGetRelativePath();
}
