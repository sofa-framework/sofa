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
#include <vector>
using std::vector;

#include <PluginExample/MyBehaviorModel.h>

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest;

using testing::Types;

namespace {

class MyBehaviorModel_test : public BaseTest,
                             public ::testing::WithParamInterface<unsigned>
{
public:
    using MyBehaviorModel = sofa::component::behaviormodel::MyBehaviorModel;

    void TearDown()
    {

    }

    void SetUp()
    {
        m_behaviorModel = sofa::core::objectmodel::New< MyBehaviorModel >();
    }

    void dummyTest(unsigned param)
    {
        m_behaviorModel->d_regularUnsignedData.setValue(param);
        auto regularUnsignedDataFromBehaviorModel = sofa::helper::getReadAccessor(m_behaviorModel->d_regularUnsignedData);

        EXPECT_EQ(regularUnsignedDataFromBehaviorModel, param);
    }

private:
    MyBehaviorModel::SPtr m_behaviorModel;

};

std::vector<unsigned> params = {
    { 1 },
    { 2 },
    { 3 }
};

/// run the tests
TEST_P(MyBehaviorModel_test, dummyTest) {
    unsigned param = GetParam();
    dummyTest(param);
}


}
