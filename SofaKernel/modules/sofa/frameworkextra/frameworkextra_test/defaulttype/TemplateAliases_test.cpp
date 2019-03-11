/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/defaulttype/TemplatesAliases.h>
using sofa::defaulttype::TemplateAliases;
using sofa::defaulttype::TemplateAlias;

#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest ;

namespace sofa {

class TemplateAliasTest : public BaseSimulationTest
{
protected:
    bool registerAlias(const std::string& alias, const std::string& target, bool succeed, bool warn)
    {
        if(TemplateAliases::addAlias(alias, target, warn))
        {
            EXPECT_TRUE((TemplateAliases::resolveAlias(alias) == target));

            const TemplateAlias* re = TemplateAliases::getTemplateAlias(alias);
            EXPECT_TRUE((re->first == target));
            EXPECT_TRUE((re->second == warn));
            EXPECT_TRUE(succeed);
        }else
        {
            EXPECT_FALSE(succeed);
        }
        return true;
    }
};

TEST_F(TemplateAliasTest, Register)
{
    registerAlias("TheAlias1", "TheResult1", true, true);
    registerAlias("TheAlias2", "TheResult2", true, false);
    registerAlias("TheAlias2", "TheResult2", false, true);
}


} //namespace sofa
