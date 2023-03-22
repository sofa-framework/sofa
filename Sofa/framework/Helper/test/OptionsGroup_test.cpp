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
#include <sofa/helper/OptionsGroup.h>
#include <gtest/gtest.h>

namespace sofa
{

using sofa::helper::OptionsGroup;

TEST(OptionsGroup, constructors)
{
    const OptionsGroup opt0{};
    const OptionsGroup opt1(std::vector<std::string>{"optionA", "optionB"});
    const OptionsGroup opt2(std::set<std::string>{"optionA", "optionB"});
    const OptionsGroup opt3 = OptionsGroup{{"optionA", "optionB"}}.setSelectedItem(1);
    const OptionsGroup opt4 = OptionsGroup{{"optionA", "optionB"}}.setSelectedItem(10);
    const OptionsGroup opt5 = OptionsGroup{"optionA", "optionB"}.setSelectedItem(1);

    EXPECT_EQ(opt0.getSelectedId(), 0);
    EXPECT_EQ(opt1.getSelectedId(), 0);
    EXPECT_EQ(opt2.getSelectedId(), 0);
    EXPECT_EQ(opt3.getSelectedId(), 1);
    EXPECT_EQ(opt4.getSelectedId(), 0);
    EXPECT_EQ(opt5.getSelectedId(), 1);

    EXPECT_EQ(opt1.getSelectedItem(), "optionA");
    EXPECT_EQ(opt2.getSelectedItem(), "optionA");
    EXPECT_EQ(opt3.getSelectedItem(), "optionB");
    EXPECT_EQ(opt4.getSelectedItem(), "optionA");
    EXPECT_EQ(opt5.getSelectedItem(), "optionB");
}

}
