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
#include <sofa/helper/vector[string].h>
#include <sofa/helper/testing/NumericTest.h>

namespace
{

class vector_string_test : public sofa::helper::testing::NumericTest<>,
        public ::testing::WithParamInterface<std::vector<std::string>>
{
public:
    void checkVector(const std::vector<std::string>& params)
    {
        std::string result = params[1];
        std::string errtype = params[2];

        std::stringstream in(params[0]);
        std::stringstream out ;

        sofa::helper::vector<std::string> v;
        v.read(in) ;
        v.write(out) ;

        /// If the parsed version is different that the written version & there is no warning...this
        /// means a problem will be un-noticed.
        EXPECT_EQ( result, out.str() ) ;
    }
};

TEST_P(vector_string_test, checkReadWriteBehavior)
{
    this->checkVector(GetParam()) ;
}

/// List of all the tests following this convention:
/// {"string to read", "expected value", "None"} to indicate there is no warning to expect
/// {"string to read", "expected value", "Warning"} to indicate there is warning to expect
std::vector<std::vector<std::string>> values =
{
    /// First test valid values
    {"[]", "[]", "None"},
    {"[\'\']", "[\"\"]", "None"},
    {"[\"\"]", "[\"\"]", "None"},
    {"[\"\", 'a']", "[\"\",\"a\"]", "None"},
    {"[\"a\", \"\"]", "[\"a\",\"\"]", "None"},
    {"['string1', 'string2', 'string3']", "[\"string1\", \"string2\", \"string3\"]", "None"},
    {"[\"string1\", \"string2\", \"string3\"]", "[\"string1\", \"string2\", \"string3\"]", "None"},
    {"['string 1', 'string 2', ' string 3 ']", "[\"string 1\", \"string 2\", \" string 3 \"]", "None"},
};

INSTANTIATE_TEST_CASE_P(checkReadWriteBehavior,
                        vector_string_test,
                        ::testing::ValuesIn(values));


}
