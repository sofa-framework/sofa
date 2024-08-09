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
#include <sofa/helper/StringUtils.h>
#include <gtest/gtest.h>

namespace sofa
{

// Test cases for removeTrailingCharacter
TEST(removeTrailingCharacterTest, emptyString)
{
    std::string_view input = "";
    constexpr char character = ' ';
    std::string_view result = sofa::helper::removeTrailingCharacter(input, character);
    EXPECT_EQ(result, "");
}

TEST(removeTrailingCharacterTest, singleTrailingCharacter)
{
    std::string_view input = "Hello!";
    constexpr char character = '!';
    std::string_view result = sofa::helper::removeTrailingCharacter(input, character);
    EXPECT_EQ(result, "Hello");
}

TEST(removeTrailingCharacterTest, multipleTrailingCharacters)
{
    std::string_view input = "Hello...";
    constexpr char character = '.';
    std::string_view result = sofa::helper::removeTrailingCharacter(input, character);
    EXPECT_EQ(result, "Hello");
}

// Test cases for removeTrailingCharacters
TEST(removeTrailingCharactersTest, emptyString)
{
    constexpr std::string_view input = "";
    const std::initializer_list<char> characters = {' ', '\t'};
    const std::string_view result = sofa::helper::removeTrailingCharacters(input, characters);
    EXPECT_EQ(result, "");
}

TEST(removeTrailingCharactersTest, noTrailingCharacters)
{
    constexpr std::string_view input = "Hello";
    const std::initializer_list<char> characters = {'o', 'x'};
    const std::string_view result = sofa::helper::removeTrailingCharacters(input, characters);
    EXPECT_EQ(result, "Hell");
}

TEST(removeTrailingCharactersTest, singleTrailingCharacter)
{
    std::string_view input = "Hello!";
    const std::initializer_list<char> characters = {'!'};
    const std::string_view result = sofa::helper::removeTrailingCharacters(input, characters);
    EXPECT_EQ(result, "Hello");
}

TEST(removeTrailingCharactersTest, multipleTrailingCharacters)
{
    constexpr std::string_view input = "Hello...";
    const std::initializer_list<char> characters = {'.'};
    const std::string_view result = sofa::helper::removeTrailingCharacters(input, characters);
    EXPECT_EQ(result, "Hello");
}

TEST(removeTrailingCharactersTest, mixOfCharacters)
{
    constexpr std::string_view input = "Hello!!!\t";
    const std::initializer_list<char> characters = {'!', '\t'};
    const std::string_view result = sofa::helper::removeTrailingCharacters(input, characters);
    EXPECT_EQ(result, "Hello");
}

TEST(StringUtilsTest, string_to_widestring_to_string)
{
    std::string ascii_chars;
    for (char c = 32 ; c <= 126 ; c++)
        ascii_chars.push_back(c);
    EXPECT_EQ(ascii_chars, helper::narrowString(helper::widenString(ascii_chars)));

    // This test will pass if the executable has been executed with a unicode-compliant locale
    // Windows and MacOS are unicode by default
    // But it seems some linux distrib are not (?)
#ifdef __linux
    if(std::locale("").name().find("UTF-8") == std::string::npos)
    {
        return;
    }
#endif

    const std::string s("chaîne de test avec des caractères accentués");
    EXPECT_EQ(s, helper::narrowString(helper::widenString(s)));
}

TEST(StringUtilsTest, widestring_to_string_to_widestring)
{
    const std::string s("chaîne de test avec des caractères accentués");
    const std::wstring ws = helper::widenString(s);
    EXPECT_EQ(ws, helper::widenString(helper::narrowString(ws)));
}

TEST(StringUtilsTest, downcaseString)
{
    EXPECT_EQ("abcdef", helper::downcaseString("abcdef"));
    EXPECT_EQ("abcdef", helper::downcaseString("ABCDEF"));
    EXPECT_EQ("abcdef", helper::downcaseString("AbCdEf"));
    EXPECT_EQ("abcdef", helper::downcaseString("ABCDEF"));
}

TEST(StringUtilsTest, upcaseString)
{
    EXPECT_EQ("ABCDEF", helper::upcaseString("abcdef"));
    EXPECT_EQ("ABCDEF", helper::upcaseString("ABCDEF"));
    EXPECT_EQ("ABCDEF", helper::upcaseString("AbCdEf"));
    EXPECT_EQ("ABCDEF", helper::upcaseString("ABCDEF"));
}
}
