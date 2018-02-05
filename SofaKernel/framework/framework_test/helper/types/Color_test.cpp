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
#include <sofa/defaulttype/Vec.h>
using sofa::defaulttype::Vec4d ;

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data ;

#include <sofa/defaulttype/RGBAColor.h>
using sofa::defaulttype::RGBAColor ;

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

class Color_Test : public BaseTest,
                   public ::testing::WithParamInterface<std::vector<std::string>>
{
public:
    void checkCreateFromString() ;
    void checkCreateFromDouble() ;
    void checkEquality() ;
    void checkGetSet() ;
    void checkColorDataField() ;
    void checkConstructors() ;
    void checkStreamingOperator(const std::vector<std::string>&) ;
    void checkDoubleStreamingOperator(const std::vector<std::string>&) ;
};

void Color_Test::checkCreateFromString()
{
    EXPECT_EQ( RGBAColor::fromString("white"), RGBAColor(1.0,1.0,1.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("black"), RGBAColor(0.0,0.0,0.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("red"), RGBAColor(1.0,0.0,0.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("green"), RGBAColor(0.0,1.0,0.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("blue"), RGBAColor(0.0,0.0,1.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("cyan"), RGBAColor(0.0,1.0,1.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("magenta"), RGBAColor(1.0,0.0,1.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("yellow"), RGBAColor(1.0,1.0,0.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("gray"), RGBAColor(0.5,0.5,0.5,1.0) ) ;

    RGBAColor color;
    EXPECT_TRUE( RGBAColor::read("white", color) ) ;
    EXPECT_FALSE( RGBAColor::read("invalidcolor", color) ) ;

    /// READ RGBA colors
    EXPECT_EQ( RGBAColor::fromString("1 2 3 4"), RGBAColor(1.0,2.0,3.0,4.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("0 0 3 4"), RGBAColor(0.0,0.0,3.0,4.0) ) ;

    /// READ RGB colors
    EXPECT_EQ( RGBAColor::fromString("1 2 3"), RGBAColor(1.0,2.0,3.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("0 0 3"), RGBAColor(0.0,0.0,3.0,1.0) ) ;

    RGBAColor color2;
    EXPECT_TRUE( RGBAColor::read("1 2 3 4", color2) ) ;
    EXPECT_EQ( color2, RGBAColor(1,2,3,4));

    EXPECT_TRUE( RGBAColor::read("0 0 3 4", color2) ) ;
    EXPECT_EQ( color2, RGBAColor(0,0,3,4));

    EXPECT_TRUE( RGBAColor::read("1 2 3", color2) ) ;
    EXPECT_EQ( color2, RGBAColor(1,2,3,1));

    EXPECT_FALSE( RGBAColor::read("1 2 3 4 5", color2) ) ;
    EXPECT_FALSE( RGBAColor::read("1 a 3 4", color2) ) ;
    EXPECT_FALSE( RGBAColor::read("-1 2 3 5", color2) ) ;

    ///# short hexadecimal notation
    EXPECT_EQ( RGBAColor::fromString("#000"), RGBAColor(0.0,0.0,0.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("#0000"), RGBAColor(0.0,0.0,0.0,0.0) ) ;

    EXPECT_TRUE( RGBAColor::read("#ABA", color2) ) ;
    EXPECT_TRUE( RGBAColor::read("#FFAA", color2) ) ;

    EXPECT_TRUE( RGBAColor::read("#aba", color2) ) ;
    EXPECT_TRUE( RGBAColor::read("#ffaa", color2) ) ;

    EXPECT_FALSE( RGBAColor::read("#ara", color2) ) ;
    EXPECT_FALSE( RGBAColor::read("#ffap", color2) ) ;
    EXPECT_FALSE( RGBAColor::read("##fapa", color2) ) ;
    EXPECT_FALSE( RGBAColor::read("#f#apa", color2) ) ;

    ///# long hexadecimal notation
    EXPECT_EQ( RGBAColor::fromString("#000000"), RGBAColor(0.0,0.0,0.0,1.0) ) ;
    EXPECT_EQ( RGBAColor::fromString("#00000000"), RGBAColor(0.0,0.0,0.0,0.0) ) ;

    EXPECT_TRUE( RGBAColor::read("#AABBAA", color2) ) ;
    EXPECT_TRUE( RGBAColor::read("#FFAA99AA", color2) ) ;

    EXPECT_FALSE( RGBAColor::read("#aabraa", color2) ) ;
    EXPECT_FALSE( RGBAColor::read("#fffapaba", color2) ) ;
    EXPECT_FALSE( RGBAColor::read("##fpaapddda", color2) ) ;
    EXPECT_FALSE( RGBAColor::read("#fasdqdpa", color2) ) ;

}

void Color_Test::checkCreateFromDouble()
{
    EXPECT_EQ( RGBAColor::fromFloat(1.0,1.0,1.0,1.0), RGBAColor(1.0,1.0,1.0,1.0)) ;
    EXPECT_EQ( RGBAColor::fromFloat(1.0,0.0,1.0,1.0), RGBAColor(1.0,0.0,1.0,1.0)) ;
    EXPECT_EQ( RGBAColor::fromFloat(1.0,1.0,0.0,1.0), RGBAColor(1.0,1.0,0.0,1.0)) ;
    EXPECT_EQ( RGBAColor::fromFloat(1.0,1.0,1.0,0.0), RGBAColor(1.0,1.0,1.0,0.0)) ;

    Vec4d tt(2,3,4,5) ;
    EXPECT_EQ( RGBAColor::fromVec4(tt), RGBAColor(2,3,4,5)) ;
}


void Color_Test::checkConstructors()
{
    EXPECT_EQ( RGBAColor(sofa::defaulttype::Vec<4,float>(1,2,3,4)), RGBAColor(1,2,3,4) ) ;
}


void Color_Test::checkGetSet()
{
    RGBAColor a;
    a.r(1);
    EXPECT_EQ(a.r(), 1.0) ;

    a.g(2);
    EXPECT_EQ(a.g(), 2.0) ;

    a.b(3);
    EXPECT_EQ(a.b(), 3.0) ;

    a.a(4);
    EXPECT_EQ(a.a(), 4.0) ;

    EXPECT_EQ(a, RGBAColor(1.0,2.0,3.0,4.0)) ;
}

void Color_Test::checkStreamingOperator(const std::vector<std::string>& p)
{
    assert(p.size()==3) ;

    std::stringstream input ;
    std::string result = p[1] ;
    std::string successOrFail = p[2] ;

    input << p[0] ;
    RGBAColor color ;

    input >> color ;

    std::stringstream output ;
    output << color ;


    if(successOrFail == "S"){
        EXPECT_FALSE( input.fail() ) << " Input was: " << input.str();
        EXPECT_EQ(output.str(), result) << " Input was: " << input.str();
    } else {
        EXPECT_TRUE( input.fail() ) << " Input was: " << input.str();
    }
}

void Color_Test::checkDoubleStreamingOperator(const std::vector<std::string>& p)
{
    assert(p.size()==4) ;
    RGBAColor color1 ;
    RGBAColor color2 ;

    std::stringstream input ;
    std::stringstream output ;
    std::string result = p[1] ;
    std::string successOrFail = p[2] ;

    input << p[0] ;
    input >> color1;
    input >> color2 ;
    output << color1 << " and " << color2 ;

    if(successOrFail == "S"){
        EXPECT_FALSE( input.fail() ) << " Input was: " << input.str();
        EXPECT_EQ(output.str(), result) << " Input was: " << input.str();
    } else {
        EXPECT_TRUE( input.fail() ) << " Input was: " << input.str();
    }
}

void Color_Test::checkColorDataField()
{
    Data<RGBAColor> color ;

    EXPECT_FALSE(color.read("invalidcolor"));

    EXPECT_TRUE(color.read("white"));
    EXPECT_EQ(color.getValue(), RGBAColor(1.0,1.0,1.0,1.0));

    EXPECT_TRUE(color.read("blue"));
    EXPECT_EQ(color.getValue(), RGBAColor(0.0,0.0,1.0,1.0));

    std::stringstream tmp;
    tmp << color ;
    EXPECT_TRUE(color.read(tmp.str()));
    EXPECT_EQ(color.getValue(), RGBAColor(0.0,0.0,1.0,1.0));
}

void Color_Test::checkEquality()
{
    EXPECT_EQ(RGBAColor(), RGBAColor());
    EXPECT_EQ(RGBAColor(0.0,1.0,2.0,3.0), RGBAColor(0.0,1.0,2.0,3.0));

    EXPECT_NE(RGBAColor(0.1,1.0,2.0,3.0), RGBAColor(0.0,1.0,2.0,3.0));
    EXPECT_NE(RGBAColor(0.1,1.1,2.0,3.0), RGBAColor(0.1,1.0,2.0,3.0));
    EXPECT_NE(RGBAColor(0.1,1.1,2.1,3.0), RGBAColor(0.1,1.1,2.0,3.0));
    EXPECT_NE(RGBAColor(0.1,1.1,2.1,3.1), RGBAColor(0.1,1.1,2.1,3.0));
}

TEST_F(Color_Test, checkColorDataField)
{
    this->checkColorDataField() ;
}

TEST_F(Color_Test, checkCreateFromString)
{
    this->checkCreateFromString() ;
}

TEST_F(Color_Test, checkCreateFromDouble)
{
    this->checkCreateFromString() ;
}

TEST_F(Color_Test, checkEquality)
{
    this->checkEquality() ;
}

std::vector<std::vector<std::string>> testvalues =
{
    {"    0 0 0 0","0 0 0 0", "S"},

    {"0 0 0 0","0 0 0 0", "S"},
    {"1 2 3 4","1 2 3 4", "S"},
    {"0 1 0","0 1 0 1", "S"},
    {"1 2 3","1 2 3 1", "S"},
    {"0 0 0 0 #Something","0 0 0 0", "S"},
    {"0 A 0","", "F"},
    {"A 0 0","", "F"},
    {"0 0 A","", "F"},

    {"#00000000","0 0 0 0", "S"},
    {"#FFFFFFFF","1 1 1 1", "S"},
    {"#ff00ff00","1 0 1 0", "S"},
    {"#ff00FF00","1 0 1 0", "S"},
    {"#000000","0 0 0 1", "S"},
    {"#FFFFFF","1 1 1 1", "S"},
    {"#FF00FF #AAFFFAA","1 0 1 1", "S"},
    {"#F0F #AAAAAA","1 0 1 1", "S"},
    {"#F0F0 #AAAAAA","1 0 1 0", "S"},

    {"#XXZZBBGG", "", "F"},
    {"#AAAFFFFBBDDCC","", "F"},

    {"white", "1 1 1 1", "S"},
    {"blue", "0 0 1 1", "S"},
    {"black", "0 0 0 1", "S"},
    {"white&black", "", "F"},

    {"0 0 0 0 1 1 1 1","0 0 0 0 and 1 1 1 1", "S","DOUBLE"},
    {"1 2 3 4 5 6 7 8","1 2 3 4 and 5 6 7 8", "S","DOUBLE"},
    {"1 2 3 4    5 6 7 8","1 2 3 4 and 5 6 7 8", "S","DOUBLE"},
    {"1 2 3 4   5 6 7 8","1 2 3 4 and 5 6 7 8", "S","DOUBLE"},
    {"0 0 0 1 1 1","", "F","DOUBLE"},

    {"#00ff00ff #ff00ff00","0 1 0 1 and 1 0 1 0", "S","DOUBLE"},
    {"black blue", "0 0 0 1 and 0 0 1 1", "S","DOUBLE"},
} ;

TEST_P(Color_Test, checkStreamingOperator)
{
    auto& p = GetParam();
    if(p.size()==3)
        this->checkStreamingOperator(p);
    else if(p.size()==4)
        this->checkDoubleStreamingOperator(p);
    else
        FAIL() << "There is a problem with this test.";
}

INSTANTIATE_TEST_CASE_P(checkStreamingOperator,
                        Color_Test,
                        ::testing::ValuesIn(testvalues));



