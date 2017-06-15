/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <sofa/helper/SVector.h>
#include <gtest/gtest.h>
#include <stdlib.h>

using sofa::helper::SVector;


/// testing SVector::read/write
/// (other vector functions should be tested in helper::vector)
template<class T>
struct SVector_test : public ::testing::Test
{
    SVector<T> m_vec; ///< tested SVector

    /// reading directly from a string
    void read(const std::string& s)
    {
        std::istringstream ss( s );
        ss >> m_vec;
    }

    /// regular way to read for an istream
    void read(std::istream& s)
    {
        m_vec.read( s );
    }

    /// writing in a local stringstream
    std::stringstream& write()
    {
        m_ss.str(""); // start from scratch
        m_vec.write(m_ss);
        return m_ss;
    }

private:
    std::stringstream m_ss;

};




typedef SVector_test<int> SVector_test_int;
TEST_F(SVector_test_int, int_simple)
{
    // simple
    read( "[0,1,2]" );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
        EXPECT_EQ(m_vec[i],i);

    read( write() );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
        EXPECT_EQ(m_vec[i],i);

}

TEST_F(SVector_test_int, int_extraspace)
{
    // with spaces
    read( "  [  0  ,  1  ,  2  ]  " );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
        EXPECT_EQ(m_vec[i],i);


    read( write() );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
        EXPECT_EQ(m_vec[i],i);
}



typedef SVector_test<std::string> SVector_test_string;
TEST_F(SVector_test_string, string_simple)
{
    // simple
    read( "['string0','string1','string2']" );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
    {
        std::stringstream ss;
        ss << "string" << i;
        EXPECT_STREQ(m_vec[i].c_str(),ss.str().c_str());
    }


    read( write() );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
    {
        std::stringstream ss;
        ss << "string" << i;
        EXPECT_STREQ(m_vec[i].c_str(),ss.str().c_str());
    }
}


TEST_F(SVector_test_string, string_extraspace)
{
    // with extra spaces
    read( "   [     'string0'   ,   'string1'   ,   'string2'   ]    " );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
    {
        std::stringstream ss;
        ss << "string" << i;
        EXPECT_STREQ(m_vec[i].c_str(),ss.str().c_str());
    }


    read( write() );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
    {
        std::stringstream ss;
        ss << "string" << i;
        EXPECT_STREQ(m_vec[i].c_str(),ss.str().c_str());
    }
}


TEST_F(SVector_test_string, string_simplespace)
{
    // simple with strings containing spaces
    read( "[' string 0',' string 1',' string 2']" );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
    {
        std::stringstream ss;
        ss << " string " << i;
        EXPECT_STREQ(m_vec[i].c_str(),ss.str().c_str());
    }

    read( write() );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
    {
        std::stringstream ss;
        ss << " string " << i;
        EXPECT_STREQ(m_vec[i].c_str(),ss.str().c_str());
    }
}



TEST_F(SVector_test_string, string_spaceextraspace)
{
    // with extra spaces and with strings containing spaces
    read( "   [     'string 0 '  ,    'string 1 '  ,   'string 2 '  ]    " );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
    {
        std::stringstream ss;
        ss << "string " << i <<" ";
        EXPECT_STREQ(m_vec[i].c_str(),ss.str().c_str());
    }

    read( write() );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
    {
        std::stringstream ss;
        ss << "string " << i <<" ";
        EXPECT_STREQ(m_vec[i].c_str(),ss.str().c_str());
    }


    read( "   [     ' string 0 '  ,    ' string 1 '  ,   ' string 2 '  ]    " );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
    {
        std::stringstream ss;
        ss << " string " << i <<" ";
        EXPECT_STREQ(m_vec[i].c_str(),ss.str().c_str());
    }

    read( write() );
    ASSERT_EQ(m_vec.size(),3u);
    for(int i=0;i<3;++i)
    {
        std::stringstream ss;
        ss << " string " << i <<" ";
        EXPECT_STREQ(m_vec[i].c_str(),ss.str().c_str());
    }
}

TEST_F(SVector_test_string, string_empty)
{
    read( "[]" );
    ASSERT_EQ(m_vec.size(),0u);

    read( write() );
    ASSERT_EQ(m_vec.size(),0u);



    read( "[\"\"]" );
    ASSERT_EQ(m_vec.size(),1u);
    EXPECT_STREQ(m_vec[0].c_str(),"");

    read( write() );
    ASSERT_EQ(m_vec.size(),1u);
    EXPECT_STREQ(m_vec[0].c_str(),"");



    read( "['',\"a\"]" );
    ASSERT_EQ(m_vec.size(),2u);
    EXPECT_STREQ(m_vec[0].c_str(),"");
    EXPECT_STREQ(m_vec[1].c_str(),"a");

    read( write() );
    ASSERT_EQ(m_vec.size(),2u);
    EXPECT_STREQ(m_vec[0].c_str(),"");
    EXPECT_STREQ(m_vec[1].c_str(),"a");



    read( "['a','']" );
    ASSERT_EQ(m_vec.size(),2u);
    EXPECT_STREQ(m_vec[0].c_str(),"a");
    EXPECT_STREQ(m_vec[1].c_str(),"");

    read( write() );
    ASSERT_EQ(m_vec.size(),2u);
    EXPECT_STREQ(m_vec[0].c_str(),"a");
    EXPECT_STREQ(m_vec[1].c_str(),"");
}
