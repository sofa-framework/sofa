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
#include <sofa/testing/NumericTest.h>

#include <sofa/type/trait/Rebind.h>
using sofa::testing::NumericTest ;

using ::testing::Types;

#include <sofa/type/vector.h>
using sofa::type::vector ;

#include <sofa/type/Vec.h>
#include <type_traits>

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data ;

#include <sofa/helper/logging/CountingMessageHandler.h>
using sofa::helper::logging::MainCountingMessageHandler ;
using sofa::helper::logging::CountingMessageHandler ;
using sofa::helper::logging::MessageDispatcher ;
using sofa::helper::logging::Message ;

template<class T>
class vector_test : public NumericTest<>,
        public ::testing::WithParamInterface<std::vector<std::string>>
{
public:
    void checkVector(const std::vector<std::string>& params) ;
    void checkVectorAccessFailure() const;

    void checkRebind();
};

template<class T>
void vector_test<T>::checkVector(const std::vector<std::string>& params)
{
    std::string result = params[1];
    std::string errtype = params[2];

    vector<T> v;
    std::stringstream in(params[0]);
    std::stringstream out ;

    // capture cerr
    std::stringstream cerrbuffer;
    std::streambuf *old = std::cerr.rdbuf(cerrbuffer.rdbuf());

    v.read(in) ;
    v.write(out) ;

    EXPECT_EQ( result, out.str() ) ;

    if (errtype == "Error" || errtype == "Warning")
    {
        EXPECT_FALSE( cerrbuffer.str().empty() );
        EXPECT_TRUE(in.fail());
    }
    else
    {
        EXPECT_TRUE( cerrbuffer.str().empty() );
        EXPECT_FALSE(in.fail());
    }

    // restore cerr
    std::cerr.rdbuf( old );
}

template <class T>
void vector_test<T>::checkVectorAccessFailure() const
{
    sofa::type::vector<T> initializedVector(12);

    EXPECT_NO_THROW(initializedVector[11]);
    if constexpr (sofa::type::isEnabledVectorAccessChecking)
    {
        EXPECT_THROW(initializedVector[12], std::logic_error);
    }
    else
    {
        //initializedVector[12] leads to an undefined behavior
    }
}

template <class T>
void vector_test<T>::checkRebind()
{
    constexpr bool hasRebind = sofa::type::HasRebindTypedef<vector<T>, int>::value;
    EXPECT_TRUE(hasRebind);
    using rebinded = typename sofa::type::Rebind<vector<T>, int >::to;
    using vec_int = vector<int>;
    constexpr bool isRebindOK = std::is_same_v<rebinded, vec_int >;
    EXPECT_TRUE(isRebindOK);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// TEST THE vector<int> behavior
///
////////////////////////////////////////////////////////////////////////////////////////////////////
typedef vector_test<int> vector_test_int;
TEST_P(vector_test_int, checkReadWriteBehavior)
{
    this->checkVector(GetParam()) ;
}
TEST_F(vector_test_int, checkVectorAccessFailure)
{
    this->checkVectorAccessFailure() ;
}

std::vector<std::vector<std::string>> intvalues={
    /// First test valid values
    {"0 1 2 3 4 5 6", "0 1 2 3 4 5 6", "None"},
    {"0 -1 2 -3 4 5 6", "0 -1 2 -3 4 5 6", "None"},
    {"100", "100", "None"},
    {"-100", "-100", "None"},
    {"", "", "None"},

    /// The test the A-B range notation
    {"10-15 21", "10 11 12 13 14 15 21", "None"},
    {"15-10 21", "15 14 13 12 11 10 21", "None"},
    {"10-15 -2 21", "10 11 12 13 14 15 -2 21", "None"},
    {"15-10 -2 21", "15 14 13 12 11 10 -2 21", "None"},

    /// Test the A-B-INC range notation
    {"10-16-2 21", "10 12 14 16 21", "None"},

    /// The test the A-B negative range notation
    {"-5-0", "-5 -4 -3 -2 -1 0", "None"},
    {"-5-5", "-5 -4 -3 -2 -1 0 1 2 3 4 5", "None"},
    {"5--5", "5 4 3 2 1 0 -1 -2 -3 -4 -5", "None"},
    {"-10--5", "-10 -9 -8 -7 -6 -5", "None"},
    {"-5--10", "-5 -6 -7 -8 -9 -10", "None"},

    /// Test the A-B-INC range notation
    {"10-16-2 21", "10 12 14 16 21", "None"},

    /// Test the A-B-INC negative range notation
    {"10-16--2 21", "16 14 12 10 21", "None"},
    {"-10--16--2 21", "-10 -12 -14 -16 21", "None"},

    /// Now we test correct handling of problematic input
    {"5 6 - 10 0", "5 6 0 10 0", "Warning"},
    {"5 6---10 0", "5 0 0", "Warning"},
    {"zero 1 2 trois quatre cinq 6", "0 1 2 0 0 0 6", "Warning"},
    {"3.14 4.15 5.16", "0 0 0", "Warning"}
};
INSTANTIATE_TEST_SUITE_P(checkReadWriteBehavior,
                        vector_test_int,
                        ::testing::ValuesIn(intvalues));

TEST_F(vector_test_int, checkRebind)
{
    this->checkRebind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// TEST THE vector<int> behavior
///
////////////////////////////////////////////////////////////////////////////////////////////////////
typedef vector_test<unsigned int> vector_test_unsigned_int;
TEST_P(vector_test_unsigned_int, checkReadWriteBehavior)
{
    this->checkVector(GetParam()) ;
}
TEST_F(vector_test_unsigned_int, checkVectorAccessFailure)
{
    this->checkVectorAccessFailure() ;
}

std::vector<std::vector<std::string>> uintvalues={
    /// First test valid values
    {"0 1 2 3 4 5 6", "0 1 2 3 4 5 6", "None"},
    {"100", "100", "None"},
    {"", "", "None"},

    /// Test the A-B range notation
    {"10-15 21", "10 11 12 13 14 15 21", "None"},
    {"15-10 21", "15 14 13 12 11 10 21", "None"},

    /// Test the A-B range notation
    {"10-16-2 21", "10 12 14 16 21", "None"},
    {"10-16--2 21", "16 14 12 10 21", "None"},

    /// Test the A-B negative range notation
    {"-5-5", "0 1 2 3 4 5", "Warning"},
    {"0--5", "0", "Warning"},
    {"5--5", "5 4 3 2 1 0", "Warning"},
    {"-10--5", "0", "Warning"},
    {"-5--10", "0", "Warning"},

    /// Test correct handling of problematic input
    {"-5", "0", "Warning"},
    {"0 -1 2 -3 4 5 6", "0 0 2 0 4 5 6", "Warning"},
    {"-100", "0", "Warning"},
    {"5 6 - 10 0", "5 6 0 10 0", "Warning"},
    {"zero 1 2 trois quatre cinq 6", "0 1 2 0 0 0 6", "Warning"},
    {"3.14 4.15 5.16", "0 0 0", "Warning"},
    {"5 6---10 0", "5 0 0", "Warning"}
};
INSTANTIATE_TEST_SUITE_P(checkReadWriteBehavior,
                        vector_test_unsigned_int,
                        ::testing::ValuesIn(uintvalues));

TEST_F(vector_test_unsigned_int, checkRebind)
{
    this->checkRebind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// BENCHMARK THE vector<int> behavior
///
////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
class vector_benchmark : public NumericTest<>,
        public ::testing::WithParamInterface<std::vector<std::string>>
{
public:
    void benchmark(const std::vector<std::string>& params) ;
};


template<class T>
void vector_benchmark<T>::benchmark(const std::vector<std::string>& params)
{
    const int loop1 = atoi(params[0].c_str());
    const int loop2 = atoi(params[1].c_str());
    std::stringstream tmp;
    for(int i=0;i<loop1;i++)
    {
        tmp << i << " ";
    }

    if(loop2==0)
        return ;

    sofa::type::vector<T> v;
    for(int i=0;i<loop2;i++){
        std::stringstream ntmp;
        ntmp << tmp.str() ;
        v.read(ntmp);
    }
    EXPECT_EQ((int)v.size(), loop1) ;
}

std::vector<std::vector<std::string>> benchvalues =
    {{"10","0"},
     {"10","10000"},
     {"10","100000"},
     {"10","1000000"},
     {"1000","0"},
     {"1000","1000"},
     {"1000","10000"},
     {"1000","100000"},
     {"100000","0"},
     {"100000","1000"}
    } ;

typedef vector_benchmark<unsigned int> vector_benchmark_unsigned_int;
TEST_P(vector_benchmark_unsigned_int, benchmark)
{
   this->benchmark(GetParam());
}

INSTANTIATE_TEST_SUITE_P(benchmark,
                        vector_benchmark_unsigned_int,
                        ::testing::ValuesIn(benchvalues));

typedef vector_benchmark<int> vector_benchmark_int;
TEST_P(vector_benchmark_int, benchmark)
{
   this->benchmark(GetParam());
}

INSTANTIATE_TEST_SUITE_P(benchmark,
                        vector_benchmark_int,
                        ::testing::ValuesIn(benchvalues));
