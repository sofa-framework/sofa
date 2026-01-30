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

#include <sofa/defaulttype/VecTypes.h>
#include <gtest/gtest.h>

template < sofa::Size N, typename ValueType>
void defaultConstructor()
{
    const sofa::type::Vec<N, ValueType> vec;
    for (const auto& value : vec)
    {
        EXPECT_EQ(value, ValueType());
    }
}


TEST(VecTest, DefaultConstructor)
{
    defaultConstructor<1, unsigned int>();
    defaultConstructor<2, unsigned int>();
    defaultConstructor<3, unsigned int>();
    defaultConstructor<6, unsigned int>();

    defaultConstructor<1, int>();
    defaultConstructor<2, int>();
    defaultConstructor<3, int>();
    defaultConstructor<6, int>();

    defaultConstructor<1, float>();
    defaultConstructor<2, float>();
    defaultConstructor<3, float>();
    defaultConstructor<6, float>();

    defaultConstructor<1, double>();
    defaultConstructor<2, double>();
    defaultConstructor<3, double>();
    defaultConstructor<6, double>();
}

TEST(VecTest, StructuredBindings)
{
    constexpr sofa::type::Vec3 vec { 1.0, 2.0, 3.0 };
    const auto& [a, b, c] = vec;
    EXPECT_EQ(a, 1.);
    EXPECT_EQ(b, 2.);
    EXPECT_EQ(c, 3.);
}

TEST(VecTest, DeductionGuide)
{
    constexpr sofa::type::Vec vec { 1.0_sreal, 2.0_sreal, 3.0_sreal };
    static_assert(std::is_same_v<decltype(vec)::value_type, SReal>);

    constexpr sofa::type::VecNoInit vec2 { 1.0_sreal, 2.0_sreal, 3.0_sreal };
    static_assert(std::is_same_v<decltype(vec2)::value_type, SReal>);
}

TEST(VecTest, Equality)
{
    constexpr sofa::type::Vec<3,SReal> vecf1 { 1.0_sreal, 2.0_sreal, 3.0_sreal };
    constexpr sofa::type::Vec<3,SReal> vecf2 { 1.0_sreal, 2.0_sreal, 3.0_sreal };
    constexpr sofa::type::Vec<3,SReal> vecf3 { 1.0_sreal, 2.00001_sreal, 3.0_sreal };
    
    constexpr sofa::type::Vec<3,int> veci1 { 2, 4, 7 };
    constexpr sofa::type::Vec<3,int> veci2 { 2, 4, 7 };
    constexpr sofa::type::Vec<3,int> veci3 { 2, 4, 6 };
    
    EXPECT_TRUE(vecf1 == vecf2);
    EXPECT_FALSE(vecf1 != vecf2);
    EXPECT_FALSE(vecf1 == vecf3);
    EXPECT_TRUE(vecf1 != vecf3);
    
    EXPECT_TRUE(veci1 == veci2);
    EXPECT_FALSE(veci1 != veci2);
    EXPECT_FALSE(veci1 == veci3);
    EXPECT_TRUE(veci1 != veci3);
}

TEST(VecTest, toVecN)
{
    // test toVecN<x,y> (to a smaller vec)
    constexpr sofa::type::Vec<5,int> vec5i  {1, 2, 3, 4, 5};
    constexpr sofa::type::Vec<2,float> vec2f = sofa::type::toVecN<2, float>(vec5i);
    
    // static_assert(vec2f == sofa::type::Vec<2, float>{1.0f, 2.0f}); // possible only in c++23 (needs std::abs to be constevaluable)
    constexpr sofa::type::Vec<2,float> vec2f_ref {1.0f, 2.0f};
    EXPECT_TRUE(vec2f == vec2f_ref);
    
    // test toVecN<x,y> (to a bigger vec)
    constexpr sofa::type::Vec<6, unsigned long> vec6ul = sofa::type::toVecN<6, unsigned long>(vec5i);
    constexpr sofa::type::Vec<6, unsigned long> vec6ul_ref {1ul, 2ul, 3ul, 4ul, 5ul, 0ul};
    EXPECT_TRUE(vec6ul == vec6ul_ref);
    
    // test toVecN<OtherVec>
    using WhateverVecItIs = sofa::type::Vec<4, double>;
    constexpr WhateverVecItIs othervec = sofa::type::toVecN<WhateverVecItIs>(vec5i);
    constexpr WhateverVecItIs othervec_ref {1.0, 2.0, 3.0, 4.0};
    EXPECT_TRUE(othervec == othervec_ref);
    
    // test toVecN<OtherVec> without knowing what type is the incoming vec is
    using IDontKnowWhatVecItIs = sofa::type::Vec<7, unsigned int>;
    constexpr auto incomingvec = IDontKnowWhatVecItIs{1u, 2u, 3u, 4u, 5u, 6u, 7u};
    constexpr auto autotypevec = sofa::type::toVecN<decltype(incomingvec)>(vec5i);
    constexpr IDontKnowWhatVecItIs autotypevec_ref {1u, 2u, 3u, 4u, 5u, 0u, 0u};
    
    EXPECT_TRUE(autotypevec == autotypevec_ref);

    // test toVec3
    constexpr sofa::type::Vec3 vec3r = sofa::type::toVec3(vec5i);
    constexpr sofa::type::Vec3 vec3r_ref {1.0_sreal, 2.0_sreal, 3.0_sreal};
    EXPECT_TRUE(vec3r == vec3r_ref);
    
    
    // test toVecN<x,y> (to a bigger vec with filler)
    constexpr sofa::type::Vec<9, long double> vec9ld = sofa::type::toVecN<9, long double, 5, int>(vec5i, 42.0L);
    constexpr sofa::type::Vec<9, long double> vec9ld_ref {1.0L, 2.0L, 3.0L, 4.0L, 5.0L, 42.0L, 42.0L, 42.0L, 42.0L};
    EXPECT_TRUE(vec9ld == vec9ld_ref);
    
}

TEST(VecTest, NOINITConstructor)
{
    sofa::type::Vec<3, double> v(sofa::type::NOINIT);
    [[maybe_unused]] auto x = v[0];
    EXPECT_NO_THROW(v[1] = 5.0);
}

TEST(VecTest, singleElementConstructor)
{
    sofa::type::Vec<1, double> v(5.0);
    EXPECT_EQ(v[0], 5.0);
}

TEST(VecTest, multiElementConstructor)
{
    sofa::type::Vec<3, double> v(1.0, 2.0, 3.0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 2.0);
    EXPECT_EQ(v[2], 3.0);
}

TEST(VecTest, copyConstructor)
{
    sofa::type::Vec<3, double> v1(1.0, 2.0, 3.0);
    sofa::type::Vec<3, double> v2(v1);
    EXPECT_EQ(v2[0], 1.0);
    EXPECT_EQ(v2[1], 2.0);
    EXPECT_EQ(v2[2], 3.0);
}

TEST(VecTest, moveConstructor)
{
    sofa::type::Vec<3, double> v1(1.0, 2.0, 3.0);
    sofa::type::Vec<3, double> v2(std::move(v1));
    EXPECT_EQ(v2[0], 1.0);
    EXPECT_EQ(v2[1], 2.0);
    EXPECT_EQ(v2[2], 3.0);
}

TEST(VecTest, constructorFromFixedArray)
{
    sofa::type::fixed_array<double, 3> arr = {1.0, 2.0, 3.0};
    sofa::type::Vec<3, double> v(arr);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 2.0);
    EXPECT_EQ(v[2], 3.0);
}

TEST(VecTest, constructorFromDifferentSize)
{
    sofa::type::Vec<3, double> v1(1.0, 2.0, 3.0);
    sofa::type::Vec<5, double> v2(v1);
    EXPECT_EQ(v2[0], 1.0);
    EXPECT_EQ(v2[1], 2.0);
    EXPECT_EQ(v2[2], 3.0);
    EXPECT_EQ(v2[3], 0.0);
    EXPECT_EQ(v2[4], 0.0);
}

TEST(VecTest, setMethod)
{
    sofa::type::Vec<3, double> v;
    v.set(1.0, 2.0, 3.0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 2.0);
    EXPECT_EQ(v[2], 3.0);
}

TEST(VecTest, setFromOtherVec)
{
    sofa::type::Vec<3, double> v1(1.0, 2.0, 3.0);
    sofa::type::Vec<5, double> v;
    v.set(v1);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 2.0);
    EXPECT_EQ(v[2], 3.0);
}

TEST(VecTest, elementAccessOperatorParen)
{
    sofa::type::Vec<3, double> v(1.0, 2.0, 3.0);
    EXPECT_EQ(v(0), 1.0);
    EXPECT_EQ(v(1), 2.0);
    EXPECT_EQ(v(2), 3.0);
}

TEST(VecTest, accessorsXYZW)
{
    sofa::type::Vec<4, double> v(1.0, 2.0, 3.0, 4.0);
    EXPECT_EQ(v.x(), 1.0);
    EXPECT_EQ(v.y(), 2.0);
    EXPECT_EQ(v.z(), 3.0);
    EXPECT_EQ(v.w(), 4.0);

    const sofa::type::Vec<4, double> cv(1.0, 2.0, 3.0, 4.0);
    EXPECT_EQ(cv.x(), 1.0);
    EXPECT_EQ(cv.y(), 2.0);
    EXPECT_EQ(cv.z(), 3.0);
    EXPECT_EQ(cv.w(), 4.0);
}

TEST(VecTest, pointerAccess)
{
    sofa::type::Vec<3, double> v(1.0, 2.0, 3.0);
    const double* p = v.ptr();
    EXPECT_EQ(p[0], 1.0);
    EXPECT_EQ(p[1], 2.0);
    EXPECT_EQ(p[2], 3.0);

    double* np = v.ptr();
    np[0] = 10.0;
    EXPECT_EQ(v[0], 10.0);
}

TEST(VecTest, dataAccess)
{
    sofa::type::Vec<3, double> v(1.0, 2.0, 3.0);
    const double* p = v.data();
    EXPECT_EQ(p[0], 1.0);
    EXPECT_EQ(p[1], 2.0);
    EXPECT_EQ(p[2], 3.0);
}

TEST(VecTest, fill)
{
    sofa::type::Vec<3, double> v;
    v.fill(5.0);
    EXPECT_EQ(v[0], 5.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 5.0);
}

TEST(VecTest, assign)
{
    sofa::type::Vec<3, double> v;
    v.assign(7.0);
    EXPECT_EQ(v[0], 7.0);
    EXPECT_EQ(v[1], 7.0);
    EXPECT_EQ(v[2], 7.0);
}

TEST(VecTest, clear)
{
    sofa::type::Vec<3, double> v(5.0, 6.0, 7.0);
    v.clear();
    EXPECT_EQ(v[0], 0.0);
    EXPECT_EQ(v[1], 0.0);
    EXPECT_EQ(v[2], 0.0);
}

TEST(VecTest, addition)
{
    sofa::type::Vec<3, double> v1(1.0, 2.0, 3.0);
    sofa::type::Vec<3, double> v2(4.0, 5.0, 6.0);
    auto v3 = v1 + v2;
    EXPECT_EQ(v3[0], 5.0);
    EXPECT_EQ(v3[1], 7.0);
    EXPECT_EQ(v3[2], 9.0);
}

TEST(VecTest, additionAssignment)
{
    sofa::type::Vec<3, double> v1(1.0, 2.0, 3.0);
    sofa::type::Vec<3, double> v2(4.0, 5.0, 6.0);
    v1 += v2;
    EXPECT_EQ(v1[0], 5.0);
    EXPECT_EQ(v1[1], 7.0);
    EXPECT_EQ(v1[2], 9.0);
}

TEST(VecTest, subtraction)
{
    sofa::type::Vec<3, double> v1(5.0, 7.0, 9.0);
    sofa::type::Vec<3, double> v2(4.0, 5.0, 6.0);
    auto v3 = v1 - v2;
    EXPECT_EQ(v3[0], 1.0);
    EXPECT_EQ(v3[1], 2.0);
    EXPECT_EQ(v3[2], 3.0);
}

TEST(VecTest, subtractionAssignment)
{
    sofa::type::Vec<3, double> v1(5.0, 7.0, 9.0);
    sofa::type::Vec<3, double> v2(4.0, 5.0, 6.0);
    v1 -= v2;
    EXPECT_EQ(v1[0], 1.0);
    EXPECT_EQ(v1[1], 2.0);
    EXPECT_EQ(v1[2], 3.0);
}

TEST(VecTest, scalarMultiplication)
{
    sofa::type::Vec<3, double> v(1.0, 2.0, 3.0);
    auto r = v * 2.0;
    EXPECT_EQ(r[0], 2.0);
    EXPECT_EQ(r[1], 4.0);
    EXPECT_EQ(r[2], 6.0);
}

TEST(VecTest, scalarMultiplicationAssignment)
{
    sofa::type::Vec<3, double> v(1.0, 2.0, 3.0);
    v *= 2.0;
    EXPECT_EQ(v[0], 2.0);
    EXPECT_EQ(v[1], 4.0);
    EXPECT_EQ(v[2], 6.0);
}

TEST(VecTest, scalarDivision)
{
    sofa::type::Vec<3, double> v(2.0, 4.0, 6.0);
    auto r = v / 2.0;
    EXPECT_EQ(r[0], 1.0);
    EXPECT_EQ(r[1], 2.0);
    EXPECT_EQ(r[2], 3.0);
}

TEST(VecTest, scalarDivisionAssignment)
{
    sofa::type::Vec<3, double> v(2.0, 4.0, 6.0);
    v /= 2.0;
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 2.0);
    EXPECT_EQ(v[2], 3.0);
}

TEST(VecTest, dotProduct)
{
    sofa::type::Vec<3, double> v1(1.0, 2.0, 3.0);
    sofa::type::Vec<3, double> v2(4.0, 5.0, 6.0);
    EXPECT_EQ(v1 * v2, 32.0);
}

TEST(VecTest, linearProduct)
{
    sofa::type::Vec<3, double> v1(2.0, 4.0, 6.0);
    sofa::type::Vec<3, double> v2(3.0, 5.0, 7.0);
    auto r = v1.linearProduct(v2);
    EXPECT_EQ(r[0], 6.0);
    EXPECT_EQ(r[1], 20.0);
    EXPECT_EQ(r[2], 42.0);
}

TEST(VecTest, linearDivision)
{
    sofa::type::Vec<3, double> v1(6.0, 10.0, 14.0);
    sofa::type::Vec<3, double> v2(2.0, 5.0, 7.0);
    auto r = v1.linearDivision(v2);
    EXPECT_EQ(r[0], 3.0);
    EXPECT_EQ(r[1], 2.0);
    EXPECT_EQ(r[2], 2.0);
}

TEST(VecTest, norm2)
{
    sofa::type::Vec<3, double> v(3.0, 4.0, 0.0);
    EXPECT_EQ(v.norm2(), 25.0);
}

TEST(VecTest, norm)
{
    sofa::type::Vec<3, double> v(3.0, 4.0, 0.0);
    EXPECT_NEAR(v.norm(), 5.0, 1e-10);
}

TEST(VecTest, normalize)
{
    sofa::type::Vec<3, double> v(3.0, 4.0, 0.0);
    bool result = v.normalize();
    EXPECT_TRUE(result);
    EXPECT_NEAR(v[0], 0.6, 1e-6);
    EXPECT_NEAR(v[1], 0.8, 1e-6);
    EXPECT_NEAR(v.norm(), 1.0, 1e-10);
}

TEST(VecTest, normalizeTooSmall)
{
    sofa::type::Vec<3, double> v1(1e-6, 2e-6, 3e-6);
    sofa::type::Vec<3, double> v2 = v1;
    bool result = v1.normalize(); // threshold is really too small by default
    EXPECT_TRUE(result);
    
    bool result2 = v2.normalize(1e-5); // more reasonable threshold
    EXPECT_FALSE(result2);
}

TEST(VecTest, normalized)
{
    sofa::type::Vec<3, double> v(3.0, 4.0, 0.0);
    auto n = v.normalized();
    EXPECT_NEAR(n[0], 0.6, 1e-6);
    EXPECT_NEAR(n[1], 0.8, 1e-6);
}

TEST(VecTest, isNormalized)
{
    sofa::type::Vec<3, double> v(1.0, 0.0, 0.0);
    EXPECT_TRUE(v.isNormalized());
    
    sofa::type::Vec<3, double> v2(2.0, 0.0, 0.0);
    EXPECT_FALSE(v2.isNormalized());
}

TEST(VecTest, normalizeWithFailsafe)
{
    sofa::type::Vec<3, double> v1(1e-8, 2e-8, 3e-8);
    sofa::type::Vec<3, double> v2 = v1;
    sofa::type::Vec<3, double> failsafe(7.0, 8.0, 9.0);
    v1.normalize(failsafe); // threshold is really too small by default
    EXPECT_NE(v1[0], 7.0);
    EXPECT_NE(v1[1], 8.0);
    EXPECT_NE(v1[2], 9.0);
    
    v2.normalize(failsafe, 1e-5); // more reasonable threshold
    EXPECT_EQ(v2[0], 7.0);
    EXPECT_EQ(v2[1], 8.0);
    EXPECT_EQ(v2[2], 9.0);
}

TEST(VecTest, crossProduct)
{
    sofa::type::Vec<3, double> a(1.0, 0.0, 0.0);
    sofa::type::Vec<3, double> b(0.0, 1.0, 0.0);
    auto c = sofa::type::cross(a, b);
    EXPECT_EQ(c[0], 0.0);
    EXPECT_EQ(c[1], 0.0);
    EXPECT_EQ(c[2], 1.0);
}

TEST(VecTest, sum)
{
    sofa::type::Vec<3, double> v(1.0, 2.0, 3.0);
    EXPECT_EQ(v.sum(), 6.0);
}

TEST(VecTest, equalityWithThreshold)
{
    sofa::type::Vec<3, double> v1(1.0000001, 2.0000001, 3.0000001);
    sofa::type::Vec<3, double> v2(1.0, 2.0, 3.0);
    EXPECT_TRUE(v1 == v2);
}

TEST(VecTest, inequalityWithThreshold)
{
    sofa::type::Vec<3, double> v1(1.1, 2.1, 3.1);
    sofa::type::Vec<3, double> v2(1.0, 2.0, 3.0);
    EXPECT_TRUE(v1 != v2);
}

TEST(VecTest, iterators)
{
    sofa::type::Vec<3, double> v(1.0, 2.0, 3.0);
    auto it = v.begin();
    EXPECT_EQ(*it, 1.0);
    ++it;
    EXPECT_EQ(*it, 2.0);
    ++it;
    EXPECT_EQ(*it, 3.0);
}

TEST(VecTest, constIterators)
{
    const sofa::type::Vec<3, double> v(1.0, 2.0, 3.0);
    auto it = v.begin();
    EXPECT_EQ(*it, 1.0);
    ++it;
    EXPECT_EQ(*it, 2.0);
    ++it;
    EXPECT_EQ(*it, 3.0);
}

TEST(VecTest, front)
{
    sofa::type::Vec<3, double> v(5.0, 6.0, 7.0);
    EXPECT_EQ(v.front(), 5.0);
}

TEST(VecTest, back)
{
    sofa::type::Vec<3, double> v(5.0, 6.0, 7.0);
    EXPECT_EQ(v.back(), 7.0);
}

TEST(VecTest, getSubVector)
{
    sofa::type::Vec<5, double> v(1.0, 2.0, 3.0, 4.0, 5.0);
    sofa::type::Vec<2, double> sub;
    v.getsub(1, sub);
    EXPECT_EQ(sub[0], 2.0);
    EXPECT_EQ(sub[1], 3.0);
}

TEST(VecTest, getSubScalar)
{
    sofa::type::Vec<3, double> v(5.0, 6.0, 7.0);
    double scalar;
    v.getsub(1, scalar);
    EXPECT_EQ(scalar, 6.0);
}

TEST(VecTest, staticSize)
{
    constexpr sofa::Size s = sofa::type::Vec<3, double>::static_size;
    EXPECT_EQ(s, 3u);
}

TEST(VecTest, sizeMethod)
{
    constexpr sofa::Size s = sofa::type::Vec<5, double>::size();
    EXPECT_EQ(s, 5);
}
