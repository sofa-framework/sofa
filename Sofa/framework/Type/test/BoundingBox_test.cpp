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
#include <sofa/type/BoundingBox.h>
#include <gtest/gtest.h>

namespace sofa
{

using sofa::type::BoundingBox;
using sofa::type::Vec3;

TEST(BoundingBoxTest, DefaultConstructor)
{
    static constexpr BoundingBox bbox;
    EXPECT_TRUE(bbox.isNegligible()); // Default neutral box should be negligible
}

TEST(BoundingBoxTest, ConstructorWithEndpoints) {
    static constexpr Vec3 minVec(0.0, 0.0, 0.0);
    static constexpr Vec3 maxVec(1.0, 1.0, 1.0);
    static constexpr BoundingBox bbox(minVec, maxVec);

    EXPECT_EQ(bbox.minBBox(), minVec);
    EXPECT_EQ(bbox.maxBBox(), maxVec);
}

TEST(BoundingBoxTest, ConstructorWithLimits) {
    static constexpr BoundingBox bbox(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    EXPECT_EQ(bbox.minBBox(), Vec3(0.0, 0.0, 0.0));
    EXPECT_EQ(bbox.maxBBox(), Vec3(1.0, 1.0, 1.0));
}

TEST(BoundingBoxTest, NeutralBoundingBox) {
    static constexpr auto neutral = BoundingBox::neutral_bbox();
    EXPECT_FALSE(neutral.isValid()); // Neutral bbox is invalid
}

TEST(BoundingBoxTest, Invalidate) {
    BoundingBox bbox(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
    bbox.invalidate();
    EXPECT_FALSE(bbox.isValid());
}

TEST(BoundingBoxTest, IsFlat) {
    static constexpr BoundingBox flatBBox(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0));
    EXPECT_TRUE(flatBBox.isFlat());

    static constexpr BoundingBox nonFlatBBox(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
    EXPECT_FALSE(nonFlatBBox.isFlat());
}

TEST(BoundingBoxTest, ContainsPoint) {
    static constexpr BoundingBox bbox(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
    static constexpr Vec3 pointInside(0.5, 0.5, 0.5);
    static constexpr Vec3 pointOutside(1.5, 1.5, 1.5);

    EXPECT_TRUE(bbox.contains(pointInside));
    EXPECT_FALSE(bbox.contains(pointOutside));
}

TEST(BoundingBoxTest, ContainsBoundingBox) {
    static constexpr BoundingBox bbox(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 2.0, 2.0));
    static constexpr BoundingBox containedBBox(Vec3(0.5, 0.5, 0.5), Vec3(1.5, 1.5, 1.5));
    static constexpr BoundingBox outsideBBox(Vec3(2.5, 2.5, 2.5), Vec3(3.0, 3.0, 3.0));

    EXPECT_TRUE(bbox.contains(containedBBox));
    EXPECT_FALSE(bbox.contains(outsideBBox));
}

TEST(BoundingBoxTest, Intersection) {
    static constexpr BoundingBox bbox1(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 2.0, 2.0));
    static constexpr BoundingBox bbox2(Vec3(1.0, 1.0, 1.0), Vec3(3.0, 3.0, 3.0));
    static constexpr BoundingBox expectedIntersection(Vec3(1.0, 1.0, 1.0), Vec3(2.0, 2.0, 2.0));

    EXPECT_TRUE(bbox1.intersect(bbox2));
    EXPECT_EQ(bbox1.getIntersection(bbox2), expectedIntersection);
}

TEST(BoundingBoxTest, Inflate) {
    BoundingBox bbox(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
    bbox.inflate(1.0);

    EXPECT_EQ(bbox.minBBox(), Vec3(-1.0, -1.0, -1.0));
    EXPECT_EQ(bbox.maxBBox(), Vec3(2.0, 2.0, 2.0));
}

TEST(BoundingBoxTest, IncludePoint) {
    BoundingBox bbox(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
    static constexpr Vec3 point(2.0, 2.0, 2.0);
    bbox.include(point);

    EXPECT_EQ(bbox.maxBBox(), point);
}

TEST(BoundingBoxTest, IncludeBoundingBox) {
    BoundingBox bbox(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
    static constexpr BoundingBox other(Vec3(-1.0, -1.0, -1.0), Vec3(2.0, 2.0, 2.0));
    bbox.include(other);

    EXPECT_EQ(bbox.minBBox(), other.minBBox());
    EXPECT_EQ(bbox.maxBBox(), other.maxBBox());
}

}
