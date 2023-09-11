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

#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/CollisionModel.h>
#include <gtest/gtest.h>

namespace sofa::core::collision
{
/// Empty class inheriting from the abstract class NarrowPhaseDetection
class DummyNarrowPhaseDetection : public NarrowPhaseDetection
{
public:
    void addCollisionPair(const std::pair<core::CollisionModel *, core::CollisionModel *> &/*cmPair*/) override
    {}
};

class DummyDetectionOutputVector : public DetectionOutputVector
{
public:
    void clear() override { m_size = 0; }
    unsigned int size() const override { return m_size;}
    DummyDetectionOutputVector(unsigned int size, bool* isDestroyed) : m_size(size), m_isDestroyed(isDestroyed)
    {
        if (m_isDestroyed)
        {
            *m_isDestroyed = false;
        }
    }

    ~DummyDetectionOutputVector() override
    {
        if (m_isDestroyed)
        {
            *m_isDestroyed = true;
        }
    }

    /// Const iterator to iterate the detection pairs
    virtual type::Vec3 getFirstPosition(unsigned /*idx*/) override
    {
        return type::Vec3();
    }

    /// Const iterator end to iterate the detection pairs
    virtual type::Vec3 getSecondPosition(unsigned /*idx*/) override
    {
        return type::Vec3();
    }



private:
    unsigned int m_size { 0 };
    bool* m_isDestroyed {nullptr};
    sofa::type::vector<DetectionOutput> m_empty;

};
} //namespace sofa::core::collision

namespace sofa::core
{
class DummyCollisionModel : public CollisionModel
{
public:
    void computeBoundingTree(int /*maxDepth*/) override {}
};
} //namespace sofa::collision


namespace sofa
{
using sofa::core::objectmodel::New;
TEST(NarrowPhaseDetection_test, DetectionOutputMap)
{
    const auto narrowPhaseDetection = New<sofa::core::collision::DummyNarrowPhaseDetection>();

    //Generate a collection of dummy collision models
    sofa::type::vector<core::DummyCollisionModel::SPtr> collisionModels;
    for (unsigned int i = 0; i < 10; ++i)
    {
        collisionModels.push_back(New<core::DummyCollisionModel>());
    }

    const auto& outputMap = narrowPhaseDetection->getDetectionOutputs();

    // After creation of the component, the output map is empty
    EXPECT_TRUE(outputMap.empty());

    // beginNarrowPhase does nothing on an empty map
    EXPECT_NO_THROW(narrowPhaseDetection->beginNarrowPhase());

    // Create a detection output for every combination of pair of collision models
    for (auto a : collisionModels)
    {
        for (auto b : collisionModels)
        {
            narrowPhaseDetection->getDetectionOutputs(a.get(), b.get());
        }
    }

    // The map is no longer empty
    EXPECT_FALSE(outputMap.empty());
    // The number of combination is n^2, the map has this size
    EXPECT_EQ(outputMap.size(), 10*10);

    // beginNarrowPhase iterates over the map, but does nothing as all detection output are nullptr
    EXPECT_NO_THROW(narrowPhaseDetection->beginNarrowPhase());

    //endNarrowPhase erases all nullptr detection output and their entries in the map
    narrowPhaseDetection->endNarrowPhase();
    EXPECT_TRUE(outputMap.empty());

    //add a single detection output
    auto*& detection_0 = narrowPhaseDetection->getDetectionOutputs(collisionModels[0].get(), collisionModels[1].get());
    EXPECT_EQ(outputMap.size(), 1);

    bool isDestroyed_0 { false };
    detection_0 = new sofa::core::collision::DummyDetectionOutputVector(0, &isDestroyed_0);

    // size is 0, so it should be removed from the map and destroyed
    narrowPhaseDetection->endNarrowPhase();
    EXPECT_TRUE(outputMap.empty());
    EXPECT_TRUE(isDestroyed_0);

    //add a single detection output
    auto*& detection_1 = narrowPhaseDetection->getDetectionOutputs(collisionModels[2].get(), collisionModels[1].get());
    EXPECT_EQ(outputMap.size(), 1);

    bool isDestroyed_1 { false };
    detection_1 = new sofa::core::collision::DummyDetectionOutputVector(1, &isDestroyed_1);

    // size is not 0, so it is NOT removed from the map and or destroyed
    narrowPhaseDetection->endNarrowPhase();
    EXPECT_FALSE(outputMap.empty());
    EXPECT_FALSE(isDestroyed_1);

    // beginNarrowPhase set the size to 0, so it will be removed from the map and destroyed
    narrowPhaseDetection->beginNarrowPhase();
    narrowPhaseDetection->endNarrowPhase();
    EXPECT_TRUE(outputMap.empty());
    EXPECT_TRUE(isDestroyed_1);
}
} //namespace sofa