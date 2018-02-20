/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_BRUTEFORCEDETECTION_H
#define SOFA_COMPONENT_COLLISION_BRUTEFORCEDETECTION_H
#include "config.h"

#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/CollisionElement.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/defaulttype/Vec.h>
#include <set>


namespace sofa
{

namespace component
{

namespace collision
{


class SOFA_BASE_COLLISION_API MirrorIntersector : public core::collision::ElementIntersector
{
public:
    core::collision::ElementIntersector* intersector;

    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    virtual bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
    {
        return intersector->canIntersect(elem2, elem1);
    }

    /// Begin intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    /// If the given contacts vector is NULL, then this method should allocate it.
    virtual int beginIntersect(core::CollisionModel* model1, core::CollisionModel* model2, core::collision::DetectionOutputVector*& contacts)
    {
        return intersector->beginIntersect(model2, model1, contacts);
    }

    /// Compute the intersection between 2 elements. Return the number of contacts written in the contacts vector.
    virtual int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, core::collision::DetectionOutputVector* contacts)
    {
        return intersector->intersect(elem2, elem1, contacts);
    }

    /// End intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    virtual int endIntersect(core::CollisionModel* model1, core::CollisionModel* model2, core::collision::DetectionOutputVector* contacts)
    {
        return intersector->endIntersect(model2, model1, contacts);
    }

    virtual std::string name() const
    {
        return intersector->name() + std::string("<SWAP>");
    }

};


class SOFA_BASE_COLLISION_API BruteForceDetection :
    public core::collision::BroadPhaseDetection,
    public core::collision::NarrowPhaseDetection
{
public:
    SOFA_CLASS2(BruteForceDetection, core::collision::BroadPhaseDetection, core::collision::NarrowPhaseDetection);

private:
    bool _is_initialized;
    sofa::helper::vector<core::CollisionModel*> collisionModels;

    Data< helper::fixed_array<sofa::defaulttype::Vector3,2> > box; ///< if not empty, objects that do not intersect this bounding-box will be ignored

    CubeModel::SPtr boxModel;


protected:
    BruteForceDetection();

    ~BruteForceDetection();

    virtual bool keepCollisionBetween(core::CollisionModel *cm1, core::CollisionModel *cm2);

public:

    void init() override;
    void reinit() override;

    void addCollisionModel (core::CollisionModel *cm) override;
    void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair) override;

    virtual void beginBroadPhase() override
    {
        core::collision::BroadPhaseDetection::beginBroadPhase();
        collisionModels.clear();
    }

    void draw(const core::visual::VisualParams* /* vparams */) override { }

    inline virtual bool needsDeepBoundingTree()const override {return true;}
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
