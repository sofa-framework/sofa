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
#include <SofaUserInteraction/FixParticlePerformer.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/Factory.inl>

#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>

namespace sofa::component::collision
{
using FixParticlePerformer3d = FixParticlePerformer<defaulttype::Vec3Types>;

template class SOFA_SOFAUSERINTERACTION_API FixParticlePerformer<defaulttype::Vec3Types>;
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer3d >  FixParticlePerformerVec3dClass("FixParticle",true);

std::unordered_map<std::type_index, FixParticlePerformer3d::PairModelFunction> FixParticlePerformer3d::s_mapSupportedModels;

int triangleFixParticle = FixParticlePerformer3d::RegisterSupportedModel<TriangleCollisionModel<defaulttype::Vec3Types>>(
    []
    (sofa::core::sptr<sofa::core::CollisionModel> model, const Index idx, helper::vector<Index>& points, FixParticlePerformer3d::Coord& fixPoint)
    {
        auto* triangle = dynamic_cast<TriangleCollisionModel<defaulttype::Vec3Types>*>(model.get());
        
        if (!triangle)
            return false;

        Triangle t(triangle, idx);
        fixPoint = (t.p1() + t.p2() + t.p3()) / 3.0;
        points.push_back(t.p1Index());
        points.push_back(t.p2Index());
        points.push_back(t.p3Index());

        return true;
    }
);

int sphereFixParticle = FixParticlePerformer3d::RegisterSupportedModel<SphereCollisionModel<defaulttype::Vec3Types>>(
    []
(sofa::core::sptr<sofa::core::CollisionModel> model, const Index idx, helper::vector<Index>& points, FixParticlePerformer3d::Coord& fixPoint)
    {
        auto* sphere = dynamic_cast<SphereCollisionModel<defaulttype::Vec3Types>*>(model.get());

        if (!sphere)
            return false;

        Sphere s(sphere, idx);
        fixPoint = s.p();
        points.push_back(s.getIndex());

        return true;
    }
);

int rigidSphereFixParticle = FixParticlePerformer3d::RegisterSupportedModel<SphereCollisionModel<defaulttype::Rigid3Types>>(
    []
(sofa::core::sptr<sofa::core::CollisionModel> model, const Index idx, helper::vector<Index>& points, FixParticlePerformer3d::Coord& fixPoint)
    {
        auto* rigidSphere = dynamic_cast<SphereCollisionModel<defaulttype::Rigid3Types>*>(model.get());

        if (!rigidSphere)
            return false;

        auto* collisionState = model->getContext()->getMechanicalState();
        fixPoint[0] = collisionState->getPX(idx);
        fixPoint[1] = collisionState->getPY(idx);
        fixPoint[2] = collisionState->getPZ(idx);

        points.push_back(idx);

        return true;
    }
);

} // namespace sofa::component::collision
