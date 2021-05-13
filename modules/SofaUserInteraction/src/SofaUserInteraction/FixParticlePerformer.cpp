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

namespace sofa::component::collision
{
using FixParticlePerformer3d = FixParticlePerformer<defaulttype::Vec3Types>;

template class SOFA_SOFAUSERINTERACTION_API FixParticlePerformer<defaulttype::Vec3Types>;
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer3d >  FixParticlePerformerVec3dClass("FixParticle",true);

std::unordered_map<std::type_index, FixParticlePerformer3d::PairModelFunction> FixParticlePerformer3d::s_mapSupportedModels;

int triangleFixParticle = FixParticlePerformer3d::RegisterSupportedModel<sofa::component::collision::TriangleModel>(
    []
    (sofa::core::sptr<sofa::core::CollisionModel> model, const Index idx, helper::vector<Index>& points, FixParticlePerformer3d::Coord& fixPoint)
    {
        std::cout << "Triangle was registered you know" << std::endl;

        auto* triangle = dynamic_cast<TriangleModel*>(model.get());
        
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

int sphereFixParticle = FixParticlePerformer3d::RegisterSupportedModel<sofa::component::collision::SphereModel>(
    []
(sofa::core::sptr<sofa::core::CollisionModel> model, const Index idx, helper::vector<Index>& points, FixParticlePerformer3d::Coord& fixPoint)
    {
        std::cout << "sphere was registered you know" << std::endl;

        auto* sphere = dynamic_cast<SphereModel*>(model.get());

        if (!sphere)
            return false;

        Sphere s(sphere, idx);
        fixPoint = s.p();
        points.push_back(s.getIndex());

        return true;
    }
);

} // namespace sofa::component::collision
