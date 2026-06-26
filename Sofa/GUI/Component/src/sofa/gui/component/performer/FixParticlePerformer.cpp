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
#include <sofa/gui/component/performer/FixParticlePerformer.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/Factory.inl>

#include <sofa/component/collision/geometry/TriangleCollisionModel.h>
#include <sofa/component/collision/geometry/SphereCollisionModel.h>

namespace sofa::gui::component::performer
{
using namespace sofa::component::collision::geometry;

using FixParticlePerformer3d = FixParticlePerformer<defaulttype::Vec3Types>;

template class SOFA_GUI_COMPONENT_API FixParticlePerformer<defaulttype::Vec3Types>;
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer3d >  FixParticlePerformerVec3dClass("FixParticle",true);

int triangleFixParticle = FixParticlePerformer3d::RegisterSupportedModel<TriangleCollisionModel<defaulttype::Vec3Types>>(&FixParticlePerformer3d::getFixationPointsTriangle<TriangleCollisionModel<defaulttype::Vec3Types>>);
int sphereFixParticle = FixParticlePerformer3d::RegisterSupportedModel<SphereCollisionModel<defaulttype::Vec3Types>>(&FixParticlePerformer3d::getFixationPointsSphere);
int rigidSphereFixParticle = FixParticlePerformer3d::RegisterSupportedModel<SphereCollisionModel<defaulttype::Rigid3Types>>(&FixParticlePerformer3d::getFixationPointsSphere);


} // namespace sofa::gui::component::performer
