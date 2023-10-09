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
#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaCUDA/component/solidmechanics/spring/CudaSpringForceField.inl>
#include <SofaCUDA/component/statecontainer/CudaMechanicalObject.inl>
#include <SofaCUDA/component/mapping/linear/CudaIdentityMapping.inl>
#include "CudaContactMapper.h"
#include <SofaCUDA/component/collision/response/contact/CudaPenalityContactForceField.h>
#include <SofaCUDA/component/collision/geometry/CudaSphereModel.h>
#include <SofaCUDA/component/collision/geometry/CudaPointModel.h>



#include <sofa/component/collision/detection/intersection/NewProximityIntersection.inl>
#include <sofa/component/collision/detection/intersection/MeshNewProximityIntersection.inl>
#include <sofa/component/collision/detection/intersection/RayDiscreteIntersection.inl>
#include <sofa/component/collision/detection/intersection/NewProximityIntersection.inl>
#include <sofa/component/collision/detection/intersection/DiscreteIntersection.h>

#include <sofa/component/collision/response/contact/RayContact.h>
#include <sofa/component/collision/response/contact/BarycentricPenalityContact.inl>
#include <sofa/component/collision/response/contact/PenalityContactForceField.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.inl>

#include <sofa/component/solidmechanics/spring/VectorSpringForceField.h>

#include <sofa/gl/gl.h>
#include <sofa/helper/Factory.inl>
#include <sofa/core/Mapping.inl>
#include <fstream>




namespace sofa::gpu::cuda
{

using namespace sofa::component::collision;
using namespace sofa::component::collision::geometry;
using namespace sofa::component::collision::detection::intersection;
using namespace sofa::component::collision::response::contact;


class CudaProximityIntersection : public NewProximityIntersection
{
public:
    SOFA_CLASS(CudaProximityIntersection, NewProximityIntersection);

    virtual void init() override
    {
        using CudaSphereCollisionModel = SphereCollisionModel<gpu::cuda::CudaVec3Types>;

        sofa::component::collision::NewProximityIntersection::init();
        intersectors.add<CudaSphereCollisionModel, CudaSphereCollisionModel, NewProximityIntersection>(this);
        RayDiscreteIntersection* rayIntersector = new RayDiscreteIntersection(this, false);
        intersectors.add<RayCollisionModel,        CudaSphereCollisionModel,   RayDiscreteIntersection>(rayIntersector);
        MeshNewProximityIntersection* meshIntersector = new MeshNewProximityIntersection(this, false);
        intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>,   CudaSphereCollisionModel,   MeshNewProximityIntersection>(meshIntersector);
    }

};


int CudaProximityIntersectionClass = core::RegisterObject("GPGPU Proximity Intersection based on CUDA")
        .add< CudaProximityIntersection >()
        ;

sofa::helper::Creator<core::collision::Contact::Factory, RayContact<SphereCollisionModel<gpu::cuda::CudaVec3Types>> > RayCudaSphereContactClass("RayContact",true);

} // namespace sofa::gpu::cuda
