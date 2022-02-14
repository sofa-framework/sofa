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
#include <sofa/gpu/cuda/CudaSpringForceField.inl>
#include <sofa/gpu/cuda/CudaMechanicalObject.inl>
#include <sofa/gpu/cuda/CudaIdentityMapping.inl>
#include <sofa/gpu/cuda/CudaContactMapper.h>
#include <sofa/gpu/cuda/CudaPenalityContactForceField.h>
#include <sofa/gpu/cuda/CudaSpringForceField.h>

#include <sofa/gpu/cuda/CudaSphereModel.h>
#include <sofa/gpu/cuda/CudaTriangleModel.h>
#include <sofa/gpu/cuda/CudaLineModel.h>
#include <sofa/gpu/cuda/CudaPointModel.h>

#include <SofaConstraint/LocalMinDistance.h>
#include <SofaUserInteraction/MouseInteractor.inl>
#include <SofaBaseCollision/MinProximityIntersection.h>
#include <SofaBaseCollision/NewProximityIntersection.inl>
#include <SofaMeshCollision/MeshNewProximityIntersection.inl>
#include <SofaUserInteraction/RayDiscreteIntersection.inl>
#include <SofaBaseCollision/DiscreteIntersection.h>
#include <SofaUserInteraction/ComponentMouseInteraction.inl>
#include <SofaUserInteraction/AttachBodyPerformer.inl>
#include <SofaUserInteraction/FixParticlePerformer.inl>
#include <SofaUserInteraction/RayContact.h>
#include <SofaMeshCollision/BarycentricPenalityContact.inl>
#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include <SofaObjectInteraction/PenalityContactForceField.h>
#include <sofa/component/solidmechanics/spring/VectorSpringForceField.h>
#include <sofa/gl/gl.h>
#include <sofa/helper/Factory.inl>
#include <sofa/core/Mapping.inl>
#include <fstream>

namespace sofa::component::collision
{

using namespace sofa::gpu::cuda;

template class SOFA_GPU_CUDA_API MouseInteractor<CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API TComponentMouseInteraction< CudaVec3fTypes >;
template class SOFA_GPU_CUDA_API AttachBodyPerformer< CudaVec3fTypes >;
template class SOFA_GPU_CUDA_API FixParticlePerformer< CudaVec3fTypes >;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API MouseInteractor<CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API TComponentMouseInteraction< CudaVec3dTypes >;
template class SOFA_GPU_CUDA_API AttachBodyPerformer< CudaVec3dTypes >;
template class SOFA_GPU_CUDA_API FixParticlePerformer< CudaVec3dTypes >;
#endif

ContactMapperCreator< ContactMapper<CudaSphereCollisionModel> > CudaSphereContactMapperClass("PenalityContactForceField",true);
ContactMapperCreator< ContactMapper<CudaTriangleCollisionModel> > CudaTiangleContactMapperClass("PenalityContactForceField", true);

helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<CudaVec3fTypes> > ComponentMouseInteractionCudaVec3fClass ("MouseSpringCudaVec3f",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer <CudaVec3fTypes> >  AttachBodyPerformerCudaVec3fClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<CudaVec3fTypes> >  FixParticlePerformerCudaVec3fClass("FixParticle",true);


///////////////////////////////////////////////
/////   Add CUDA support for Contact    ///////
///////////////////////////////////////////////

using namespace core::collision;
using namespace sofa::helper;
const Creator<Contact::Factory, BarycentricPenalityContact<CudaSphereCollisionModel, CudaSphereCollisionModel, gpu::cuda::CudaVec3Types> > CudaSphereSpherePenalityContactClass("PenalityContactForceField", true);
const Creator<Contact::Factory, BarycentricPenalityContact<CudaSphereCollisionModel, CudaTriangleCollisionModel, gpu::cuda::CudaVec3Types> > CudaSphereTrianglePenalityContactClass("PenalityContactForceField", true);
const Creator<Contact::Factory, BarycentricPenalityContact<CudaTriangleCollisionModel, CudaSphereCollisionModel, gpu::cuda::CudaVec3Types> > CudaTriangleSpherePenalityContactClass("PenalityContactForceField", true);

template class SOFA_GPU_CUDA_API BarycentricPenalityContact<CudaSphereCollisionModel, CudaSphereCollisionModel, gpu::cuda::CudaVec3Types>;
template class SOFA_GPU_CUDA_API BarycentricPenalityContact<CudaSphereCollisionModel, CudaTriangleCollisionModel, gpu::cuda::CudaVec3Types>;


#ifdef SOFA_GPU_CUDA_DOUBLE
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<CudaVec3dTypes> > ComponentMouseInteractionCudaVec3dClass ("MouseSpringCudaVec3d",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer <CudaVec3dTypes> >  AttachBodyPerformerCudaVec3dClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<CudaVec3dTypes> >  FixParticlePerformerCudaVec3dClass("FixParticle",true);
#endif

using FixParticlePerformerCuda3d = FixParticlePerformer<gpu::cuda::CudaVec3Types>;

int triangleFixParticle = FixParticlePerformerCuda3d::RegisterSupportedModel<TriangleCollisionModel<gpu::cuda::Vec3Types>>(&FixParticlePerformerCuda3d::getFixationPointsTriangle<TriangleCollisionModel<gpu::cuda::Vec3Types>>);


} //namespace sofa::component::collision


namespace sofa::gpu::cuda
{


int MouseInteractorCudaClass = core::RegisterObject("Supports Mouse Interaction using CUDA")
        .add< component::collision::MouseInteractor<CudaVec3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::collision::MouseInteractor<CudaVec3dTypes> >()
#endif
        ;


using namespace sofa::component::collision;

class CudaProximityIntersection : public sofa::component::collision::NewProximityIntersection
{
public:
    SOFA_CLASS(CudaProximityIntersection,sofa::component::collision::NewProximityIntersection);

    virtual void init() override
    {
        sofa::component::collision::NewProximityIntersection::init();
        intersectors.add<CudaSphereCollisionModel, CudaSphereCollisionModel, NewProximityIntersection>(this);
        //intersectors.add<CudaSphereCollisionModel, CudaTriangleCollisionModel, NewProximityIntersection>(this);
        
        RayDiscreteIntersection* rayIntersector = new RayDiscreteIntersection(this, false);
        intersectors.add<RayCollisionModel,        CudaSphereCollisionModel,   RayDiscreteIntersection>(rayIntersector);
        //MeshNewProximityIntersection* meshIntersector = new MeshNewProximityIntersection(this, false);
        //intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>,   CudaSphereCollisionModel,   MeshNewProximityIntersection>(meshIntersector);

        

    }

};

int CudaProximityIntersectionClass = core::RegisterObject("GPGPU Proximity Intersection based on CUDA")
    .add< CudaProximityIntersection >();



class CudaMinProximityIntersection : public sofa::component::collision::MinProximityIntersection
{
public:
    SOFA_CLASS(CudaMinProximityIntersection, sofa::component::collision::MinProximityIntersection);
    
    virtual void init() override
    {
        intersectors.add<CudaSphereCollisionModel, CudaSphereCollisionModel, CudaMinProximityIntersection>(this);
        //intersectors.add<CudaSphereCollisionModel, CudaTriangleCollisionModel, CudaMinProximityIntersection>(this);

        MinProximityIntersection::init();
    }
};

int CudaMinProximityIntersectionClass = core::RegisterObject("GPGPU Proximity Intersection based on CUDA")
    .add< CudaMinProximityIntersection >();




class CudaMeshMinProximityIntersection : public core::collision::BaseIntersector
{
    typedef MinProximityIntersection::OutputVector OutputVector;

public:
    CudaMeshMinProximityIntersection(MinProximityIntersection* object, bool addSelf = true)
        : intersection(object)
    {
        intersection->intersectors.add<CudaTriangleCollisionModel, CudaSphereCollisionModel, CudaMeshMinProximityIntersection>(this);        
    }


    bool testIntersection(CudaTriangle& e2, CudaSphere& e1)
    {
        const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

        const type::Vector3 x13 = e2.p1() - e2.p2();
        const type::Vector3 x23 = e2.p1() - e2.p3();
        const type::Vector3 x03 = e2.p1() - e1.center();
        type::Matrix2 A;
        type::Vector2 b;
        A[0][0] = x13 * x13;
        A[1][1] = x23 * x23;
        A[0][1] = A[1][0] = x13 * x23;
        b[0] = x13 * x03;
        b[1] = x23 * x03;
        const SReal det = type::determinant(A);

        SReal alpha = 0.5;
        SReal beta = 0.5;

        //if (det < -0.000001 || det > 0.000001)
        {
            alpha = (b[0] * A[1][1] - b[1] * A[0][1]) / det;
            beta = (b[1] * A[0][0] - b[0] * A[1][0]) / det;
            if (alpha < 0.000001 ||
                beta < 0.000001 ||
                alpha + beta  > 0.999999)
                return false;
        }

        type::Vector3 P, Q, PQ;
        P = e1.center();
        Q = e2.p1() - x13 * alpha - x23 * beta;
        PQ = Q - P;

        if (PQ.norm2() < alarmDist * alarmDist)
        {
            return true;
        }
        else
            return false;
    }

    int computeIntersection(CudaTriangle& e2, CudaSphere& e1, OutputVector* contacts)
    {
        const SReal alarmDist = intersection->getAlarmDistance() + e1.r() + e1.getProximity() + e2.getProximity();

        const type::Vector3 x13 = e2.p1() - e2.p2();
        const type::Vector3 x23 = e2.p1() - e2.p3();
        const type::Vector3 x03 = e2.p1() - e1.center();
        type::Matrix2 A;
        type::Vector2 b;
        A[0][0] = x13 * x13;
        A[1][1] = x23 * x23;
        A[0][1] = A[1][0] = x13 * x23;
        b[0] = x13 * x03;
        b[1] = x23 * x03;
        const SReal det = type::determinant(A);

        SReal alpha = 0.5;
        SReal beta = 0.5;

        //if (det < -0.000001 || det > 0.000001)
        {
            alpha = (b[0] * A[1][1] - b[1] * A[0][1]) / det;
            beta = (b[1] * A[0][0] - b[0] * A[1][0]) / det;
            if (alpha < 0.000001 ||
                beta < 0.000001 ||
                alpha + beta  > 0.999999)
                return 0;
        }

        type::Vector3 P = e1.center();
        type::Vector3 Q = e2.p1() - x13 * alpha - x23 * beta;
        type::Vector3 QP = P - Q;
        //Vector3 PQ = Q-P;

        if (QP.norm2() >= alarmDist * alarmDist)
            return 0;

        const SReal contactDist = intersection->getContactDistance() + e1.r() + e1.getProximity() + e2.getProximity();

        contacts->resize(contacts->size() + 1);
        sofa::core::collision::DetectionOutput* detection = &*(contacts->end() - 1);
        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e2, e1);
        detection->id = e1.getIndex();
        detection->normal = QP;
        detection->value = detection->normal.norm();
        if (detection->value > 1e-15)
        {
            detection->normal /= detection->value;
        }
        else
        {
            msg_warning(intersection) << "Null distance between contact detected";
            detection->normal = type::Vector3(1, 0, 0);
        }
        detection->value -= contactDist;
        detection->point[0] = Q;
        detection->point[1] = e1.getContactPointByNormal(detection->normal);

        return 1;
    }


protected:
    MinProximityIntersection* intersection;
};

IntersectorCreator<MinProximityIntersection, CudaMeshMinProximityIntersection> CudaMeshMinProximityIntersectors("CudaMesh");

sofa::helper::Creator<core::collision::Contact::Factory, component::collision::RayContact<sofa::component::collision::SphereCollisionModel<gpu::cuda::CudaVec3Types>> > RayCudaSphereContactClass("RayContact",true);

} // namespace sofa::gpu::cuda
