#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaCUDA/component/solidmechanics/spring/CudaSpringForceField.inl>
#include <SofaCUDA/component/statecontainer/CudaMechanicalObject.inl>
#include <SofaCUDA/component/mapping/linear/CudaIdentityMapping.inl>
#include <SofaCUDA/component/collision/response/contact/CudaPenalityContactForceField.h>
#include <SofaCUDA/component/collision/geometry/CudaSphereModel.h>
#include <sofa/gui/component/performer/MouseInteractor.inl>

#include <sofa/gpu/cuda/CudaContactMapper.h>
#include <sofa/gui/component/performer/ComponentMouseInteraction.inl>
#include <sofa/gui/component/performer/AttachBodyPerformer.inl>
#include <sofa/gui/component/performer/FixParticlePerformer.inl>

#include <sofa/component/collision/detection/intersection/RayDiscreteIntersection.inl>
#include <sofa/component/collision/detection/intersection/DiscreteIntersection.h>

#include <sofa/component/collision/response/contact/RayContact.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.inl>

#include <sofa/component/solidmechanics/spring/VectorSpringForceField.h>

#include <sofa/gl/gl.h>
#include <sofa/helper/Factory.inl>
#include <sofa/core/Mapping.inl>
#include <fstream>

namespace sofa::gui::component::performer
{
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

using namespace sofa::gpu::cuda;
using namespace sofa::component::collision;
using namespace sofa::component::collision::geometry;
using namespace sofa::component::collision::response::mapper;

sofa::component::collision::response::mapper::ContactMapperCreator< sofa::component::collision::response::mapper::ContactMapper<CudaSphereCollisionModel> > CudaSphereContactMapperClass("PenalityContactForceField",true);

helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<CudaVec3fTypes> > ComponentMouseInteractionCudaVec3fClass ("MouseSpringCudaVec3f",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer <CudaVec3fTypes> >  AttachBodyPerformerCudaVec3fClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<CudaVec3fTypes> >  FixParticlePerformerCudaVec3fClass("FixParticle",true);

#ifdef SOFA_GPU_CUDA_DOUBLE
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<CudaVec3dTypes> > ComponentMouseInteractionCudaVec3dClass ("MouseSpringCudaVec3d",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer <CudaVec3dTypes> >  AttachBodyPerformerCudaVec3dClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<CudaVec3dTypes> >  FixParticlePerformerCudaVec3dClass("FixParticle",true);
#endif

using FixParticlePerformerCuda3d = FixParticlePerformer<gpu::cuda::CudaVec3Types>;

int triangleFixParticle = FixParticlePerformerCuda3d::RegisterSupportedModel<sofa::component::collision::geometry::TriangleCollisionModel<gpu::cuda::Vec3Types>>(&FixParticlePerformerCuda3d::getFixationPointsTriangle<sofa::component::collision::geometry::TriangleCollisionModel<gpu::cuda::Vec3Types>>);

} // namespace sofa::gui::component::performer



namespace sofa::gpu::cuda
{

using namespace sofa::gui::component::performer;

int MouseInteractorCudaClass = core::RegisterObject("Supports Mouse Interaction using CUDA")
                                   .add< MouseInteractor<CudaVec3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
                                   .add< MouseInteractor<CudaVec3dTypes> >()
#endif
    ;

}
