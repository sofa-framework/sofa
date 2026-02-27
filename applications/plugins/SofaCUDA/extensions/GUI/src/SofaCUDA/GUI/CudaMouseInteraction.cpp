#include <SofaCUDA/GUI/config.h>

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/gui/component/performer/MouseInteractor.inl>

#include <SofaCUDA/component/collision/response/mapper/CudaContactMapper.h>
#include <sofa/gui/component/performer/ComponentMouseInteraction.inl>
#include <sofa/gui/component/performer/AttachBodyPerformer.inl>
#include <sofa/gui/component/performer/FixParticlePerformer.inl>
#include <sofa/gui/component/performer/BaseAttachBodyPerformer.inl>


#include <sofa/helper/Factory.inl>
#include <sofa/core/ObjectFactory.h>


namespace sofa::gui::component::performer
{

using namespace sofa::gpu::cuda;

template class SOFACUDA_GUI_API MouseInteractor<CudaVec3fTypes>;
template class SOFACUDA_GUI_API TComponentMouseInteraction< CudaVec3fTypes >;
template class SOFACUDA_GUI_API AttachBodyPerformer< CudaVec3fTypes >;
template class SOFACUDA_GUI_API FixParticlePerformer< CudaVec3fTypes >;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFACUDA_GUI_API MouseInteractor<CudaVec3dTypes>;
template class SOFACUDA_GUI_API TComponentMouseInteraction< CudaVec3dTypes >;
template class SOFACUDA_GUI_API AttachBodyPerformer< CudaVec3dTypes >;
template class SOFACUDA_GUI_API FixParticlePerformer< CudaVec3dTypes >;
#endif

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

    void registerMouseInteractor(sofa::core::ObjectFactory* factory)
    {
        factory->registerObjects(sofa::core::ObjectRegistrationData("Supports GPU-side computations using CUDA for the MouseInteractor")
        .add< MouseInteractor<CudaVec3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< MouseInteractor<CudaVec3dTypes> >()
#endif
        );
    }

}
