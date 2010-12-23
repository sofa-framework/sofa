#define FRAME_FRAMEVOLUMEPRESERVATIONFORCEFIELD_CPP

#include "FrameVolumePreservationForceField.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(FrameVolumePreservationForceField)

// Register in the Factory
int FrameVolumePreservationForceFieldClass = core::RegisterObject("Compute volume preservation forces on deformation gradients")
#ifndef SOFA_FLOAT
        .add< FrameVolumePreservationForceField<DeformationGradient331dTypes> >()
        .add< FrameVolumePreservationForceField<DeformationGradient332dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< FrameVolumePreservationForceField<DeformationGradient331fTypes> >()
        .add< FrameVolumePreservationForceField<DeformationGradient332fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameVolumePreservationForceField<DeformationGradient331dTypes>;
template class SOFA_FRAME_API FrameVolumePreservationForceField<DeformationGradient332dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API FrameVolumePreservationForceField<DeformationGradient331fTypes>;
template class SOFA_FRAME_API FrameVolumePreservationForceField<DeformationGradient332fTypes>;
#endif



}
}

} // namespace sofa


