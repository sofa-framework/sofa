#define FRAME_COROTATIONALFORCEFIELD_CPP

#include "DeformationGradientTypes.h"
#include "CorotationalForceField.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(CorotationalForceField)

// Register in the Factory
int CorotationalForceFieldClass = core::RegisterObject("Define a specific mass for each particle")
#ifndef SOFA_FLOAT
        .add< CorotationalForceField<DeformationGradient331dTypes> >()
        .add< CorotationalForceField<DeformationGradient332dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< CorotationalForceField<DeformationGradient331fTypes> >()
        .add< CorotationalForceField<DeformationGradient332fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_FRAME_API CorotationalForceField<DeformationGradient331dTypes>;
template class SOFA_FRAME_API CorotationalForceField<DeformationGradient332dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API CorotationalForceField<DeformationGradient331fTypes>;
template class SOFA_FRAME_API CorotationalForceField<DeformationGradient332fTypes>;
#endif



}
}

} // namespace sofa


