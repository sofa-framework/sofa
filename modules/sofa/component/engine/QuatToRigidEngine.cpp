#define QUATTORIGIDENGINE_CPP

#include <sofa/component/engine/QuatToRigidEngine.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(QuatToRigidEngine)

int QuatToRigidEngineClass = core::RegisterObject("Transform a vector of Rigids into two independant vectors for positions (Vec3) and orientations (Quat).")
#ifndef SOFA_FLOAT
        .add< QuatToRigidEngine<sofa::defaulttype::Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< QuatToRigidEngine<sofa::defaulttype::Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_ENGINE_API QuatToRigidEngine<sofa::defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_ENGINE_API QuatToRigidEngine<sofa::defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE


} // namespace engine

} // namespace component

} // namespace sofa
