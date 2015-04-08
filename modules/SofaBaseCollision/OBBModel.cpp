#include <SofaBaseCollision/OBBModel.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

SOFA_DECL_CLASS(OBB)

int OBBModelClass = core::RegisterObject("Collision model which represents a set of OBBs")
#ifndef SOFA_FLOAT
        .add<  TOBBModel<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add < TOBBModel<Rigid3fTypes> >()
#endif
        .addAlias("OBB")
        .addAlias("OBBModel")
//.addAlias("OBBMesh")
//.addAlias("OBBSet")
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TOBBModel<defaulttype::Rigid3dTypes>;
template class SOFA_BASE_COLLISION_API TOBB<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TOBBModel<defaulttype::Rigid3fTypes>;
template class SOFA_BASE_COLLISION_API TOBB<defaulttype::Rigid3fTypes>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa


