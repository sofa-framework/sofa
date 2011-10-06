#define INDEXVALUEMAPPER_CPP_

#include "IndexValueMapper.inl"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(IndexValueMapper)

int IndexValueMapperClass = core::RegisterObject("?")
#ifndef SOFA_FLOAT
        .add< IndexValueMapper<Vec3dTypes> >(true)
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< IndexValueMapper<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_ENGINE_API IndexValueMapper<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_ENGINE_API IndexValueMapper<Vec3fTypes>;
#endif //SOFA_DOUBLE


} // namespace engine

} // namespace component

} // namespace sofa
