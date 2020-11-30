#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATAQUAT_NOEXTERN

#include <sofa/core/datatypes/DataQuat.inl>
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/defaulttype/typeinfo/TypeInfo_Quat.h>

namespace sofa::core::objectmodel
{
template class Data<sofa::defaulttype::Quatf>;
template class Data<sofa::defaulttype::Quatd>;
}
