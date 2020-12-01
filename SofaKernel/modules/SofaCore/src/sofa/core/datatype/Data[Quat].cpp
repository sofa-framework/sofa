#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATAQUAT_NOEXTERN

#include <sofa/core/datatype/Data[Quat].inl>
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[Quat].h>

namespace sofa::core::objectmodel
{
template class Data<sofa::defaulttype::Quatf>;
template class Data<sofa::defaulttype::Quatd>;

template class Data<sofa::helper::vector<sofa::defaulttype::Quatf>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Quatd>>;
}
