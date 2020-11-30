#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATABOUNDINGBOX_INTERN
#include <sofa/core/datatypes/DataList.inl>
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/defaulttype/typeinfo/TypeInfo_BoundingBox.h>

namespace sofa::core::objectmodel
{
template<>
bool Data<sofa::defaulttype::BoundingBox>::operator==(const sofa::defaulttype::BoundingBox& value) const { return false; }

template<>
bool Data<sofa::defaulttype::BoundingBox>::operator!=(const sofa::defaulttype::BoundingBox& value) const { return false; }

template class Data<sofa::defaulttype::BoundingBox>;
};
