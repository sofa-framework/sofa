#pragma once

#include <sofa/core/datatypes/DataMaterial.h>
#include <sofa/core/objectmodel/Data.inl>

#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>
namespace sofa::defaulttype
{
template<> struct DataTypeInfo<sofa::helper::types::Material> : public IncompleteTypeInfo<sofa::helper::types::Material> {};
}

namespace sofa::core::objectmodel
{
template<> bool Data<sofa::helper::types::Material>::operator==(const sofa::helper::types::Material& value) const;
template<> bool Data<sofa::helper::types::Material>::operator!=(const sofa::helper::types::Material& value) const;
}

