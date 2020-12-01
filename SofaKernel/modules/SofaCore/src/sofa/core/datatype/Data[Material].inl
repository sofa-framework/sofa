#pragma once

#include <sofa/core/datatype/Data[Material].h>
#include <sofa/core/objectmodel/Data.inl>

#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>
namespace sofa::defaulttype
{
template<> struct DataTypeInfo<sofa::helper::types::Material> : public IncompleteTypeInfo<sofa::helper::types::Material> {};
}


