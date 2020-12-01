#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATAOPTIONGROUP_NOEXTERN
#include <sofa/core/datatype/Data[OptionGroup].inl>
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>

namespace sofa::defaulttype
{
template<>
struct DataTypeInfo<sofa::helper::OptionsGroup> : public IncompleteTypeInfo<sofa::helper::OptionsGroup>{};
}

namespace sofa::core::objectmodel
{
template class Data<sofa::helper::OptionsGroup>;
}
