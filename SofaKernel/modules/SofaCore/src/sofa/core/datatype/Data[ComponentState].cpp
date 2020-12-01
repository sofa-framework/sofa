#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATACOMPONENTSTATE_NOEXTERN
#include <sofa/core/datatype/Data[ComponentState].h>
#include <sofa/core/objectmodel/Data.inl>

#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>
namespace sofa::defaulttype
{

template<>
struct DataTypeInfo<sofa::core::objectmodel::ComponentState> : public IncompleteTypeInfo<sofa::core::objectmodel::ComponentState>
{

};

}


namespace sofa::core::objectmodel
{
template class Data<sofa::core::objectmodel::ComponentState>;
}
