#include <sofa/core/objectmodel/Data.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfoDynamicWrapper.h>
#include <sofa/defaulttype/TypeInfoRegistry.h>
namespace sofa::core::objectmodel
{

/// Get info about the value type of the associated variable
template<class T>
const sofa::defaulttype::AbstractTypeInfo* Data<T>::getValueTypeInfo() const
{
    return sofa::defaulttype::TypeInfoRegistry::get<T>();
}

}
