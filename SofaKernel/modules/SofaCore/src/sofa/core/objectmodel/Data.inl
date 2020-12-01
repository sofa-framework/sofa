#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/DataTypeInfoHelper.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfoDynamicWrapper.h>
#include <sofa/defaulttype/TypeInfoRegistry.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[vector].h>
#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>
namespace sofa::core::objectmodel
{
template<class T>
const sofa::defaulttype::AbstractTypeInfo* Data<T>::GetValueTypeInfoValidTypeInfo()
{
    return sofa::defaulttype::TypeInfoRegistry::get<T>();
}
}
