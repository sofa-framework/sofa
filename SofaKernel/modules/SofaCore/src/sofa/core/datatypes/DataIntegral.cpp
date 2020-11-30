#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATAMATERIAL_NOEXTERN
#include <sofa/core/datatypes/DataIntegral.inl>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Bool.h>
#include <sofa/core/objectmodel/Data.inl>

namespace sofa::core::objectmodel
{
template class Data<bool>;
template class Data<char>;
template class Data<unsigned char>;
template class Data<short>;
template class Data<unsigned short>;
template class Data<int>;
template class Data<unsigned int>;
template class Data<long>;
template class Data<unsigned long>;
template class Data<long long>;
template class Data<unsigned long long>;
}
