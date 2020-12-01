#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATASCALAR_INTERN
#include <sofa/core/datatype/Data[Scalar].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[Scalar].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[vector].h>
#include <sofa/core/objectmodel/Data.inl>

namespace sofa::core::objectmodel
{
template class Data<double>;
template class Data<float>;

template class Data<sofa::helper::vector<float>>;
template class Data<sofa::helper::vector<double>>;

template class Data<sofa::helper::vector<sofa::helper::vector<float>>>;
template class Data<sofa::helper::vector<sofa::helper::vector<double>>>;
}
