#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATAVECTOR_INTERN
#include <sofa/core/datatype/Data[Vector].inl>
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[RGBAColor].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[string].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[vector].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[Vec].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[Integer].h>
#include <sofa/helper/types/RGBAColor.h>

namespace sofa::core::objectmodel
{
template class Data<sofa::helper::vector<float>>;
template class Data<sofa::helper::vector<double>>;

template class Data<sofa::helper::vector<unsigned int>>;
template class Data<sofa::helper::vector<int>>;
template class Data<sofa::helper::vector<std::string>>;
template class Data<sofa::helper::vector<sofa::helper::types::RGBAColor>>;
template class Data<sofa::helper::types::RGBAColor>;

template class Data<sofa::helper::vector<sofa::defaulttype::Vec3d>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Vec3f>>;

template class Data<sofa::helper::vector<sofa::helper::vector<unsigned char>>>;
template class Data<sofa::helper::vector<sofa::helper::vector<unsigned int>>>;
template class Data<sofa::helper::vector<sofa::helper::vector<unsigned short>>>;
template class Data<sofa::helper::vector<sofa::helper::vector<int>>>;
}
