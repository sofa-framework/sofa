#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATAVECTOR_INTERN
#include <sofa/core/datatypes/DataVector.inl>
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vector.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>
#include <sofa/helper/types/RGBAColor.h>

namespace sofa::core::objectmodel
{
template class Data<sofa::helper::vector<unsigned int>>;
template class Data<sofa::helper::vector<int>>;
template class Data<sofa::helper::vector<std::string>>;
template class Data<sofa::helper::vector<sofa::helper::types::RGBAColor>>;
template class Data<sofa::helper::types::RGBAColor>;

template class Data<sofa::helper::vector<sofa::defaulttype::Vec3d>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Vec3f>>;

template class Data<sofa::helper::vector<sofa::helper::vector<unsigned int>>>;
}
