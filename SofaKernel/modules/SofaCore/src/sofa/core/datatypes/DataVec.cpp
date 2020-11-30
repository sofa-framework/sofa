#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATAVEC_INTERN
#include <sofa/core/datatypes/DataVec.h>
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>

namespace sofa::core::objectmodel
{
template class Data<sofa::defaulttype::Vec3d>;
template class Data<sofa::defaulttype::Vec3f>;
}
