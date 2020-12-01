#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATAVEC_INTERN
#include <sofa/core/datatype/Data[Vec].h>
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[Vec].h>

REGISTER_DATATYPEINFO(sofa::defaulttype::Vec1d);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec2d);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec3d);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec4d);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec6d);

REGISTER_DATATYPEINFO(sofa::defaulttype::Vec1f);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec2f);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec3f);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec4f);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec6f);

REGISTER_DATATYPEINFO(sofa::defaulttype::Vec1i);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec2i);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec3i);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec4i);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec6i);

REGISTER_DATATYPEINFO(sofa::defaulttype::Vec2u);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec3u);

REGISTER_DATATYPEINFO(sofa::defaulttype::Vec2l);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec3l);

REGISTER_DATATYPEINFO(sofa::defaulttype::Vec2L);
REGISTER_DATATYPEINFO(sofa::defaulttype::Vec3L);

namespace sofa::core::objectmodel
{
template class Data<sofa::helper::fixed_array<sofa::defaulttype::Vec3d, 2>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Vec<10,float>>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Vec<10,double>>>;
}
