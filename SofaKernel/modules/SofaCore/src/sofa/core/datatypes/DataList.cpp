#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATALIST_NOEXTERN
#include <sofa/core/datatypes/DataList.inl>
#include <sofa/core/objectmodel/Data.inl>

namespace sofa::core::objectmodel
{
    template class Data<std::list<sofa::core::topology::TopologyChange const*>>;
};
