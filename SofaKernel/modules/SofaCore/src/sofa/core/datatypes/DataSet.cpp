#define SOFA_CORE_OBJECTMODEL_DATATYPES_DATASET_INTERN
#include <sofa/core/datatypes/DataSet.h>
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/core/objectmodel/Tag.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Set.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Text.h>

namespace sofa::defaulttype
{

template<>
struct DataTypeInfo< sofa::core::objectmodel::Tag > : public TextTypeInfo<sofa::core::objectmodel::Tag >
{
    static const std::string name() { return "Tag"; }
};

template<>
struct DataTypeInfo< sofa::core::objectmodel::TagSet > : public SetTypeInfo<sofa::core::objectmodel::TagSet >
{
    static const std::string name() { return "TagSet"; }
};

template<class T>
struct DataTypeInfo<std::set<T>> : public SetTypeInfo<sofa::core::objectmodel::TagSet >
{
    static const std::string name() { return "set<"+DataTypeInfo<T>::name()+">"; }
};

}

namespace sofa::core::objectmodel
{
template class Data<std::set<int>>;
template class Data<sofa::core::objectmodel::Tag>;
template class Data<sofa::core::objectmodel::TagSet>;
};
