#include "Binding_Base.h"
#include "Binding_BaseData.h"

#include <sofa/defaulttype/DataTypeInfo.h>
using sofa::defaulttype::AbstractTypeInfo;

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

#include <sofa/core/objectmodel/BaseObject.h>
using  sofa::core::objectmodel::BaseObject;

#include <sofa/core/objectmodel/BaseNode.h>
using  sofa::core::objectmodel::BaseNode;

#include <pybind11/numpy.h>

void moduleAddDataAsString(py::module& m)
{
    py::class_<DataAsString, BaseData, raw_ptr<DataAsString>> s(m, "DataString");
    s.def("__getitem__", [](BaseData& self, py::size_t index) -> py::object
    {
        auto nfo = self.getValueTypeInfo();
        return py::str(&nfo->getTextValue(self.getValueVoidPtr(),0).at(index),1);
    });
}

std::string getPathTo(Base* b)
{
    BaseNode* node = dynamic_cast<BaseNode*>(b);
    if(node)
        return node->getPathName();
    BaseObject* object = dynamic_cast<BaseObject*>(b);
    if(object)
        return object->getPathName();

    assert(true && "Only Base & BaseObject are supported");
}

const char* getFormat(const AbstractTypeInfo& nfo)
{
    if(nfo.Integer())
    {
        return py::format_descriptor<long>::value;
    } else if(nfo.Scalar() )
    {
        if(nfo.byteSize() == 8)
            return py::format_descriptor<double>::value;
        else if(nfo.byteSize() == 4)
            return py::format_descriptor<float>::value;
    }
    return nullptr;
}

void moduleAddDataAsContainer(py::module& m)
{
    py::class_<DataAsContainer, BaseData, raw_ptr<DataAsContainer>> p(m, "DataContainer",
                                                                      py::buffer_protocol());

    p.def("__getitem__", [](DataAsContainer& self, py::size_t index) -> py::object
    {
        const AbstractTypeInfo& nfo { *self.getValueTypeInfo() };

        if( index >= nfo.size(self.getValueVoidPtr())/nfo.size() )
            throw py::index_error();

        int cols = nfo.size();
        py::buffer_info binfo(
                    nfo.getValuePtr(self.beginEditVoidPtr())+nfo.byteSize()*cols*index, /* Pointer to buffer */
                    nfo.size(),                            /* Size of one scalar */
                    getFormat(nfo),                        /* Python struct-style format descriptor */
                    1,                                     /* Number of dimensions */
                    { cols },                              /* Buffer dimensions */
                    { nfo.byteSize() }                     /* Strides (in bytes) for each index */
        );

        return py::array(binfo);
    });

    p.def("__getitem__", [](DataAsContainer& self, py::slice slice) -> py::object
    {
        std::cout << "  single slice" << std::endl ;
        return py::none();
    });

    p.def("__getitem__", [](DataAsContainer& self, py::tuple ij) -> py::object
    {
        std::cout << "  dual axis " << std::endl ;
        return py::none();
    });

    p.def("__getitem__", [](DataAsContainer& self, py::function fct) -> py::object
    {
        std::cout << "  functional " << std::endl ;
        return py::none();
    });

    // TODO: Implementation should look like: https://github.com/sofa-framework/sofa/issues/767
    p.def("__setitem__", [](DataAsContainer& self, py::object& key, py::object& value)
    {
        std::cout << "mapping protocol, __setitem__ to implement)" << std::endl ;
        return py::none();
    });

    p.def("__str__", [](BaseData* self)
    {
        return py::str(toPython(self));
    });

    p.def("__repr__", [](BaseData* self)
    {
        return py::repr(toPython(self));
    });

    p.def("shape", [](BaseData& b) -> py::tuple
    {
        auto nfo = b.getValueTypeInfo();
        return py::make_tuple(py::int_(nfo->size(b.getValueVoidPtr())/nfo->size()),
                              py::int_(nfo->size()));
    });


    /// https://julien.danjou.info/high-performance-in-python-with-zero-copy-and-the-buffer-protocol/
    p.def_buffer([](BaseData& m) -> py::buffer_info
    {
        const AbstractTypeInfo& nfo { *m.getValueTypeInfo() };

        const char* format;
        if(nfo.Integer())
        {
            format = py::format_descriptor<long>::value;
        } else if(nfo.Scalar() )
        {
            if(nfo.byteSize() == 8)
                format = py::format_descriptor<double>::value;
            else if(nfo.byteSize() == 4)
                format = py::format_descriptor<float>::value;
        }
        int rows = nfo.size(m.getValueVoidPtr()) / nfo.size();
        int cols = nfo.size();
        size_t datasize = nfo.byteSize();

        if(rows == 1 && nfo.FixedSize()){
            return py::buffer_info(
                        nfo.getValuePtr(m.beginEditVoidPtr()), /* Pointer to buffer */
                        datasize,                              /* Size of one scalar */
                        format,                                /* Python struct-style format descriptor */
                        1,                                     /* Number of dimensions */
            { cols },                              /* Buffer dimensions */
            {                                      /* Strides (in bytes) for each index */
                                                   datasize }
                        );
        }
        return py::buffer_info(
                    nfo.getValuePtr(m.beginEditVoidPtr()), /* Pointer to buffer */
                    datasize,                              /* Size of one scalar */
                    format,                                /* Python struct-style format descriptor */
                    2,                                     /* Number of dimensions */
        { rows, cols },                        /* Buffer dimensions */
        { datasize * cols ,                           /* Strides (in bytes) for each index */
          datasize }
                    );
    });
}

void moduleAddBaseData(py::module& m)
{
    py::class_<BaseData, raw_ptr<BaseData>> p(m, "Data");
    p.def("setName", &BaseData::setName);
    p.def("getName", &BaseData::getName);

    // TODO: Implementation should look like: https://github.com/sofa-framework/sofa/issues/767
    p.def("__setitem__", [](BaseData& self, py::object& key, py::object& value)
    {
        std::cout << "mapping protocol, __setitem__ to implement)" << std::endl ;
        return py::none();
    });

    p.def("__len__", [](BaseData& b) -> size_t
    {
        auto nfo = b.getValueTypeInfo();
        return nfo->size(b.getValueVoidPtr()) / nfo->size();
    });

    p.def("getPath", [](BaseData& self){
        Base* b= self.getOwner();
        std::string prefix = getPathTo(b);
        return prefix+"."+self.getName();
    });

    p.def("__str__", [](BaseData* self)
    {
        return py::str(toPython(self));
    });

    p.def("__repr__", [](BaseData* self)
    {
        return py::repr(toPython(self));
    });
}
