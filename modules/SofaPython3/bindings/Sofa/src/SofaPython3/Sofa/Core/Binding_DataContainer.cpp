
#include <sofa/defaulttype/DataTypeInfo.h>
using sofa::defaulttype::AbstractTypeInfo;

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

#include <sofa/core/objectmodel/BaseObject.h>
using  sofa::core::objectmodel::BaseObject;

#include <sofa/core/objectmodel/BaseNode.h>
using  sofa::core::objectmodel::BaseNode;

#include "DataHelper.h"
#include "Binding_Base.h"
#include "Binding_BaseData.h"
#include "Binding_DataContainer.h"

//namespace pybind11 { namespace detail {
//    template <> struct type_caster<sofa::helper::types::RGBAColor> {
//    public:
//        /**
//         * This macro establishes the name 'inty' in
//         * function signatures and declares a local variable
//         * 'value' of type inty
//         */
//        PYBIND11_TYPE_CASTER(sofa::helper::types::RGBAColor, _("RGBAColor"));

//        /**
//         * Conversion part 1 (Python->C++): convert a PyObject into a inty
//         * instance or return false upon failure. The second argument
//         * indicates whether implicit conversions should be applied.
//         */
//        bool load(handle src, bool) {
//            std::cout << "LOAD LOAD" << std::endl ;
//        }

//        /**
//         * Conversion part 2 (C++ -> Python): convert an inty instance into
//         * a Python object. The second and third arguments are used to
//         * indicate the return value policy and parent object (for
//         * ``return_value_policy::reference_internal``) and are generally
//         * ignored by implicit casters.
//         */
//        static handle cast(sofa::core::objectmodel::RGBAColor src, return_value_policy /* policy */, handle /* parent */) {
//            std::cout << "CAST CAST" << std::endl ;
//        }
//    };
//}} // namespace pybind11::detail

namespace sofapython3
{
class WriteAccessor
{
public:
    WriteAccessor(BaseData* data_, py::object fct_) : data(data_), fct(fct_){}

    BaseData* data {nullptr};
    py::object wrap;
    py::object fct;
};

void moduleAddDataAsString(py::module& m)
{
    py::class_<DataAsString, BaseData, raw_ptr<DataAsString>> s(m, "DataString");
    s.def("__getitem__", [](BaseData& self, py::size_t index) -> py::object
    {
        auto nfo = self.getValueTypeInfo();
        return py::str(&nfo->getTextValue(self.getValueVoidPtr(),0).at(index),1);
    });
}



void moduleAddDataContainer(py::module& m)
{

    py::class_<DataContainer, BaseData, raw_ptr<DataContainer>> p(m, "DataContainer",
                                                                      py::buffer_protocol());

    p.def("__getitem__", [](DataContainer* self, py::size_t index) -> py::object
    {
        py::array a = getPythonArrayFor(self);
        py::buffer_info parentinfo = a.request();
    });

    p.def("__setitem__", [](DataContainer* self, size_t& index, py::object& value)
    {
        scoped_writeonly_access access(self);
        setItem(getPythonArrayFor(self), py::slice(index, index+1, 1), value);
        return py::none();
    });

    p.def("__setitem__", [](DataContainer* self, py::slice& slice, py::object& value) -> py::object
    {
        scoped_writeonly_access access(self);

        setItem(getPythonArrayFor(self), slice, value);
        return py::none();
    });

    p.def("__setitem__", [](DataContainer* self, py::tuple key, py::object& value)
    {
        scoped_writeonly_access access(self);

        py::array a=getPythonArrayFor(self);
        py::slice s0=toSlice(key[0]);
        py::slice s1=toSlice(key[1]);

        setItem2D(a, s0, s1, value);

        return py::none();
    });

    p.def("apply", [](DataContainer* self, py::function f)
    {
        scoped_write_access access(self);
        py::array a = getPythonArrayFor(self);

        auto aa=a.mutable_unchecked<double>();
        for(auto i=0;i<aa.shape(0);++i){
            for(auto j=0;j<aa.shape(1);++j){
                aa(i,j) = py::cast<double>(f(i,j, aa(i,j)));
            }
        }
    });

    p.def("__str__", [](BaseData* self)
    {
        return py::str(convertToPython(self));
    });

    p.def("__repr__", [](BaseData* self)
    {
        return py::repr(convertToPython(self));
    });

    p.def("tolist", [](DataContainer* self){
        return convertToPython(self);
    });

    p.def("toarray", [](DataContainer* self){
        auto capsule = py::capsule(new Base::SPtr(self->getOwner()));
        py::buffer_info ninfo = toBufferInfo(*self);
        py::array a(pybind11::dtype(ninfo), ninfo.shape,
                    ninfo.strides, ninfo.ptr, capsule);
        a.attr("flags").attr("writeable") = false;
        return a;
    });

    p.def("writeable", [](DataContainer* self, py::object f) -> py::object
    {
        if(self!=nullptr)
            return py::cast(new WriteAccessor(self, f));

        return py::none();
    });

    p.def("writeable", [](DataContainer* self) -> py::object
    {
        if(self!=nullptr)
            return py::cast(new WriteAccessor(self, py::none()));

        return py::none();
    });

    p.def("__iadd__", [](DataContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read+write access we
        /// use the scoped_write_access object.
        scoped_write_access access(self);

        /// Search if this container object has a PythonArray in the case, if this is not the case
        /// create one.
        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataContainer>(value))
            value = getPythonArrayFor(py::cast<BaseData*>(value));

        /// Returns a new reference on the result.
        /// We don't want to keep this reference so we decref it to avoid memory leak.
        Py_DECREF(PyNumber_InPlaceAdd(p.ptr(), value.ptr()));

        /// Instead, returns the self object as we are in an in-place add operator.
        return self;
    });

    p.def("__add__", [](DataContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read only access we
        /// use the scoped_read_access object. This imply that the data will updates the content
        /// of this object.
        scoped_read_access access(self);
        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataContainer>(value))
            value = getPythonArrayFor(py::cast<BaseData*>(value));

        return py::reinterpret_steal<py::object>(PyNumber_Add(p.ptr(), value.ptr()));
    });

    p.def("__isub__", [](DataContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read+write access we
        /// use the scoped_write_access object.
        scoped_write_access access(self);

        /// Search if this container object has a PythonArray in the case, if this is not the case
        /// create one.
        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataContainer>(value))
            value = getPythonArrayFor(py::cast<BaseData*>(value));

        /// Returns a new reference on the result.
        /// We don't want to keep this reference so we decref it to avoid memory leak.
        Py_DECREF(PyNumber_InPlaceSubtract(p.ptr(), value.ptr()));

        /// Instead, returns the self object as we are in an in-place add operator.
        return self;
    });

    p.def("__sub__", [](DataContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read only access we
        /// use the scoped_read_access object. This imply that the data will updates the content
        /// of this object.
        scoped_read_access access(self);
        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataContainer>(value))
            value = getPythonArrayFor(py::cast<BaseData*>(value));

        return py::reinterpret_steal<py::object>(PyNumber_Subtract(p.ptr(), value.ptr()));
    });

    p.def("__imul__", [](DataContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read+write access we
        /// use the scoped_write_access object.
        scoped_write_access access(self);

        /// Search if this container object has a PythonArray in the case, if this is not the case
        /// create one.
        if( !hasArrayFor(self) )
            throw py::type_error("NOOO");

        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataContainer>(value))
        {
            value = getPythonArrayFor(py::cast<BaseData*>(value));
        }

        /// Returns a new reference on the result.
        /// We don't want to keep this reference so we decref it to avoid memory leak.
        Py_DECREF(PyNumber_InPlaceMultiply(p.ptr(), value.ptr()));

        /// Instead, returns the self object as we are in an in-place add operator.
        return self;
    });

    p.def("__mul__", [](DataContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read only access we
        /// use the scoped_read_access object. This imply that the data will updates the content
        /// of this object.
        scoped_read_access access(self);
        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataContainer>(value))
            value = getPythonArrayFor(py::cast<BaseData*>(value));

        return py::reinterpret_steal<py::object>(PyNumber_Multiply(p.ptr(), value.ptr()));
    });
}

void moduleAddWriteAccessor(py::module& m)
{
    py::class_<WriteAccessor> wa(m, "DataContainerContextManager");
    wa.def("__enter__", [](WriteAccessor& wa)
    {
        wa.data->beginEditVoidPtr();
        py::array mainbuffer = getPythonArrayFor(wa.data);
        py::buffer_info info = mainbuffer.request();
        wa.wrap = py::array(pybind11::dtype(info), info.shape,
                            info.strides, info.ptr, mainbuffer);

        if(!wa.fct.is_none())
            wa.wrap = wa.fct(wa.wrap);

        return wa.wrap;
    });

    wa.def("__exit__",
           [](WriteAccessor& wa, py::object type, py::object value, py::object traceback)
    {
        SOFA_UNUSED(type);
        SOFA_UNUSED(value);
        SOFA_UNUSED(traceback);
        wa.wrap.attr("flags").attr("writeable") = false;
        wa.data->endEditVoidPtr();
    });
}

}
