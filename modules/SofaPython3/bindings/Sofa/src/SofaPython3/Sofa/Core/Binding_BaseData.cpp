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
#include <pybind11/eval.h>

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

template<class Array, typename Value>
void setValueArray1D(Array p,
                     const py::slice& slice,
                     const Value& v)
{
    size_t start, stop, step, slicelength;
    if (!slice.compute(p.shape(0), &start, &stop, &step, &slicelength))
        throw py::error_already_set();

    for (size_t i = 0; i < slicelength; ++i) {
        p(start) = v;
        start += step;
    }
}

template<class Array, typename Value>
void setValueArray2D(Array p,
                     const py::slice& slice,
                     const Value& v)
{
    size_t start, stop, step, slicelength;
    if (!slice.compute(p.shape(0), &start, &stop, &step, &slicelength))
        throw py::error_already_set();

    for (size_t i = 0; i < slicelength; ++i, start+=step) {
        for(size_t j=0; j<p.shape(1);++j){
            p(start, j) = v;
        }
    }
}

template<class Array, typename Value>
void setItem2DTyped(Array a, py::slice slice, Value dvalue)
{
    size_t start, stop, step, slicelength;
    if (!slice.compute(a.shape(0), &start, &stop, &step, &slicelength))
        throw py::error_already_set();

    for(size_t i=0;i<slicelength;++i, start+=step)
        for(size_t j=0;j<a.shape(1);++j)
            a(start, j) = dvalue;
}

template<class Array, typename Value>
void setItem2DTyped(Array a, py::slice sliceI, py::slice sliceJ, Value dvalue)
{
    size_t startI, stopI, stepI, slicelengthI;
    if (!sliceI.compute(a.shape(0), &startI, &stopI, &stepI, &slicelengthI))
        throw py::error_already_set();

    size_t startJ, stopJ, stepJ, slicelengthJ;
    if (!sliceJ.compute(a.shape(1), &startJ, &stopJ, &stepJ, &slicelengthJ))
        throw py::error_already_set();

    for(size_t i=0;i<slicelengthI;++i, startI+=stepI)
    {
        for(size_t j=0, itJ=startJ;j<slicelengthJ;++j, itJ+=stepJ)
        {
            a(startI, itJ) = dvalue;
        }
    }
}

void setItem2D(py::array a, py::slice slice, py::object o)
{
    if(a.request().format=="d")
        setItem2DTyped(a.mutable_unchecked<double, 2>(), slice, py::cast<double>(o));
    else if(a.request().format=="f")
        setItem2DTyped(a.mutable_unchecked<float, 2>(), slice, py::cast<float>(o));
    else
        throw py::type_error("Invalid type");
}

void setItem2D(py::array a, const py::slice& slice, const py::slice& slice1, py::object o)
{
    if(a.request().format=="d")
        setItem2DTyped(a.mutable_unchecked<double, 2>(), slice, slice1, py::cast<double>(o));
    else if(a.request().format=="f")
        setItem2DTyped(a.mutable_unchecked<float, 2>(), slice, slice1, py::cast<float>(o));
    else
        throw py::type_error("Invalid type");
}

template<class Array, typename Value>
void setItem1DTyped(Array a, py::slice slice, Value dvalue)
{
    size_t start, stop, step, slicelength;
    if (!slice.compute(a.shape(0), &start, &stop, &step, &slicelength))
        throw py::error_already_set();

    for(size_t i=0;i<slicelength;++i, start+=step)
        a(start) = dvalue;
}

void setItem1D(py::array a, py::slice slice, py::object o)
{
    if(a.request().format=="d")
        setItem1DTyped(a.mutable_unchecked<double, 1>(), slice, py::cast<double>(o));
    else if(a.request().format=="f")
        setItem1DTyped(a.mutable_unchecked<float, 1>(), slice, py::cast<float>(o));
    else
        throw py::type_error("Invalid type");
}

void setItem(py::array a, py::slice slice, py::object value)
{
    if(a.ndim()>2)
        throw py::index_error("DataContainer can only operate on 1 or 2D array.");

    else if(a.ndim()==1)
        setItem1D(a, slice, value);

    else if(a.ndim()==2)
        setItem2D(a, slice, value);
}

py::slice toSlice(const py::object& o)
{
    if( py::isinstance<py::slice>(o))
        return py::cast<py::slice>(o);

    size_t v = py::cast<size_t>(o);
    return py::slice(v,v+1,1);
}

void moduleAddDataAsContainer(py::module& m)
{
    py::class_<DataAsContainer, BaseData, raw_ptr<DataAsContainer>> p(m, "DataContainer",
                                                                      py::buffer_protocol());

    p.def("__getitem__", [](DataAsContainer* self, py::size_t index) -> py::object
    {
        py::array a = getPythonArrayFor(self);
        py::buffer_info parentinfo = a.request();
    });

    p.def("toarray", [](DataAsContainer* self){
        auto capsule = py::capsule(new Base::SPtr(self->getOwner()));
        py::buffer_info ninfo = toBufferInfo(*self);
        py::array a(pybind11::dtype(ninfo), ninfo.shape,
                    ninfo.strides, ninfo.ptr, capsule);
        a.attr("flags").attr("writeable") = false;
        return a;
    });

    p.def("__setitem__", [](DataAsContainer* self, size_t& index, py::object& value)
    {
        scoped_writeonly_access access(self);
        setItem(getPythonArrayFor(self), py::slice(index, index+1, 1), value);
        return py::none();
    });

    p.def("__setitem__", [](DataAsContainer* self, py::slice& slice, py::object& value) -> py::object
    {
        scoped_writeonly_access access(self);

        setItem(getPythonArrayFor(self), slice, value);
        return py::none();
    });

    p.def("__setitem__", [](DataAsContainer* self, py::tuple key, py::object& value)
    {
        scoped_writeonly_access access(self);

        py::array a=getPythonArrayFor(self);
        py::slice s0=toSlice(key[0]);
        py::slice s1=toSlice(key[1]);

        setItem2D(a, s0, s1, value);

        return py::none();
    });

    p.def("apply", [](DataAsContainer* self, py::function f)
    {
        scoped_write_access access(self);
        py::array a = getPythonArrayFor(self);

        auto aa=a.mutable_unchecked<double>();
        for(size_t i=0;i<aa.shape(0);++i){
            for(size_t j=0;j<aa.shape(1);++j){
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

    p.def("tolist", [](DataAsContainer* self){
        return convertToPython(self);
    });

    p.def("writeable", [](DataAsContainer* self, py::object f) -> py::object
    {
        if(self!=nullptr)
            return py::cast(new WriteAccessor(self, f));

        return py::none();
    });

    p.def("writeable", [](DataAsContainer* self) -> py::object
    {
        if(self!=nullptr)
            return py::cast(new WriteAccessor(self, py::none()));

        return py::none();
    });

    p.def("__iadd__", [](DataAsContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read+write access we
        /// use the scoped_write_access object.
        scoped_write_access access(self);

        /// Search if this container object has a PythonArray in the case, if this is not the case
        /// create one.
        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataAsContainer>(value))
            value = getPythonArrayFor(py::cast<BaseData*>(value));

        /// Returns a new reference on the result.
        /// We don't want to keep this reference so we decref it to avoid memory leak.
        Py_DECREF(PyNumber_InPlaceAdd(p.ptr(), value.ptr()));

        /// Instead, returns the self object as we are in an in-place add operator.
        return self;
    });

    p.def("__add__", [](DataAsContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read only access we
        /// use the scoped_read_access object. This imply that the data will updates the content
        /// of this object.
        scoped_read_access access(self);
        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataAsContainer>(value))
            value = getPythonArrayFor(py::cast<BaseData*>(value));

        return py::reinterpret_steal<py::object>(PyNumber_Add(p.ptr(), value.ptr()));
    });

    p.def("__isub__", [](DataAsContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read+write access we
        /// use the scoped_write_access object.
        scoped_write_access access(self);

        /// Search if this container object has a PythonArray in the case, if this is not the case
        /// create one.
        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataAsContainer>(value))
            value = getPythonArrayFor(py::cast<BaseData*>(value));

        /// Returns a new reference on the result.
        /// We don't want to keep this reference so we decref it to avoid memory leak.
        Py_DECREF(PyNumber_InPlaceSubtract(p.ptr(), value.ptr()));

        /// Instead, returns the self object as we are in an in-place add operator.
        return self;
    });

    p.def("__sub__", [](DataAsContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read only access we
        /// use the scoped_read_access object. This imply that the data will updates the content
        /// of this object.
        scoped_read_access access(self);
        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataAsContainer>(value))
            value = getPythonArrayFor(py::cast<BaseData*>(value));

        return py::reinterpret_steal<py::object>(PyNumber_Subtract(p.ptr(), value.ptr()));
    });

    p.def("__imul__", [](DataAsContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read+write access we
        /// use the scoped_write_access object.
        scoped_write_access access(self);

        /// Search if this container object has a PythonArray in the case, if this is not the case
        /// create one.
        if( !hasArrayFor(self) )
            throw py::type_error("NOOO");

        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataAsContainer>(value))
        {
            value = getPythonArrayFor(py::cast<BaseData*>(value));
        }

        /// Returns a new reference on the result.
        /// We don't want to keep this reference so we decref it to avoid memory leak.
        Py_DECREF(PyNumber_InPlaceMultiply(p.ptr(), value.ptr()));

        /// Instead, returns the self object as we are in an in-place add operator.
        return self;
    });

    p.def("__mul__", [](DataAsContainer* self, py::object value)
    {
        /// Acquire an access to the underlying data. As this is a read only access we
        /// use the scoped_read_access object. This imply that the data will updates the content
        /// of this object.
        scoped_read_access access(self);
        py::array p = getPythonArrayFor(self);

        if(py::isinstance<DataAsContainer>(value))
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

void moduleAddBaseData(py::module& m)
{
    py::class_<BaseData, raw_ptr<BaseData>> p(m, "Data");
    p.def("setName", &BaseData::setName);
    p.def("getName", &BaseData::getName);
    p.def("getCounter", [](BaseData& d){ return d.getCounter(); } );
    p.def("getHelp", &BaseData::getHelp);
    p.def("unset", &BaseData::unset);
    p.def("getOwner", &BaseData::getOwner);

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

    p.def("getPath", [](BaseData& self)
    {
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
