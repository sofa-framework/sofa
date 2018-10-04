#include "Binding_Base.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

#include <sofa/core/objectmodel/BaseLink.h>
using sofa::core::objectmodel::BaseLink;

#include "Binding_BaseData.h"
using sofa::defaulttype::AbstractTypeInfo;

#include <pybind11/numpy.h>

#include <sofa/helper/accessor.h>
using sofa::helper::WriteOnlyAccessor;

std::map<void*, py::object>& getObjectCache()
{
    static std::map<void*, py::object>* s_objectcache {nullptr} ;
    if(!s_objectcache)
    {
        std::cout << "CREATE A NEW CACHE" << std::endl ;
        s_objectcache=new std::map<void*, py::object>();
    }
    return *s_objectcache;
}

void trimCache()
{
    auto& memcache = getObjectCache();
    if(memcache.size() > 1000)
    {
        std::cout << "flushing the cache (it is too late to implement LRU)" << std::endl ;
        memcache.clear();
    }
}

py::buffer_info toBufferInfo(BaseData& m)
{
    const AbstractTypeInfo& nfo { *m.getValueTypeInfo() };

    const char* format = nullptr;
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

    void* ptr = const_cast<void*>(nfo.getValuePtr(m.getValueVoidPtr()));
    if(rows == 1 && nfo.FixedSize()){
        std::cout << "WEIRD BUFFER" << std::endl ;
        std::cout << "COLD" << cols << " d " << datasize << std::endl;
        return py::buffer_info(
                    ptr, /* Pointer to buffer */
                    datasize,                              /* Size of one scalar */
                    format,                                /* Python struct-style format descriptor */
                    1,                                     /* Number of dimensions */
        { cols },                              /* Buffer dimensions */
        { datasize }                           /* Strides (in bytes) for each index */

                    );
    }
    py::buffer_info ninfo(
                ptr,  /* Pointer to buffer */
                datasize,                              /* Size of one scalar */
                format,                                /* Python struct-style format descriptor */
                2,                                     /* Number of dimensions */
    { rows, cols },                        /* Buffer dimensions */
    { datasize * cols ,    datasize }                         /* Strides (in bytes) for each index */

                );
    return ninfo;
}

py::object convertToPython(BaseData* d)
{
    const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };
    if(hasArrayFor(d))
        return getPythonArrayFor(d);

    if(nfo.Container())
    {
        size_t dim0 = nfo.size(d->getValueVoidPtr())/nfo.size();
        size_t dim1 = nfo.size();
        py::list list;
        for(size_t i=0;i<dim0;i++)
        {
            py::list list1;
            for(size_t j=0;j<dim1;j++)
            {
                if(nfo.Integer())
                    list1.append(nfo.getIntegerValue(d->getValueVoidPtr(),i*dim1+j));
                if(nfo.Scalar())
                    list1.append(nfo.getScalarValue(d->getValueVoidPtr(),i*dim1+j));
                if(nfo.Text())
                    list1.append(nfo.getTextValue(d->getValueVoidPtr(),0));
            }
            list.append(list1);
        }
        return list;
    }

    if(nfo.Integer())
        return py::cast(nfo.getIntegerValue(d->getValueVoidPtr(), 0));
    if(nfo.Text())
        return py::cast(d->getValueString());
    if(nfo.Scalar())
        return py::cast(nfo.getScalarValue(d->getValueVoidPtr(), 0));

    return py::cast(d->getValueString());
}

bool hasArrayFor(BaseData* d)
{
    auto& memcache = getObjectCache();
    return memcache.find(d) != memcache.end();
}

py::object getPythonArrayFor(BaseData* d)
{
    auto& memcache = getObjectCache();
    if(memcache.find(d) == memcache.end())
    {
        auto capsule = py::capsule(d->getOwner(),
                                   [](void *v){});

        py::buffer_info ninfo = toBufferInfo(*d);
        py::array a(pybind11::dtype(ninfo), ninfo.shape,
                    ninfo.strides, ninfo.ptr, capsule);

        memcache[d] = a;
        std::cout << "ADDING AN ARRAY SIZE: " << ninfo.ndim << std::endl;
    }

    return memcache[d];
}



/// Make a python version of our BaseData.
/// If possible the data is exposed as a numpy.array to minmize copy and data conversion
py::object toPython(BaseData* d, bool writeable)
{
    const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };

    /// In case the data is a container with a simple layout
    /// we can expose the field as a numpy.array
    if(nfo.Container() && nfo.SimpleLayout())
    {
        if(!writeable)
            return py::cast(reinterpret_cast<DataAsContainer*>(d));

        return getPythonArrayFor(d);
    }

    /// If this is not the case we returns converted datas
    return convertToPython(d);
}

void copyFromListScalar(BaseData& d, const AbstractTypeInfo& nfo, const py::list& l)
{
    /// Check if the data is a single dimmension or not.
    py::buffer_info dstinfo = toBufferInfo(d);

    if(dstinfo.ndim>2)
        throw py::index_error("Invalid number of dimension only 1 or 2 dimensions are supported).");

    if(dstinfo.ndim==1)
    {
        void* ptr = d.beginEditVoidPtr();

        if( dstinfo.shape[0] != l.size())
            nfo.setSize(ptr, l.size());
        for(size_t i=0;i<l.size();++i)
        {
            nfo.setScalarValue(ptr, i, py::cast<double>(l[i]));
        }
        d.endEditVoidPtr();
        return;
    }
    void* ptr = d.beginEditVoidPtr();
    if( dstinfo.shape[0] != l.size())
        nfo.setSize(ptr, l.size());

    for(size_t i=0;i<dstinfo.shape[0];++i)
    {
        py::list ll = l[i];
        for(size_t j=0;j<dstinfo.shape[1];++j)
        {
            nfo.setScalarValue(ptr, i*dstinfo.shape[1]+j, py::cast<double>(ll[j]));
        }
    }
    d.endEditVoidPtr();
    return;
}

void fromPython(BaseData* d, const py::object& o)
{
    const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };
    if(!nfo.Container())
    {
        if(nfo.Integer())
            nfo.setIntegerValue(d->beginEditVoidPtr(), 0, py::cast<int>(o));
        if(nfo.Text())
            nfo.setTextValue(d->beginEditVoidPtr(), 0, py::cast<py::str>(o));
        if(nfo.Scalar())
            nfo.setScalarValue(d->beginEditVoidPtr(), 0, py::cast<double>(o));
        d->endEditVoidPtr();
    }

    if(nfo.Scalar())
        return copyFromListScalar(*d, nfo,o);

    msg_error("SofaPython3") << "binding problem";
}

py::object BindingBase::GetAttr(Base& self, const std::string& s)
{
    /// I'm not sure implicit behavior is nice but we could do:
    ///    - The attribute is a data,
    ///         returns it if it is a container
    ///         returns the value/specific binding otherwise
    ///    - The attribute is a link, return it.
    ///    - The attribute is an object or a child return it.
    ///    - The attribute is not existing:
    ///                raise an exception or search using difflib for close match.
    BaseData* d = self.findData(s);
    if(d!=nullptr)
        return toPython(d);

    if( s == "__data__")
        return py::cast( DataDict(&self) );

    throw py::attribute_error(s);
}

void BindingBase::SetAttr(py::object self, const std::string& s, pybind11::object &value)
{
    /// I'm not sure implicit behavior is nice but we could do:
    ///    - The attribute is a data, set its value.
    ///          If the data is a container...check dimmensions and do type coercion.
    ///    - The attribute is a link, set its value.
    ///    - The attribute is an object or a child, raise an exception.
    ///    - The attribute is not existing, add it has data with type deduced from value ?
    Base& self_base = py::cast<Base&>(self);
    BaseData* d = self_base.findData(s);

    std::cout << "SETTER " << std::endl ;
    if(d!=nullptr)
    {
        const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };

        /// We go for the container path.
        if(nfo.Container())
        {
            fromPython(d,value);
            return;
        }
        fromPython(d, value);
        return;
    }

    BaseLink* l = self_base.findLink(s);
    if(l!=nullptr)
    {
        return;
    }

    /// We are falling back to dynamically adding the objet into the object dict.
    py::dict t = self.attr("__dict__");
    if(!t.is_none())
    {
        t[s.c_str()] = value;
        return;
    }

    /// Well this should never happen unless there is no __dict__
    throw py::attribute_error();
}

std::ostream& operator<<(std::ostream& out, const py::buffer_info& p)
{
    out << "buffer{"<< p.format << ", " << p.ndim << ", " << p.shape[0];
    if(p.ndim==2)
        out << ", " << p.shape[1];
    out << ", " << p.size << "}";
    return out;
}

template<typename T>
void copyScalar(BaseData* a, const AbstractTypeInfo& nfo, py::array_t<T, py::array::c_style> src)
{
    void* ptr = a->beginEditVoidPtr();

    auto r = src.unchecked();
    for (ssize_t i = 0; i < r.shape(0); i++)
        for (ssize_t j = 0; j < r.shape(1); j++)
        {
            nfo.setScalarValue( ptr, i*r.shape(1)+j, r(i,j) );
        }
    a->endEditVoidPtr();
}

void BindingBase::SetAttrFromArray(py::object self, const std::string& s, const py::array& value)
{
    /// I'm not sure implicit behavior is nice but we could do:
    ///    - The attribute is a data, set its value.
    ///          If the data is a container...check dimmensions and do type coercion.
    ///    - The attribute is a link, set its value.
    ///    - The attribute is an object or a child, raise an exception.
    ///    - The attribute is not existing, add it has data with type deduced from value ?
    Base& self_base = py::cast<Base&>(self);
    BaseData* d = self_base.findData(s);

    if(d!=nullptr)
    {
        const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };

        /// We go for the container path.
        if(nfo.Container())
        {
            py::array dst = getPythonArrayFor(d);
            py::buffer_info dstinfo = dst.request();

            py::array src = value;
            py::buffer_info srcinfo = src.request();
            if( srcinfo.ptr == dstinfo.ptr )
            {
                d->beginEditVoidPtr();
                d->endEditVoidPtr();
                return;
            }

            /// Invalid dimmensions
            if( srcinfo.ndim != dst.ndim() )
                throw py::type_error("Invalid dimension");


            bool needResize = false;
            size_t resizeShape;
            size_t srcSize = 1;
            for(size_t i=0;i<srcinfo.ndim;++i){
                srcSize *= srcinfo.shape[i];
                if( srcinfo.shape[i] != dstinfo.shape[i])
                {
                    resizeShape = i;
                    needResize = true;
                }
            }

            if(nfo.FixedSize() && needResize)
                throw py::index_error("The destination is not large enough and cannot be resized. Please clamp the source data set before setting.");

            if(resizeShape != 0 && needResize)
                throw py::index_error("The destination can only be resized on the first dimension. ");

            if(needResize)
            {
                nfo.setSize(d->beginEditVoidPtr(), srcSize);
            }

            bool sameDataType = (srcinfo.format == dstinfo.format);
            if(sameDataType && (nfo.BaseType()->FixedSize() || nfo.SimpleCopy()))
            {
                //std::cout << "SetAttrFromArray is going the fast way" << s << std::endl;

                memcpy(nfo.getValuePtr(d->beginEditVoidPtr()), srcinfo.ptr, srcSize*srcinfo.itemsize);
                d->endEditVoidPtr();
                return;
            }

            /// In this case we go for the fast path.
            if(nfo.SimpleLayout())
            {
                if(srcinfo.format=="d")
                    copyScalar<double>(d, nfo, src);
                else if(srcinfo.format=="f")
                    copyScalar<float>(d, nfo, src);
                return;
            }

            std::cout << "SetAttrFromArray is going the slow way" << s << std::endl;
            fromPython(d, value);
        }
        std::cout << "SetAttrFromArray is GOING THE SUPER SLOW PATH" << s << std::endl ;
        fromPython(d, value);
        return;
    }


    std::cout << "SETTING TO A DICT" << std::endl ;
    /// We are falling back to dynamically adding the objet into the object dict.
    py::dict t = self.attr("__dict__");
    if(!t.is_none())
    {
        t[s.c_str()] = value;
        return;
    }

    /// Well this should never happen unless there is no __dict__
    throw py::attribute_error();
}

void moduleAddDataDict(py::module& m)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////
    /// DataDict binding
    ////////////////////////////////////////////////////////////////////////////////////////////////
    py::class_<DataDict> d(m, "DataDict",
                           R"(DataDict exposes the data of a sofa object in a way similar to a normal python dictionnary.
                           Eg:
                           for k,v in anObject.__data__.items():
                           print("Data name :"+k+" value:" +str(v)))
                           )");
    d.def("__len__", [](DataDict& self)
    {
        return self.owner->getDataFields().size();
    });


    d.def("__getitem__",[](DataDict& self, const size_t& i)
    {
        const Base::VecData& vd = self.owner->getDataFields();
        if(i > vd.size())
            throw py::index_error(std::to_string(i));
        return toPython(vd[i]);
    });

    d.def("__getitem__",
          [](DataDict& self, const std::string& s) -> py::object
    {
        std::cout << "Get ITEM" << s << std::endl ;

        BaseData* d = self.owner->findData(s);
        if(d!=nullptr)
        {
            const AbstractTypeInfo& nfo{ *(d->getValueTypeInfo()) };

            if(nfo.Container())
            {
                py::array a = py::array(toBufferInfo(*d));
                return a;
            }
            if(nfo.Text())
                return py::cast(reinterpret_cast<DataAsString*>(d));
            return py::cast(d);
        }
        throw py::attribute_error(s);
    });


    ////////////////////////////////////////////////////////////////////////////////////////////////
    d.def("__setitem__",[](DataDict& d, const std::string& s, py::object v)
    {
        std::cout << "SET ITEM TO: " << s << std::endl;
        return 0.0;
    });

    d.def("__iter__", [](DataDict& d)
    {
        return DataDictIterator(d.owner, true, false);
    });
    d.def("keys", [](DataDict& d)
    {
        return DataDictIterator(d.owner, true, false);
    });
    d.def("values", [](DataDict& d)
    {
        return DataDictIterator(d.owner, false, true);
    });
    d.def("items", [](DataDict& d)
    {
        return DataDictIterator(d.owner, true, true);
    });
}


void moduleAddDataDictIterator(py::module &m)
{
    py::class_<DataDictIterator> ddi(m, "DataDictIterator");
    ddi.def("__iter__", [](DataDictIterator& d)
    {
        return d;
    });
    ddi.def("__next__", [](DataDictIterator& d) -> py::object
    {
        if(d.index>=d.owner->getDataFields().size())
            throw py::stop_iteration();

        BaseData* data = d.owner->getDataFields()[d.index++];
        if(!d.key)
            return toPython(data);

        if(!d.value)
            return py::cast(data->getName());

        py::tuple t {2};
        t[0] = data->getName();
        t[1] = toPython(data);
        return t;
    });
}


void moduleAddBase(py::module &m)
{
    py::class_<Base, Base::SPtr> p(m, "Base");
    p.def("getData", [](Base& self, const std::string& s) -> py::object
    {
        BaseData* d = self.findData(s);
        if(d!=nullptr)
        {
            return py::cast(d);
        }
        return py::none();
    });

    p.def("__getattr__", &BindingBase::GetAttr);
    p.def("__setattr__", [](py::object self, const std::string& s, py::object value)
    {
        if(py::isinstance<DataAsContainer>(value))
        {
            BaseData* data = py::cast<BaseData*>(value);
            py::array a = getPythonArrayFor(data);
            BindingBase::SetAttrFromArray(self,s, a);
            return;
        }

        if(py::isinstance<py::array>(value))
        {
            BindingBase::SetAttrFromArray(self,s, py::cast<py::array>(value));
            return;
        }

        BindingBase::SetAttr(self,s,value);
    });
}
