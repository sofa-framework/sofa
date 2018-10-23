
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

#include <sofa/core/objectmodel/BaseLink.h>
using sofa::core::objectmodel::BaseLink;

#include <sofa/helper/Factory.h>
#include <sofa/helper/Factory.inl>

#include <sofa/helper/accessor.h>
using sofa::helper::WriteOnlyAccessor;

#include "Binding_Base.h"
#include "Binding_BaseData.h"
#include "Binding_DataContainer.h"

namespace sofapython3
{
template<typename DataType>
class DataCreator : public sofa::helper::BaseCreator<sofa::core::objectmodel::BaseData>
{
public:
    virtual sofa::core::objectmodel::BaseData* createInstance(sofa::helper::NoArgument) override { return new sofa::core::objectmodel::Data<DataType>(); }
    virtual const std::type_info& type() override { return typeid(sofa::core::objectmodel::BaseData);}
};

typedef sofa::helper::Factory< std::string, BaseData> PSDEDataFactory;

PSDEDataFactory* getFactoryInstance(){
    static PSDEDataFactory* s_localfactory = nullptr ;
    if (s_localfactory == nullptr)
    {
        // helper vector style containers
        std::string containers[] = {"vector", "ResizableExtVector"};

        s_localfactory = new PSDEDataFactory();
        // Scalars
        s_localfactory->registerCreator("string", new DataCreator<std::string>());
        s_localfactory->registerCreator("float", new DataCreator<float>());
        s_localfactory->registerCreator("double", new DataCreator<double>());
        s_localfactory->registerCreator("bool", new DataCreator<bool>());
        s_localfactory->registerCreator("int", new DataCreator<int>());

        // vectors
        s_localfactory->registerCreator(
                    "Vec2d", new DataCreator<sofa::defaulttype::Vec2d>());
        s_localfactory->registerCreator(
                    "Vec3d", new DataCreator<sofa::defaulttype::Vec3d>());
        s_localfactory->registerCreator(
                    "Vec4d", new DataCreator<sofa::defaulttype::Vec4d>());
        s_localfactory->registerCreator(
                    "Vec6d", new DataCreator<sofa::defaulttype::Vec6d>());
        s_localfactory->registerCreator(
                    "Vec2f", new DataCreator<sofa::defaulttype::Vec2f>());
        s_localfactory->registerCreator(
                    "Vec3f", new DataCreator<sofa::defaulttype::Vec3f>());
        s_localfactory->registerCreator(
                    "Vec4f", new DataCreator<sofa::defaulttype::Vec4f>());
        s_localfactory->registerCreator(
                    "Vec6f", new DataCreator<sofa::defaulttype::Vec6f>());
    }
    return s_localfactory ;
}



py::object BindingBase::GetAttr(Base* self, const std::string& s, bool doThrowException)
{
    /// I'm not sure implicit behavior is nice but we could do:
    ///    - The attribute is a data,
    ///         returns it if it is a container
    ///         returns the value/specific binding otherwise
    ///    - The attribute is a link, return it.
    ///    - The attribute is an object or a child return it.
    ///    - The attribute is not existing:
    ///                raise an exception or search using difflib for close match.
    if(self==nullptr)
        throw py::attribute_error("None object used to get an attribute.");

    BaseData* d = self->findData(s);
    if(d!=nullptr)
        return toPython(d);

    /// I'm using this syntax as this make the 'l' variable local.
    if(BaseLink* l = self->findLink(s))
    {
        return py::cast(l->getLinkedBase());
    }

    if( s == "__data__")
        return py::cast( DataDict(self) );

    if(doThrowException)
        throw py::attribute_error(s);

    return py::none();
}

void BindingBase::SetAttr(py::object self, const std::string& s, py::object value, bool withDict)
{
    Base* self_d = py::cast<Base*>(self);
    BaseData* d = self_d->findData(s);

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

    BaseLink* l = self_d->findLink(s);
    if(l!=nullptr)
    {
        return;
    }

    std::string id="none";
    if(py::isinstance<py::float_>(value))
        id = "double";
    else if(py::isinstance<py::int_>(value))
        id = "int";
    else if(py::isinstance<py::str>(value))
        id = "string";

    BaseData* data = getFactoryInstance()->createObject(id, sofa::helper::NoArgument());
    if(data)
    {
        data->setName(s);
        data->setGroup("Custom");
        data->setDisplayed(true);
        data->setPersistent(true);
        self_d->addData(data, s);
        fromPython(data, value);

        return;
    }

    /// We are falling back to dynamically adding the objet into the object dict.
    if(withDict)
    {
        py::dict t = self.attr("__dict__");
        if(!t.is_none())
        {
            t[s.c_str()] = value;
            return;
        }
    }

    /// Well this should never happen unless there is no __dict__
    throw py::attribute_error("Unable to set attribute '"+s+"', unknow data type");
}

void BindingBase::SetAttr(Base& self, const std::string& s, py::object value)
{
    BaseData* d = self.findData(s);

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

    BaseLink* l = self.findLink(s);
    if(l!=nullptr)
    {
        return;
    }

    /// Well this should never happen unless there is no __dict__
    throw py::attribute_error();
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
                /// Increment the change counter so other data field can keep track of
                /// what happens.
                d->beginEditVoidPtr();
                d->endEditVoidPtr();
                return;
            }

            /// Invalid dimmensions
            if( srcinfo.ndim != dst.ndim() )
                throw py::type_error("Invalid dimension");

            bool needResize = false;
            size_t resizeShape=0;
            size_t srcSize = 1;
            for(auto i=0;i<srcinfo.ndim;++i){
                srcSize *= srcinfo.shape[i];
                if( srcinfo.shape[i] != dstinfo.shape[i])
                {
                    resizeShape = i;
                    needResize = true;
                }
            }

            if(needResize)
            {
                if(nfo.FixedSize())
                    throw py::index_error("The destination is not large enough and cannot be resized. Please clamp the source data set before setting.");

                if(resizeShape != 0)
                    throw py::index_error("The destination can only be resized on the first dimension. ");

                /// Change the allocated memory of the data field, then update the
                /// cache entry so keep up with the changes. As we use dstinfo in the following
                /// we also update it.
                nfo.setSize(d->beginEditVoidPtr(), srcSize);
                dst = resetArrayFor(d);
                dstinfo=dst.request();
            }

            bool sameDataType = (srcinfo.format == dstinfo.format);
            if(sameDataType && (nfo.BaseType()->FixedSize() || nfo.SimpleCopy()))
            {
                scoped_writeonly_access guard(d);
                memcpy(dstinfo.ptr, srcinfo.ptr, srcSize*dstinfo.itemsize);
                return;
            }

            /// In this case we go for the fast path.
            if(nfo.SimpleLayout())
            {
                if(srcinfo.format=="d")
                    copyScalar<double>(d, nfo, src);
                else if(srcinfo.format=="f")
                    copyScalar<float>(d, nfo, src);
                else
                    std::cout << "SetAttrFromArray :: unsupported fileformat" << std::endl ;
            }
        }
        fromPython(d, value);
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
    py::class_<Base, Base::SPtr> p(m, "Base", py::dynamic_attr(),
                                   R"(Ceci est la documentation de la méthode getData
                                     Je vois pas getData()
                                   )");

    p.def("getData", [](Base& self, const std::string& s) -> py::object
    {
        BaseData* d = self.findData(s);
        if(d!=nullptr)
        {
            return py::cast(d);
        }
        return py::none();
    });

    p.def("getFromData", [](Base* self, const std::string& s) -> py::object
    {
        return BindingBase::GetAttr(self, s, false);
    });

    p.def("setToData", [](py::object self, const std::string& s, py::object value)
    {
        BindingBase::SetAttr(self, s, value,false);
    });

    p.def("getFromDict", [](py::object self, const std::string& s) -> py::object
    {
        return self.attr("__dict__")[s.c_str()];
    });

    p.def("setToDict", [](py::object self, const std::string& s, py::object value)
    {
        py::dict d = self.attr("__dict__");
        d[s.c_str()] = value;
    });

    p.def("__getattr__", [](py::object self, const std::string& s) -> py::object
    {
        py::object res = BindingBase::GetAttr( py::cast<Base*>(self), s, false );
        if( res.is_none() )
        {
            return self.attr("__dict__")[s.c_str()];
        }

        return res;
    });

    p.def("__setattr__", [](py::object self, const std::string& s, py::object value)
    {
        if(py::isinstance<DataContainer>(value))
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

        BindingBase::SetAttr(self,s,value,true);
    });
}

} /// namespace sofapython3
