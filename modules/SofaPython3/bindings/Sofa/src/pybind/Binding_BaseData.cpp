#include "Binding_BaseData.h"

#include <sofa/defaulttype/DataTypeInfo.h>
using sofa::defaulttype::AbstractTypeInfo;

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

void moduleAddBaseData(py::module &m)
{
    py::class_<BaseData, raw_ptr<BaseData>> p(m, "BaseData");
    p.def("setName", &BaseData::setName);
    p.def("getName", &BaseData::getName);

    // TODO: Implementation should look like: https://github.com/sofa-framework/sofa/issues/767
    p.def("__getitem__", [](BaseData& self, py::object& s) -> py::object
    {
        std::cout << "(mapping protocol, __getitem__ get to implement)" << std::endl ;
        return py::none();
    });

    // TODO: Implementation should look like: https://github.com/sofa-framework/sofa/issues/767
    p.def("__setitem__", [](BaseData& self, py::object& key, py::object& value)
    {
        std::cout << "mapping protocol, __setitem__ to implement)" << std::endl ;
        return py::none();
    });

    /// TODO implement.
    /// Buffer protocol
    /// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#buffer-protocol
    p.def_buffer([](BaseData &m) -> py::buffer_info {
        const AbstractTypeInfo *typeinfo = m.getValueTypeInfo();
//        if(!typeinfo->Container() || !typeinfo->SimpleLayout())
//        {
//            throw py::exception("Unable to create a buffer for this data type");
//        }

//        Py_ssize_t numEntries = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();
//        Py_ssize_t totalSize =  typeinfo->size(data->getValueVoidPtr()) * typeinfo->byteSize();

//        std::cout << "SIZE: " << typeinfo->size() << std::endl;
//        std::cout << "SIZE(): " << typeinfo->size(data->getValueVoidPtr()) << std::endl;
//        std::cout << "byteSIze: " << typeinfo->byteSize() << std::endl;

//        view->len = totalSize;
//        view->readonly = 0;

//        if(typeinfo->Scalar() && typeinfo->byteSize() == 8)
//        {

//            view->itemsize =  typeinfo->byteSize() ;
//            view->format = "d";  // integer
//        }else{
//            std::cout << "BIIIIIIIIIIG PROBLEM." << std::endl ;
//        }

//        view->ndim = 2;
//        view->shape = new Py_ssize_t[2];
//        view->shape[0] = numEntries;
//        view->shape[1] = typeinfo->size();


//        return py::buffer_info(
//                    (void*)data->getValueVoidPtr(),          /* Pointer to buffer */
//                    sizeof(Scalar),                          /* Size of one scalar */
//                    py::format_descriptor<Scalar>::format(), /* Python struct-style format descriptor */
//                    2,                                       /* Number of dimensions */
//                    { m.rows(), m.cols() },                  /* Buffer dimensions */
//                    { sizeof(Scalar) * (rowMajor ? m.cols() : 1),
//                      sizeof(Scalar) * (rowMajor ? 1 : m.rows()) }
//                    /* Strides (in bytes) for each index */
//                    );
    });

}
