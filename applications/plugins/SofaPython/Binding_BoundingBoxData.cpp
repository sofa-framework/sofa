#include "Binding_BoundingBoxData.h"
#include "Binding_Data.h"
#include "PythonToSofa.inl"

using sofa::defaulttype::BoundingBox;
using namespace sofa::core::objectmodel;

static inline Data<BoundingBox>* get_boundingbox(PyObject* obj) {
    return sofa::py::unwrap<Data<sofa::defaulttype::BoundingBox> >(obj);
}


SP_CLASS_ATTR_GET(BoundingBox, minBBox)(PyObject *self, void*)
{
    Data<BoundingBox>* obj = get_boundingbox( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return NULL;
    }

    PyObject* shape = PyList_New(3);
    PyList_SetItem(shape, 0, PyFloat_FromDouble(obj->getValue().minBBox().x()));
    PyList_SetItem(shape, 1, PyFloat_FromDouble(obj->getValue().minBBox().y()));
    PyList_SetItem(shape, 2, PyFloat_FromDouble(obj->getValue().minBBox().z()));

    return shape;
}

SP_CLASS_ATTR_GET(BoundingBox, maxBBox)(PyObject *self, void*)
{
    Data<BoundingBox>* obj = get_boundingbox( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return NULL;
    }

    PyObject* shape = PyList_New(3);
    PyList_SetItem(shape, 0, PyFloat_FromDouble(obj->getValue().maxBBox().x()));
    PyList_SetItem(shape, 1, PyFloat_FromDouble(obj->getValue().maxBBox().y()));
    PyList_SetItem(shape, 2, PyFloat_FromDouble(obj->getValue().maxBBox().z()));

    return shape;
}

SP_CLASS_ATTR_SET(BoundingBox, minBBox)(PyObject */*self*/, PyObject * /*args*/, void*)
{
    PyErr_SetString(PyExc_RuntimeError, "BoundingBox attributes are read-only");
    return -1;
}

SP_CLASS_ATTR_SET(BoundingBox, maxBBox)(PyObject */*self*/, PyObject * /*args*/, void*)
{
    PyErr_SetString(PyExc_RuntimeError, "BoundingBox attributes are read-only");
    return -1;
}


SP_CLASS_METHODS_BEGIN(BoundingBox)
SP_CLASS_METHODS_END

SP_CLASS_ATTRS_BEGIN(BoundingBox)
SP_CLASS_ATTR(BoundingBox, minBBox)
SP_CLASS_ATTR(BoundingBox, maxBBox)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_PTR_ATTR(BoundingBox, BaseData, Data)
