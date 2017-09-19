#include "Binding_TopologyChange.h"
#include <sofa/core/topology/TopologyChange.h>
#include "PythonToSofa.inl"

using sofa::core::topology::PointAncestorElem ;

/// getting a PointAncestorElem* from a PyObject*
static inline PointAncestorElem* get_PointAncestorElem(PyObject* obj) {
    return sofa::py::unwrap<PointAncestorElem>( obj );
}


// TODO should we set error flag on setattr error ?
// =============================================================================
// (de)allocator
// =============================================================================
PyObject * PointAncestorElem_PyNew(PyTypeObject * /*type*/, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PointAncestorElem *obj = new PointAncestorElem();
    return SP_BUILD_PYPTR(PointAncestorElem,PointAncestorElem,obj,true); // "true", because I manage the deletion myself (below)
}

void PointAncestorElem_PyFree(void * self)
{
    if (!((PyPtr<PointAncestorElem>*)self)->deletable) return;
    PointAncestorElem* obj = get_PointAncestorElem( (PyObject*)self );
    delete obj; // done!
}

// STOP USING MACROS FFS
SP_CLASS_ATTR_GET(PointAncestorElem,type)(PyObject *self, void*)
{
    PointAncestorElem* obj = get_PointAncestorElem( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return NULL;
    }

    return PyLong_FromLong( obj->index );
}

SP_CLASS_ATTR_SET(PointAncestorElem, type)(PyObject *self, PyObject * args, void*)
{
    PointAncestorElem* obj = get_PointAncestorElem( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return -1;
    }
    obj->type = static_cast<sofa::core::topology::TopologyObjectType>( PyLong_AsLong( args ) );

    return 0;
}

SP_CLASS_ATTR_GET(PointAncestorElem,index)(PyObject *self, void*)
{
    PointAncestorElem* obj = get_PointAncestorElem( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return NULL;
    }

    return PyLong_FromLong( obj->index );
}

SP_CLASS_ATTR_SET(PointAncestorElem, index)(PyObject *self, PyObject * args, void*)
{
    PointAncestorElem* obj = get_PointAncestorElem( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return -1;
    }
    obj->index = static_cast<unsigned int>( PyLong_AsLong( args ) );
    return 0;
}

SP_CLASS_ATTR_GET(PointAncestorElem, localCoords)( PyObject *self, void* )
{
    PointAncestorElem* obj = get_PointAncestorElem( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return NULL;
    }

    PyObject* pyLocalCoords = PyTuple_New( PointAncestorElem::LocalCoords::size() );

    for(std::size_t i=0; i<obj->localCoords.size();++i)
    {
        PyTuple_SET_ITEM( pyLocalCoords, i, PyFloat_FromDouble(obj->localCoords[i]) );
    }

    return pyLocalCoords;
}

SP_CLASS_ATTR_SET(PointAncestorElem, localCoords)( PyObject* self,  PyObject * args, void*)
{
    PointAncestorElem* obj = get_PointAncestorElem( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return -1;
    }

    PointAncestorElem::LocalCoords& localCoords = obj->localCoords;
    PyArg_ParseTuple(args, "ddd",&localCoords[0],&localCoords[1],&localCoords[2]);

    return 0;
}

SP_CLASS_METHODS_BEGIN(PointAncestorElem)
SP_CLASS_METHODS_END

SP_CLASS_ATTRS_BEGIN(PointAncestorElem)
SP_CLASS_ATTR(PointAncestorElem,type)
SP_CLASS_ATTR(PointAncestorElem,index)
SP_CLASS_ATTR(PointAncestorElem,localCoords)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_BASE_PTR_ATTR_NEW_FREE(PointAncestorElem,PointAncestorElem)
