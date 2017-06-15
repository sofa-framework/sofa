#include "Binding_TopologyChange.h"
#include <sofa/core/topology/TopologyChange.h>

using namespace sofa::core::topology;


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
    PointAncestorElem* obj=dynamic_cast<PointAncestorElem*>(((PyPtr<PointAncestorElem>*)self)->object);
    delete obj; // done!
}


SP_CLASS_ATTR_GET(PointAncestorElem,type)(PyObject *self, void*)
{
    PointAncestorElem* obj=dynamic_cast<PointAncestorElem*>(((PyPtr<PointAncestorElem>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }

    return PyLong_FromLong( obj->index );
}

SP_CLASS_ATTR_SET(PointAncestorElem, type)(PyObject *self, PyObject * args, void*)
{
    PointAncestorElem* obj=dynamic_cast<PointAncestorElem*>(((PyPtr<PointAncestorElem>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->type = static_cast<sofa::core::topology::TopologyObjectType>( PyLong_AsLong( args ) );

    return 0;
}

SP_CLASS_ATTR_GET(PointAncestorElem,index)(PyObject *self, void*)
{
    PointAncestorElem* obj=dynamic_cast<PointAncestorElem*>(((PyPtr<PointAncestorElem>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }

    return PyLong_FromLong( obj->index );
}

SP_CLASS_ATTR_SET(PointAncestorElem, index)(PyObject *self, PyObject * args, void*)
{
    PointAncestorElem* obj=dynamic_cast<PointAncestorElem*>(((PyPtr<PointAncestorElem>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->index = static_cast<unsigned int>( PyLong_AsLong( args ) );
    return 0;
}

SP_CLASS_ATTR_GET(PointAncestorElem, localCoords)( PyObject *self, void* )
{
    PointAncestorElem* obj=dynamic_cast<PointAncestorElem*>(((PyPtr<PointAncestorElem>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
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
    PointAncestorElem* obj=dynamic_cast<PointAncestorElem*>(((PyPtr<PointAncestorElem>*)self)->object);
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
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
