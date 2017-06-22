/// @author Matthieu Nesme
/// @date 2016



#include <SofaPython/PythonMacros.h>
#include "Binding_AssembledSystem.h"
#include <SofaPython/PythonToSofa.inl>


SP_DECLARE_CLASS_TYPE(AssembledSystem)


using namespace sofa::component::linearsolver;


PyObject* getMatrice( const AssembledSystem::rmat& A )
{
    PyObject* pyA = PyList_New(A.rows());
    for( AssembledSystem::rmat::Index row=0 ; row<A.rows() ; ++row )
    {
        PyObject* rowpython = PyList_New(A.cols());

        for( AssembledSystem::rmat::Index col=0 ; col<A.cols() ; ++col )
            PyList_SetItem( rowpython, col, PyFloat_FromDouble( A.coeff(row,col) ) );

        PyList_SetItem( pyA, row, rowpython );
    }

    return pyA;
}

using sofa::py::unwrap;


static PyObject* AssembledSystem_getH(PyObject * self, PyObject * /*args*/)
{
    AssembledSystem* sys = unwrap<AssembledSystem>( self );
    if (!sys)
    {
        PyErr_BadArgument();
        return NULL;
    }

    return getMatrice( sys->H );
}

static PyObject* AssembledSystem_getP(PyObject * self, PyObject * /*args*/)
{
    AssembledSystem* sys = unwrap<AssembledSystem>( self );
    if (!sys)
    {
        PyErr_BadArgument();
        return NULL;
    }

    return getMatrice( sys->P );
}

static PyObject* AssembledSystem_getJ(PyObject * self, PyObject * /*args*/)
{
    AssembledSystem* sys = unwrap<AssembledSystem>( self );
    if (!sys)
    {
        PyErr_BadArgument();
        return NULL;
    }

    return getMatrice( sys->J );
}

static PyObject* AssembledSystem_getC(PyObject * self, PyObject * /*args*/)
{
    AssembledSystem* sys = unwrap<AssembledSystem>( self );
    if (!sys)
    {
        PyErr_BadArgument();
        return NULL;
    }

    return getMatrice( sys->C );
}


SP_CLASS_METHODS_BEGIN(AssembledSystem)
SP_CLASS_METHOD(AssembledSystem,getH)
SP_CLASS_METHOD(AssembledSystem,getP)
SP_CLASS_METHOD(AssembledSystem,getJ)
SP_CLASS_METHOD(AssembledSystem,getC)
SP_CLASS_METHODS_END





SP_CLASS_ATTR_GET(AssembledSystem,m)(PyObject *self, void*)
{
    AssembledSystem* obj = unwrap<AssembledSystem>( self );
    if (!obj)
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    return PyLong_FromLong(obj->m);
}
SP_CLASS_ATTR_SET(AssembledSystem,m)(PyObject *self, PyObject * args, void*)
{
    AssembledSystem* obj = unwrap<AssembledSystem>( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->m=PyLong_AsLong(args);
    return 0;
}

SP_CLASS_ATTR_GET(AssembledSystem,n)(PyObject *self, void*)
{
    AssembledSystem* obj = unwrap<AssembledSystem>( self );
    if (!obj)
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    return PyLong_FromLong(obj->n);
}
SP_CLASS_ATTR_SET(AssembledSystem,n)(PyObject *self, PyObject * args, void*)
{
    AssembledSystem* obj = unwrap<AssembledSystem>( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->n=PyLong_AsLong(args);
    return 0;
}

SP_CLASS_ATTR_GET(AssembledSystem,dt)(PyObject *self, void*)
{
    AssembledSystem* obj = unwrap<AssembledSystem>( self );
    if (!obj)
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }
    return PyFloat_FromDouble(obj->dt);
}
SP_CLASS_ATTR_SET(AssembledSystem,dt)(PyObject *self, PyObject * args, void*)
{
    AssembledSystem* obj = unwrap<AssembledSystem>( self );
    if (!obj)
    {
        PyErr_BadArgument();
        return 0;
    }
    obj->dt=PyFloat_AsDouble(args);
    return 0;
}

// eventual attributes
SP_CLASS_ATTRS_BEGIN(AssembledSystem)
SP_CLASS_ATTR(AssembledSystem,m)
SP_CLASS_ATTR(AssembledSystem,n)
SP_CLASS_ATTR(AssembledSystem,dt)
SP_CLASS_ATTRS_END






// =============================================================================
// (de)allocator
// =============================================================================
PyObject * AssembledSystem_PyNew(PyTypeObject * /*type*/, PyObject *args, PyObject * /*kwds*/)
{
    unsigned m=0,n=0;
    if (!PyArg_ParseTuple(args, "|II",&m,&n))
        Py_RETURN_NONE;
    AssembledSystem *obj = new AssembledSystem(m,n);
    return SP_BUILD_PYPTR(AssembledSystem,AssembledSystem,obj,true); // "true", because I manage the deletion myself (below)
}
void AssembledSystem_PyFree(void * self)
{
    if (!((PyPtr<AssembledSystem>*)self)->deletable) return;
    AssembledSystem* obj = unwrap<AssembledSystem>( (PyObject*)self );
    delete obj; // done!
}



SP_CLASS_TYPE_BASE_PTR_ATTR_NEW_FREE(AssembledSystem,AssembledSystem)
