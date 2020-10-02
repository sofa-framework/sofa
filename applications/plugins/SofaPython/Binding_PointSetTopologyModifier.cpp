#include "Binding_PointSetTopologyModifier.h"
#include "Binding_BaseObject.h"
#include "PythonToSofa.inl"

using namespace sofa::component::topology;
using namespace sofa::core::topology;

/// getting a PointSetTopologyModifier* from a PyObject*
static inline PointSetTopologyModifier* get_PointSetTopologyModifier(PyObject* obj) {
    return sofa::py::unwrap<PointSetTopologyModifier>(obj);
}

static PyObject * PointSetTopologyModifier_addPoints(PyObject *self, PyObject * args)
{
    PointSetTopologyModifier* obj = get_PointSetTopologyModifier( self );

    PyObject* ancestorElemsArg = nullptr ;

    if (PyArg_UnpackTuple(args, "addPoints", 1, 1, &ancestorElemsArg) )
    {
        bool isList = PyList_Check(ancestorElemsArg);
        if(!isList)
        {
            PyErr_SetString(PyExc_TypeError, "This function is expecting a List") ;
            return nullptr ;
        }

        std::size_t nbAncestorElems  = PyList_Size(ancestorElemsArg);
        sofa::helper::vector< PointAncestorElem > ancestorElems;

        for(std::size_t i=0;i<nbAncestorElems;++i)
        {
            PyObject * pyPointAncestor = PyList_GetItem(ancestorElemsArg,i);
            PointAncestorElem* pointAncestor = sofa::py::unwrap<PointAncestorElem>( pyPointAncestor );
            ancestorElems.push_back( *pointAncestor );
        }

        obj->addPoints(ancestorElems.size(),ancestorElems);
    }

    Py_RETURN_NONE;
}



SP_CLASS_METHODS_BEGIN(PointSetTopologyModifier)
SP_CLASS_METHOD(PointSetTopologyModifier,addPoints)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(PointSetTopologyModifier,PointSetTopologyModifier,BaseObject)
