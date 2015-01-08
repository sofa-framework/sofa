#include "Binding_PointSetTopologyModifier.h"
#include "Binding_BaseObject.h"
#include <SofaBaseTopology/PointSetTopologyModifier.h>
#include <sofa/core/topology/TopologyChange.h>

using namespace sofa::component::topology;
using namespace sofa::core::topology;

    //void addPoints( const unsigned int nPoints,
    //                const sofa::helper::vector< core::topology::PointAncestorElem >& ancestorElems,
    //                const bool addDOF = true);


extern "C" PyObject * PointSetTopologyModifier_addPoints(PyObject *self, PyObject * args)
{
    PointSetTopologyModifier* obj = dynamic_cast<PointSetTopologyModifier*>(((PySPtr<Base>*)self)->object.get());

    PyObject* ancestorElemsArg = NULL;

    if (PyArg_UnpackTuple(args, "addPoints", 1, 1, &ancestorElemsArg) ) 
    {
        bool isList = PyList_Check(ancestorElemsArg);
        if(!isList)
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        std::size_t nbAncestorElems  = PyList_Size(ancestorElemsArg);
        sofa::helper::vector< PointAncestorElem > ancestorElems;

        for(std::size_t i=0;i<nbAncestorElems;++i)
        {
            PyObject * pyPointAncestor = PyList_GetItem(ancestorElemsArg,i);
            PointAncestorElem* pointAncestor = dynamic_cast<PointAncestorElem*>(((PyPtr<PointAncestorElem>*)pyPointAncestor)->object);
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
