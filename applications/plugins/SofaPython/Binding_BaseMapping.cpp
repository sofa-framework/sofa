/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "Binding_BaseMapping.h"
#include "Binding_BaseObject.h"
#include "PythonFactory.h"
#include "PythonToSofa.inl"

using namespace sofa;
using namespace sofa::core;
using namespace sofa::core::objectmodel;

static BaseMapping* get_basemapping(PyObject* self) {
    return sofa::py::unwrap<BaseMapping>(self);
}


static PyObject * BaseMapping_getFrom(PyObject * self, PyObject * /*args*/)
{
    BaseMapping* mapping  = get_basemapping( self );

    helper::vector<BaseState*> from = mapping->getFrom();

    PyObject *list = PyList_New(from.size());

    for (unsigned int i=0; i<from.size(); ++i)
        PyList_SetItem(list,i,sofa::PythonFactory::toPython(from[i]));

    return list;
}

static PyObject * BaseMapping_getTo(PyObject * self, PyObject * /*args*/)
{
    BaseMapping* mapping  = get_basemapping( self );

    helper::vector<BaseState*> to = mapping->getTo();

    PyObject *list = PyList_New(to.size());

    for (unsigned int i=0; i<to.size(); ++i)
        PyList_SetItem(list,i,sofa::PythonFactory::toPython(to[i]));

    return list;
}



static PyObject * BaseMapping_setFrom(PyObject * self, PyObject * args)
{
    BaseMapping* mapping  = get_basemapping( self );

    PyObject* pyFrom;
    if (!PyArg_ParseTuple(args, "O",&pyFrom))
    {
        return NULL;
    }

    BaseState* from = sofa::py::unwrap<BaseState>( pyFrom );
    if (!from)
    {
        PyErr_SetString(PyExc_TypeError, "Invalid argument, a BaseState* object is expected. " ) ;
        return NULL;
    }

    mapping->setFrom( from );

    Py_RETURN_NONE;
}

static PyObject * BaseMapping_setTo(PyObject * self, PyObject * args)
{
    BaseMapping* mapping  = get_basemapping( self );

    PyObject* pyTo;
    if (!PyArg_ParseTuple(args, "O",&pyTo)) {
        return NULL;
    }

    BaseState* to = sofa::py::unwrap<BaseState>( pyTo );
    if (!to)
    {
        PyErr_SetString(PyExc_TypeError, "Invalid argument, a BaseState* object is expected. " ) ;
        return NULL;
    }

    mapping->setTo( to );

    Py_RETURN_NONE;
}

static PyObject * BaseMapping_apply(PyObject * self, PyObject * /*args*/)
{
    BaseMapping* mapping  = get_basemapping( self );

    mapping->apply(MechanicalParams::defaultInstance(),VecCoordId::position(),ConstVecCoordId::position());

    Py_RETURN_NONE;
}

static PyObject * BaseMapping_applyJ(PyObject * self, PyObject * /*args*/)
{
    BaseMapping* mapping  = get_basemapping( self );

    mapping->applyJ(MechanicalParams::defaultInstance(),VecDerivId::velocity(),ConstVecDerivId::velocity());

    Py_RETURN_NONE;
}


static PyObject * BaseMapping_applyJT(PyObject * self, PyObject * /*args*/)
{
    BaseMapping* mapping  = get_basemapping( self );

    mapping->applyJT(MechanicalParams::defaultInstance(),VecDerivId::force(),ConstVecDerivId::force());

    Py_RETURN_NONE;
}


static PyObject * BaseMapping_applyDJT(PyObject * self, PyObject * /*args*/)
{
    BaseMapping* mapping  = get_basemapping( self );

    // note: the position delta must be set in dx beforehand
    mapping->applyJT(MechanicalParams::defaultInstance(),VecDerivId::force(),ConstVecDerivId::force());

    Py_RETURN_NONE;
}

// TODO inefficient
// have a look to how to directly bind Eigen sparse matrices
static PyObject * BaseMapping_getJs(PyObject * self, PyObject * /*args*/)
{
    BaseMapping* mapping  = get_basemapping( self );
    const helper::vector<sofa::defaulttype::BaseMatrix*>* Js = mapping->getJs();

    PyObject* Jspython = PyList_New(Js->size());
    for( size_t i=0 ; i<Js->size() ; ++i )
    {
        sofa::defaulttype::BaseMatrix* J = (*Js)[i];

        PyObject* Jpython = PyList_New(J->rows());
        for( sofa::defaulttype::BaseMatrix::Index row=0 ; row<J->rows() ; ++row )
        {
            PyObject* Jrowpython = PyList_New(J->cols());

            for( sofa::defaulttype::BaseMatrix::Index col=0 ; col<J->cols() ; ++col )
                PyList_SetItem( Jrowpython, col, PyFloat_FromDouble( J->element(row,col) ) );

            PyList_SetItem( Jpython, row, Jrowpython );
        }
        PyList_SetItem( Jspython, i, Jpython );
    }
    return Jspython;
}


SP_CLASS_METHODS_BEGIN(BaseMapping)
SP_CLASS_METHOD(BaseMapping,getFrom)
SP_CLASS_METHOD(BaseMapping,getTo)
SP_CLASS_METHOD(BaseMapping,setFrom)
SP_CLASS_METHOD(BaseMapping,setTo)
SP_CLASS_METHOD(BaseMapping,apply)
SP_CLASS_METHOD(BaseMapping,applyJ)
SP_CLASS_METHOD(BaseMapping,applyJT)
SP_CLASS_METHOD(BaseMapping,applyDJT)
SP_CLASS_METHOD(BaseMapping,getJs)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(BaseMapping,BaseMapping,BaseObject)


