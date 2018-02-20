/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "Binding_VisualModel.h"
#include "Binding_BaseState.h"
#include <fstream>
#include "PythonToSofa.inl"

using sofa::core::visual::VisualModel ;
using sofa::component::visualmodel::VisualModelImpl ;

static inline VisualModel* get_visualmodel(PyObject* obj) {
    return sofa::py::unwrap<VisualModel>(obj);
}


/// getting a VisualModelImpl* from a PyObject*
static inline VisualModelImpl* get_VisualModelImpl(PyObject* obj) {
    return sofa::py::unwrap<VisualModelImpl>(obj);
}

static PyObject * VisualModelImpl_setColor(PyObject *self, PyObject * args)
{
    VisualModelImpl* obj = get_VisualModelImpl( self );
    double r,g,b,a;
    if (!PyArg_ParseTuple(args, "dddd",&r,&g,&b,&a))
    {
        int ir,ig,ib,ia; // helper: you can set integer values
        if (!PyArg_ParseTuple(args, "iiii",&ir,&ig,&ib,&ia))
        {
            return NULL;
        }

        PyErr_Clear();
        r = (double)ir;
        g = (double)ig;
        b = (double)ib;
        a = (double)ia;
    }
    obj->setColor((float)r,(float)g,(float)b,(float)a);
    Py_RETURN_NONE;
}



static PyObject * VisualModel_exportOBJ(PyObject *self, PyObject * args)
{
    VisualModel* obj = get_visualmodel( self );

    char* filename;
    if (!PyArg_ParseTuple(args, "s",&filename))
    {
        return NULL;
    }

    std::ofstream outfile(filename);

    int vindex = 0;
    int nindex = 0;
    int tindex = 0;
    int count = 0;

    obj->exportOBJ(obj->getName(),&outfile,NULL,vindex,nindex,tindex,count);
    outfile.close();

    Py_RETURN_NONE;
}


static PyObject * VisualModel_updateVisual(PyObject *self, PyObject * /*args*/)
{
    VisualModel* obj = get_visualmodel( self );
    obj->updateVisual();
    Py_RETURN_NONE;
}

static PyObject * VisualModel_initVisual(PyObject *self, PyObject * /*args*/)
{
    VisualModel* obj = get_visualmodel( self );
    obj->initVisual();
    Py_RETURN_NONE;
}

SP_CLASS_METHODS_BEGIN(VisualModel)
SP_CLASS_METHOD(VisualModel,exportOBJ)
SP_CLASS_METHOD(VisualModel,updateVisual)
SP_CLASS_METHOD(VisualModel,initVisual)
SP_CLASS_METHODS_END

SP_CLASS_METHODS_BEGIN(VisualModelImpl)
SP_CLASS_METHOD(VisualModelImpl,setColor)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(VisualModel,VisualModel,BaseState)
SP_CLASS_TYPE_SPTR(VisualModelImpl,VisualModelImpl,VisualModel)


