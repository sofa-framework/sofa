/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef PYTHONMACROS_H
#define PYTHONMACROS_H

#include "PythonCommon.h"
#include <boost/intrusive_ptr.hpp>

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseContext.h>


#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/cast.h>

#include <SofaPython/config.h>




// =============================================================================
// Python structures names in sofa...
// naming convention:
// classe C++:          X
// objet python:        X_PyObject
// object type python:  X_PyTypeObject
// table des méthodes:  X_PyMethods
// =============================================================================
#define SP_SOFAPYOBJECT(X) X##_PyObject
#define SP_SOFAPYTYPEOBJECT(X) X##_PyTypeObject
#define SP_SOFAPYMETHODS(X) X##_PyMethods
#define SP_SOFAPYATTRIBUTES(X) X##_PyAttributes
#define SP_SOFAPYNEW(X) X##_PyNew       // allocator
#define SP_SOFAPYFREE(X) X##_PyFree     // deallocator


// =============================================================================
// Module declarations & methods
// =============================================================================

// PyObject *MyModule = SP_INIT_MODULE(MyModuleName)
#define SP_INIT_MODULE(MODULENAME) Py_InitModule(#MODULENAME,MODULENAME##ModuleMethods);

#define SP_MODULE_METHODS_BEGIN(MODULENAME) PyMethodDef MODULENAME##ModuleMethods[] = {
#define SP_MODULE_METHODS_END {NULL,NULL,0,NULL} };
#define SP_MODULE_METHOD(MODULENAME,M) {#M, MODULENAME##_##M, METH_VARARGS, ""},
#define SP_MODULE_METHOD_KW(MODULENAME,M) {#M, (PyCFunction)MODULENAME##_##M, METH_KEYWORDS|METH_VARARGS, ""},



// =============================================================================
// SPtr objects passed to python
// object MUST inherit from Base, or the dynamic_cast will fail later
// =============================================================================
template <class T>
struct PySPtr
{
    PyObject_HEAD
    boost::intrusive_ptr<T> object;
//    PySPtr()        { object=0; }
//    PySPtr(T *obj)  { object=obj; }

    static void dealloc ( PySPtr* _self ) {
        // The PySPtr struct is allocated and deallocated by the Python C API
        // and so the destructor of the smart ptr is not called when the PyObject
        // is destroyed. To prevent leaking we explicitly remove the reference here.
        _self->object.reset ();
        _self->ob_type->tp_free( (PyObject*)_self );
    }
};

template <class T>
PyObject* BuildPySPtr(T* obj,PyTypeObject *pto)
{
    PySPtr<T> * pyObj = (PySPtr<T> *)PyType_GenericAlloc(pto, 0);
    pyObj->object = obj;
    return (PyObject*)pyObj;
}

//#define SP_BUILD_PYSPTR(PyType,OBJ) BuildPySPtr<Base>(OBJ,&SP_SOFAPYTYPEOBJECT(PyType))
// on n'utilise pas BuildPySPtr<T> car la dynamic_cast se fera sur un T* et pas un T::SPtr
// le type cpp n'est pas nécessaire, tous les SPtr de Sofa héritant de Base

// nouvelle version, retournant automatiquement le type Python de plus haut niveau possible,
// en fonction du type de l'objet Cpp
// afin de permettre l'utilisation de fonctions des sous-classes de Base
SOFA_SOFAPYTHON_API PyObject* SP_BUILD_PYSPTR(sofa::core::objectmodel::Base* obj);



// =============================================================================
// Ptr objects passed to python
// deletion can be made by Python IF the "deletable" flag is true,
// and if a FreeFunc is provided in the class definition (see SP_CLASS_TYPE_DEF)
// (I miss smart pointers...)
// =============================================================================
template <class T>
struct PyPtr
{
    PyObject_HEAD
    T* object;
    bool deletable;
};

template <class T>
PyObject* BuildPyPtr(T* obj,PyTypeObject *pto,bool del)
{
    PyPtr<T> * pyObj = (PyPtr<T> *)PyType_GenericAlloc(pto, 0);
    pyObj->object = obj;
    pyObj->deletable = del;
    return (PyObject*)pyObj;
}
#define SP_BUILD_PYPTR(PyType,CppType,OBJ,DEL) BuildPyPtr<CppType>(OBJ,&SP_SOFAPYTYPEOBJECT(PyType),DEL)





// =============================================================================
// SOFA PYTHON CLASSES DECLARATION&DEFINITION
// =============================================================================

/*
static PyMethodDef DummyClass_PyMethods[] =
{
    {"setValue", DummyClass_setValue, METH_VARARGS, "doc string"},
    {"getValue", DummyClass_getValue, METH_VARARGS, "doc string"},
    {NULL}
};

becomes...

SP_CLASS_METHODS_BEGIN(DummyClass)
SP_CLASS_METHOD(DummyClass,setValue)
SP_CLASS_METHOD(DummyClass,getValue)
SP_CLASS_METHODS_END
*/

#define SP_CLASS_METHODS_BEGIN(C) static PyMethodDef SP_SOFAPYMETHODS(C)[] = {
#define SP_CLASS_METHODS_END {0,0,0,0} };
#define SP_CLASS_METHOD(C,M) {#M, C##_##M, METH_VARARGS, ""},
#define SP_CLASS_METHOD_KW(C,M) {#M, (PyCFunction)C##_##M, METH_KEYWORDS|METH_VARARGS, ""},

/*
static PyGetSetDef DummyClass_PyAttributes[] =
{
    {"name", DummyClass_getAttr_name, DummyClass_setAttr_name, "", 0},
    {NULL}
}

becomes...

SP_CLASS_ATTRS_BEGIN(DummyClass)
SP_CLASS_ATTR(DummyClass,name)
SP_CLASS_ATTRS_END

*/
#define SP_CLASS_ATTRS_BEGIN(C) static PyGetSetDef SP_SOFAPYATTRIBUTES(C)[] = {
#define SP_CLASS_ATTR(C,A) {(char*)#A, C##_getAttr_##A, C##_setAttr_##A, NULL, 0},
#define SP_CLASS_ATTRS_END {NULL,NULL,NULL,NULL,NULL} };

/*
extern "C" PyObject * Data_getAttr_name(PyObject *self, void*)

becomes...

SP_CLASS_ATTR_GET(Datamname)(PyObject *self, void*)
 */
#define SP_CLASS_ATTR_GET(C,A) extern "C" PyObject * C##_getAttr_##A
#define SP_CLASS_ATTR_SET(C,A) extern "C" int C##_setAttr_##A



/*
    if (PyType_Ready(&DummyClass_PyTypeObject) < 0) return 0;
    Py_INCREF(&DummyClass_PyTypeObject);
    PyModule_AddObject(module, "DummyClass", (PyObject *)&DummyClass_PyTypeObject);

becomes...

    SP_ADD_CLASS(module,DummyClass)
*/

#define SP_ADD_CLASS(M,C)   PyType_Ready(&SP_SOFAPYTYPEOBJECT(C));   \
                            Py_INCREF(&SP_SOFAPYTYPEOBJECT(C));      \
                            PyModule_AddObject(M, #C, (PyObject *)&SP_SOFAPYTYPEOBJECT(C));





/*
static PyTypeObject DummyChild_PyTypeObject = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "python_sandbox.DummyChild",        //tp_name
    sizeof(VoidClass_PyObject),        //tp_basicsize
    0,                                  //tp_itemsize
    // methods
    0,                                  //tp_dealloc
    0,                                  //tp_print
    0,                                  //tp_getattr
    0,                                  //tp_setattr
    0,                                  //tp_compare
    0,                                  //tp_repr
    0,                                  //tp_as_number
    0,                                  //tp_as_sequence
    0,                                  //tp_as_mapping
    0,                                  //tp_hash
    0,                                  //tp_call
    0,                                  //tp_str
    0,                                  //tp_getattro
    0,                                  //tp_setattro
    0,                                  //tp_as_buffer
    Py_TPFLAGS_DEFAULT,// | Py_TPFLAGS_BASETYPE,           //tp_flags

    0,                   //tp_doc
    0,                                  //tp_traverse
    0,                                  //tp_clear
    0,                                  //tp_richcompare
    0,                                  //tp_weaklistoffset
    0,                                  //tp_iter
    0,                                  //tp_iternext
    DummyChild_PyMethods,               //tp_methods
    0,                                  //tp_members
    0,                                  //tp_getset
    &DummyClass_PyTypeObject,           //tp_base
    0,                                  //tp_dict
    0,                                  //tp_descr_get
    0,                                  //tp_descr_set
    0,                                  //tp_dictoffset
    0,                                  //tp_init
    0,                                  //tp_alloc
    DummyChild_new,                     //tp_new
    VoidClass_free,                     //tp_free
};
*/

// déclaration, .h
#define SP_DECLARE_CLASS_TYPE(Type) SOFA_SOFAPYTHON_API extern PyTypeObject SP_SOFAPYTYPEOBJECT(Type);

// définition générique (macro intermédiaire)
#define SP_CLASS_TYPE_DEF(Type,ObjSize,AttrTable,ParentTypeObjet,NewFunc,FreeFunc,GetAttrFunc,SetAttrFunc, DeallocFunc)   \
                                                                    PyTypeObject SP_SOFAPYTYPEOBJECT(Type) = { \
                                                                    PyVarObject_HEAD_INIT(NULL, 0) \
                                                                    "Sofa."#Type, \
                                                                    ObjSize, \
                                                                    0, \
                                                                    (destructor)DeallocFunc, 0, \
                                                                    0,0,0,0,0,0,0,0,0,0, \
                                                                    GetAttrFunc, \
                                                                    SetAttrFunc, \
                                                                    0, \
                                                                    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, \
                                                                    0,0,0,0,0,0,0, \
                                                                    SP_SOFAPYMETHODS(Type), \
                                                                    0, \
                                                                    AttrTable, \
                                                                    ParentTypeObjet, \
                                                                    0,0,0,0,0,0, \
                                                                    NewFunc, FreeFunc, \
                                                                    0,0,0,0,0,0,0,0 \
                                                                };


// définition type de base(=sans parent) sans attributs
#define SP_CLASS_TYPE_BASE_SPTR(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),0,&PyBaseObject_Type,0,0,0,0,PySPtr<sofa::core::objectmodel::Base>::dealloc)
#define SP_CLASS_TYPE_BASE_PTR(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),0,&PyBaseObject_Type ,0,0,0,0,0)
#define SP_CLASS_TYPE_BASE_PTR_NEW_FREE(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),0,&PyBaseObject_Type ,SP_SOFAPYNEW(PyType),SP_SOFAPYFREE(PyType),0,0,0)


// définition type de base(=sans parent) avec attributs (voir SP_CLASS_ATTRS_BEGIN, SP_CLASS_ATTRS_END & SP_CLASS_ATTR)
#define SP_CLASS_TYPE_BASE_SPTR_ATTR(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&PyBaseObject_Type,0,0,0,0,PySPtr<sofa::core::objectmodel::Base>::dealloc)
#define SP_CLASS_TYPE_BASE_SPTR_ATTR_GETATTR(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&PyBaseObject_Type,0,0,PyType##_GetAttr,PyType##_SetAttr,PySPtr<sofa::core::objectmodel::Base>::dealloc)
#define SP_CLASS_TYPE_BASE_PTR_ATTR(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&PyBaseObject_Type,0,0,0,0,0)
#define SP_CLASS_TYPE_BASE_PTR_ATTR_NEW_FREE(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&PyBaseObject_Type,SP_SOFAPYNEW(PyType),SP_SOFAPYFREE(PyType),0,0,0)

// définition type hérité de "Parent" sans attributs
#define SP_CLASS_TYPE_SPTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),0,&SP_SOFAPYTYPEOBJECT(Parent),0,0,0,0,PySPtr<sofa::core::objectmodel::Base>::dealloc)
#define SP_CLASS_TYPE_SPTR_GETATTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),0,&SP_SOFAPYTYPEOBJECT(Parent),0,0,PyType##_GetAttr,PyType##_SetAttr,PySPtr<sofa::core::objectmodel::Base>::dealloc)
#define SP_CLASS_TYPE_PTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),0,&SP_SOFAPYTYPEOBJECT(Parent),0,0,0,0,0)

// définition type hérité de "Parent" avec attributs (voir SP_CLASS_ATTRS_BEGIN, SP_CLASS_ATTRS_END & SP_CLASS_ATTR)
#define SP_CLASS_TYPE_SPTR_ATTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&SP_SOFAPYTYPEOBJECT(Parent),0,0,0,0,PySPtr<sofa::core::objectmodel::Base>::dealloc)
#define SP_CLASS_TYPE_SPTR_ATTR_GETATTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&SP_SOFAPYTYPEOBJECT(Parent),0,0,PyType##_GetAttr,PyType##_SetAttr,PySPtr<sofa::core::objectmodel::Base>::dealloc)
#define SP_CLASS_TYPE_PTR_ATTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&SP_SOFAPYTYPEOBJECT(Parent),0,0,0,0,0)

// =============================================================================
// SOFA DATA MEMBERS ACCESS AS ATTRIBUTES
// replace the two definitions of "Base_getAttr_name" and "Base_setAttr_name"
// by a single line:
// SP_CLASS_DATA_ATTRIBUTE(Base,name)
// (+ the entry in the SP_CLASS_ATTR array)
// =============================================================================

#define SP_CLASS_DATA_ATTRIBUTE(C,D) \
    extern "C" PyObject * C##_getAttr_##D(PyObject *self, void*) \
    { \
        C::SPtr obj=((PySPtr<C>*)self)->object;  \
        return PyString_FromString(obj->findData(#D)->getValueString().c_str()); \
    } \
    extern "C" int C##_setAttr_##D(PyObject *self, PyObject * args, void*) \
    { \
        C::SPtr obj=((PySPtr<C>*)self)->object; \
        char *str = PyString_AsString(args); \
        obj->findData(#D)->read(str); \
        return 0; \
    }


// =============================================================================
// ERROR / WARNING MESSAGES
// =============================================================================

//#define SP_MESSAGE_BASE( level, msg ) { MAINLOGGER( level, msg, "SofaPython" ) }
#define SP_MESSAGE_INFO( msg ) msg_info("SofaPython") << msg;
#define SP_MESSAGE_DEPRECATED( msg ) msg_deprecated("SofaPython") << msg;
#define SP_MESSAGE_WARNING( msg ) msg_warning("SofaPython") << msg;
#define SP_MESSAGE_ERROR( msg ) msg_error("SofaPython") << msg;
#define SP_MESSAGE_EXCEPTION( msg ) msg_fatal("SofaPython") << msg;


// get python exceptions and print their error message
void printPythonExceptions();


// =============================================================================
// PYTHON SCRIPT METHOD CALL
// =============================================================================
#define SP_CALL_MODULEFUNC(func, ...) \
{ \
    if (func) { \
        PyObject *res = PyObject_CallObject(func,Py_BuildValue(__VA_ARGS__)); \
        if (!res) { \
            SP_MESSAGE_EXCEPTION("SP_CALL_MODULEFUNC") PyErr_Print(); \
        } \
        else Py_DECREF(res); \
    } \
}


#define SP_CALL_MODULEFUNC_NOPARAM(func) \
{ \
    if (func) { \
        PyObject *res = PyObject_CallObject(func,0); \
        if (!res) { \
            SP_MESSAGE_EXCEPTION("SP_CALL_MODULEFUNC_NOPARAM") PyErr_Print(); \
         } \
        else Py_DECREF(res); \
    } \
}



#endif // PYTHONMACROS_H
