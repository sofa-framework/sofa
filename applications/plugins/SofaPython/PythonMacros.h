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
#ifndef PYTHONMACROS_H
#define PYTHONMACROS_H

// TODO DEPRECATE AND REMOVE THIS MESS

#include <sofa/config.h>

#include "PythonCommon.h"
#include <sofa/core/sptr.h>

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseContext.h>


#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/cast.h>

#include <SofaPython/config.h>
#include <SofaPython/PythonEnvironment.h>

/// This function converts an PyObject into a sofa string.
/// string that can be safely parsed in helper::vector<int> or helper::vector<double>
SOFA_SOFAPYTHON_API std::ostream& pythonToSofaDataString(PyObject* value, std::ostream& out) ;

// =============================================================================
// Python structures names in sofa...
// naming convention:
// classe C++:          X
// objet python:        X_PyObject
// object type python:  X_PyTypeObject
// table des méthodes:  X_PyMethods
// =============================================================================
//#define SP_SOFAPYOBJECT(X) X##_PyObject
#define SP_SOFAPYTYPEOBJECT(X) X##_PyTypeObject
#define SP_SOFAPYMETHODS(X) X##_PyMethods
#define SP_SOFAPYATTRIBUTES(X) X##_PyAttributes
#define SP_SOFAPYMAPPING(X) X##_PyMapping
#define SP_SOFAPYNEW(X) X##_PyNew       // allocator
#define SP_SOFAPYFREE(X) X##_PyFree     // deallocator



// =============================================================================
// Module declarations & methods + docstring creation
// =============================================================================

// PyObject *MyModule = SP_INIT_MODULE(MyModuleName)
#define SP_INIT_MODULE(MODULENAME) Py_InitModule(#MODULENAME,MODULENAME##ModuleMethods); sofa::simulation::PythonEnvironment::excludeModuleFromReload(#MODULENAME);

#define SP_MODULE_METHODS_BEGIN(MODULENAME) PyMethodDef MODULENAME##ModuleMethods[] = {
#define SP_MODULE_METHODS_END {NULL,NULL,0,NULL} };
#define SP_MODULE_METHOD(MODULENAME,M) {#M, MODULENAME##_##M, METH_VARARGS, ""},
#define SP_MODULE_METHOD_DOC(MODULENAME,M, D) {#M, MODULENAME##_##M, METH_VARARGS, D},
#define SP_MODULE_METHOD_KW(MODULENAME,M) {#M, (PyCFunction)MODULENAME##_##M, METH_KEYWORDS|METH_VARARGS, ""},
#define SP_MODULE_METHOD_KW_DOC(MODULENAME,M, D) {#M, (PyCFunction)MODULENAME##_##M, METH_KEYWORDS|METH_VARARGS, D},



// =============================================================================
// SPtr objects passed to python
// object MUST inherit from Base, or the dynamic_cast will fail later
// =============================================================================
template <class T>
struct PySPtr
{
    PyObject_HEAD
    sofa::core::sptr<T> object;
    
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
static inline PyObject* BuildPySPtr(T* obj, PyTypeObject *pto) {
    PySPtr<T> * pyObj = (PySPtr<T> *) PyType_GenericAlloc(pto, 0);
    pyObj->object = obj;
    return (PyObject*)pyObj;
}

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
static inline PyObject* BuildPyPtr(T* obj, PyTypeObject *pto, bool del) {
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
#define SP_CLASS_METHOD_DOC(C,M,D) {#M, C##_##M, METH_VARARGS, D},
#define SP_CLASS_METHOD_KW(C,M) {#M, (PyCFunction)C##_##M, METH_KEYWORDS|METH_VARARGS, ""},
#define SP_CLASS_METHOD_KW_DOC(C,M,D) {#M, (PyCFunction)C##_##M, METH_KEYWORDS|METH_VARARGS, D},

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
#define SP_CLASS_ATTR_GET(C,A) static PyObject * C##_getAttr_##A
#define SP_CLASS_ATTR_SET(C,A) static int C##_setAttr_##A



/*
static PyMappingMethods DummyClass_PyMapping =
    {DummyClass_size, DummyClass_getitem, DummyClass_setitem};

becomes

SP_CLASS_MAPPING(DummyClass)


Note this is how to create a sequence (operators x[] and len(x))

*/
#define SP_CLASS_MAPPING(C) static PyMappingMethods SP_SOFAPYMAPPING(C) = { C##_length, C##_getitem, C##_setitem };



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
#define SP_CLASS_TYPE_DEF(Type,ObjSize,AttrTable,ParentTypeObjet,NewFunc,FreeFunc,GetAttrFunc,SetAttrFunc, DeallocFunc,Mapping)   \
                                                                    PyTypeObject SP_SOFAPYTYPEOBJECT(Type) = { \
                                                                    PyVarObject_HEAD_INIT(NULL, 0) \
                                                                    "Sofa."#Type, \
                                                                    ObjSize, \
                                                                    0, \
                                                                    (destructor)DeallocFunc, 0, \
                                                                    0,0,0,0,0,0, \
                                                                    Mapping, \
                                                                    0,0,0, \
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
#define SP_CLASS_TYPE_BASE_SPTR(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),0,&PyBaseObject_Type,0,0,0,0,PySPtr<sofa::core::objectmodel::Base>::dealloc,0)
#define SP_CLASS_TYPE_BASE_PTR(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),0,&PyBaseObject_Type ,0,0,0,0,0,0)
#define SP_CLASS_TYPE_BASE_PTR_NEW_FREE(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),0,&PyBaseObject_Type ,SP_SOFAPYNEW(PyType),SP_SOFAPYFREE(PyType),0,0,0,0)


// définition type de base(=sans parent) avec attributs (voir SP_CLASS_ATTRS_BEGIN, SP_CLASS_ATTRS_END & SP_CLASS_ATTR)
#define SP_CLASS_TYPE_BASE_SPTR_ATTR(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&PyBaseObject_Type,0,0,0,0,PySPtr<sofa::core::objectmodel::Base>::dealloc,0)
#define SP_CLASS_TYPE_BASE_SPTR_ATTR_GETATTR(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&PyBaseObject_Type,0,0,PyType##_GetAttr,PyType##_SetAttr,PySPtr<sofa::core::objectmodel::Base>::dealloc,0)
#define SP_CLASS_TYPE_BASE_PTR_ATTR(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&PyBaseObject_Type,0,0,0,0,0,0)
#define SP_CLASS_TYPE_BASE_PTR_ATTR_NEW_FREE(PyType,CppType) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&PyBaseObject_Type,SP_SOFAPYNEW(PyType),SP_SOFAPYFREE(PyType),0,0,0,0)

// définition type hérité de "Parent" sans attributs
#define SP_CLASS_TYPE_SPTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),0,&SP_SOFAPYTYPEOBJECT(Parent),0,0,0,0,PySPtr<sofa::core::objectmodel::Base>::dealloc,0)
#define SP_CLASS_TYPE_SPTR_GETATTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),0,&SP_SOFAPYTYPEOBJECT(Parent),0,0,PyType##_GetAttr,PyType##_SetAttr,PySPtr<sofa::core::objectmodel::Base>::dealloc,0)
#define SP_CLASS_TYPE_PTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),0,&SP_SOFAPYTYPEOBJECT(Parent),0,0,0,0,0,0)

// définition type hérité de "Parent" avec attributs (voir SP_CLASS_ATTRS_BEGIN, SP_CLASS_ATTRS_END & SP_CLASS_ATTR)
#define SP_CLASS_TYPE_SPTR_ATTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&SP_SOFAPYTYPEOBJECT(Parent),0,0,0,0,PySPtr<sofa::core::objectmodel::Base>::dealloc,0)
#define SP_CLASS_TYPE_SPTR_ATTR_GETATTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&SP_SOFAPYTYPEOBJECT(Parent),0,0,PyType##_GetAttr,PyType##_SetAttr,PySPtr<sofa::core::objectmodel::Base>::dealloc,0)
#define SP_CLASS_TYPE_PTR_ATTR(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&SP_SOFAPYTYPEOBJECT(Parent),0,0,0,0,0,0)


// définition type hérité de "Parent" avec attributs (voir SP_CLASS_ATTRS_BEGIN, SP_CLASS_ATTRS_END & SP_CLASS_ATTR) et Mapping (voir SP_CLASS_MAPPING)
#define SP_CLASS_TYPE_SPTR_ATTR_MAPPING(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&SP_SOFAPYTYPEOBJECT(Parent),0,0,0,0,PySPtr<sofa::core::objectmodel::Base>::dealloc,&SP_SOFAPYMAPPING(PyType))
#define SP_CLASS_TYPE_SPTR_ATTR_GETATTR_MAPPING(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PySPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&SP_SOFAPYTYPEOBJECT(Parent),0,0,PyType##_GetAttr,PyType##_SetAttr,PySPtr<sofa::core::objectmodel::Base>::dealloc,&SP_SOFAPYMAPPING(PyType))
#define SP_CLASS_TYPE_PTR_ATTR_MAPPING(PyType,CppType,Parent) SP_CLASS_TYPE_DEF(PyType,sizeof(PyPtr<CppType>),SP_SOFAPYATTRIBUTES(PyType),&SP_SOFAPYTYPEOBJECT(Parent),0,0,0,0,0,&SP_SOFAPYMAPPING(PyType))



// =============================================================================
// SOFA DATA MEMBERS ACCESS AS ATTRIBUTES
// replace the two definitions of "Base_getAttr_name" and "Base_setAttr_name"
// by a single line:
// SP_CLASS_DATA_ATTRIBUTE(Base,name)
// (+ the entry in the SP_CLASS_ATTR array)
// =============================================================================

#define SP_CLASS_DATA_ATTRIBUTE(C,D)                                    \
    static PyObject * C##_getAttr_##D(PyObject *self, void*)            \
    {                                                                   \
        C::SPtr obj=((PySPtr<C>*)self)->object;                         \
        return PyString_FromString(obj->findData(#D)->getValueString().c_str()); \
    }                                                                   \
    static int C##_setAttr_##D(PyObject *self, PyObject * args, void*)  \
    {                                                                   \
        C::SPtr obj=((PySPtr<C>*)self)->object;                         \
        char *str = PyString_AsString(args);                            \
        obj->findData(#D)->read(str);                                   \
        return 0;                                                       \
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

#define SP_PYERR_SETSTRING_INVALIDTYPE( o ) PyErr_SetString(PyExc_TypeError, "Invalid argument, a " o " object is expected.");
#define SP_PYERR_SETSTRING_OUTOFBOUND( o ) PyErr_SetString(PyExc_RuntimeError, "Out of bound exception.");


// get python exceptions and print their error message
SOFA_SOFAPYTHON_API void printPythonExceptions();

// deal with SystemExit before PyErr_Print does
SOFA_SOFAPYTHON_API void handle_python_error(const char* message);


// =============================================================================
// PYTHON SEARCH FOR A FUNCTION WITH A GIVEN NAME
// @warning storing the function pointer in a variable called 'm_Func_funcName'
// @warning getting the function pointer from a dictionnary called 'pDict'
// @todo Is it really generic enough to be here?
// =============================================================================
#define BIND_SCRIPT_FUNC(funcName){\
        m_Func_##funcName = PyDict_GetItemString(pDict, #funcName);\
        if (!PyCallable_Check(m_Func_##funcName)) m_Func_##funcName=0; \
    }

// =============================================================================
// PYTHON SEARCH FOR A METHOD WITH A GIVEN NAME
// @warning storing the function pointer in a variable called 'm_Func_funcName'
// @warning getting the function pointer from a PythonScriptController
// @todo Is it really generic enough to be here?
// =============================================================================
#define BIND_OBJECT_METHOD(funcName) \
    { \
    if( PyObject_HasAttrString((PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptController),#funcName ) && \
        PyObject_RichCompareBool( PyObject_GetAttrString(m_ScriptControllerClass, #funcName),\
                                   PyObject_GetAttrString((PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptController), #funcName),Py_NE ) && \
        PyObject_HasAttrString(m_ScriptControllerInstance,#funcName ) ) { \
            m_Func_##funcName = PyObject_GetAttrString(m_ScriptControllerInstance,#funcName); \
            if (!PyCallable_Check(m_Func_##funcName)) \
                {m_Func_##funcName=0; sout<<#funcName<<" not callable"<<sendl;} \
            else \
                {sout<<#funcName<<" found"<<sendl;} \
    }else{ \
        m_Func_##funcName=0; sout<<#funcName<<" not found"<<sendl; } \
    }



// =============================================================================
// Copy of the above with adaption for PythonScriptDataEngine
// =============================================================================

#define BIND_OBJECT_METHOD_DATA_ENGINE(funcName) \
    { \
    if( PyObject_HasAttrString((PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptDataEngine),#funcName ) && \
        PyObject_RichCompareBool( PyObject_GetAttrString(m_ScriptDataEngineClass, #funcName),\
                                   PyObject_GetAttrString((PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptDataEngine), #funcName),Py_NE ) && \
        PyObject_HasAttrString(m_ScriptDataEngineInstance,#funcName ) ) { \
            m_Func_##funcName = PyObject_GetAttrString(m_ScriptDataEngineInstance,#funcName); \
            if (!PyCallable_Check(m_Func_##funcName)) \
                {m_Func_##funcName=0; sout<<#funcName<<" not callable"<<sendl;} \
            else \
                {sout<<#funcName<<" found"<<sendl;} \
    }else{ \
        m_Func_##funcName=0; sout<<#funcName<<" not found"<<sendl; } \
    }

// =============================================================================
// PYTHON SCRIPT METHOD CALL
// =============================================================================
// call a function that returns void
#define SP_CALL_FILEFUNC(func, ...){\
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));\
    PyObject *pFunc = PyDict_GetItemString(pDict, func);\
    if (PyCallable_Check(pFunc))\
{\
    PyObject *res = PyObject_CallFunction(pFunc,__VA_ARGS__); \
    if( res )  Py_DECREF(res); \
}\
}

#define SP_CALL_MODULEFUNC(func, ...)                                   \
    {                                                                   \
     if (func) {                                                        \
         PyObject *res = PyObject_CallObject(func,Py_BuildValue(__VA_ARGS__)); \
         if (!res) {                                                    \
             handle_python_error("SP_CALL_MODULEFUNC");                 \
         }                                                              \
         else Py_DECREF(res);                                           \
     }                                                                  \
    }


#define SP_CALL_MODULEFUNC_NOPARAM(func)                            \
    {                                                               \
        if (func) {                                                 \
            PyObject *res = PyObject_CallObject(func,0);            \
            if (!res) {                                             \
                handle_python_error("SP_CALL_MODULEFUNC_NOPARAM");  \
            }                                                       \
            else Py_DECREF(res);                                    \
        }                                                           \
    }

// call a function that returns a boolean
#define SP_CALL_MODULEBOOLFUNC(func, ...) {                             \
        if (func) {                                                     \
            PyObject *res = PyObject_CallObject(func,Py_BuildValue(__VA_ARGS__)); \
            if (!res)                                                   \
                {                                                       \
                    handle_python_error("SP_CALL_MODULEFUNC_BOOL");     \
                }                                                       \
            else                                                        \
                {                                                       \
                    if PyBool_Check(res) b = ( res == Py_True );        \
                    Py_DECREF(res);                                     \
                }                                                       \
        }                                                               \
}


#endif // PYTHONMACROS_H
