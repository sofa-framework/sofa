#include "Binding_PythonScriptDataEngine.h"
#include "Binding_BaseObject.h"

using namespace sofa::component::controller;

#include <sofa/simulation/Node.h>
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;

#include "PythonToSofa.inl"

#include <sofa/helper/logging/Messaging.h>

// These functions are empty ones: they are meant to be overriden by real python
// controller scripts


// #define LOG_UNIMPLEMENTED_METHODS // prints a message each time a
// non-implemented (in the script) method is called

/// This function converts an PyObject into a sofa string.
/// string that can be safely parsed in helper::vector<int> or helper::vector<double>
//static std::ostream& pythonToSofaDataString_(PyObject* value, std::ostream& out)
//{
//    /// String are just returned as string.
//    if (PyString_Check(value))
//    {
//        return out << PyString_AsString(value) ;
//    }

//    /// Unicode are converted to string.
//    if(PyUnicode_Check(value))
//    {
//        PyObject* tmpstr = PyUnicode_AsUTF8String(value);
//        out << PyString_AsString(tmpstr) ;
//        Py_DECREF(tmpstr);

//        return out;
//    }

//    if( PySequence_Check(value) )
//    {
//        if(!PyList_Check(value))
//        {
//            msg_warning("SofaPython") << "A sequence which is not a list will be convert to a sofa string.";
//        }
//        /// It is a sequence...so we can iterate over it.
//        PyObject *iterator = PyObject_GetIter(value);
//        if(iterator)
//        {
//            bool first = true;
//            while(PyObject* next = PyIter_Next(iterator))
//            {
//                if(first) first = false;
//                else out << ' ';

//                pythonToSofaDataString(next, out);
//                Py_DECREF(next);
//            }
//            Py_DECREF(iterator);

//            if (PyErr_Occurred())
//            {
//                msg_error("SofaPython") << "error while iterating." << msgendl
//                                        << PythonEnvironment::getStackAsString() ;
//            }
//            return out;
//        }
//    }


//    /// Check if the object has an explicit conversion to a Sofa path. If this is the case
//    /// we use it.
//    if( PyObject_HasAttrString(value, "getAsACreateObjectParameter") ){
//        PyObject* retvalue = PyObject_CallMethod(value, (char*)"getAsACreateObjectParameter", nullptr) ;
//        return pythonToSofaDataString(retvalue, out);
//    }

//    /// Default conversion for standard type:
//    if( !(PyInt_Check(value) || PyLong_Check(value) || PyFloat_Check(value) || PyBool_Check(value) ))
//    {
//        msg_warning("SofaPython") << "You are trying to convert a non primitive type to Sofa using the 'str' operator." << msgendl
//                                  << "Automatic conversion is provided for: String, Integer, Long, Float and Bool and Sequences." << msgendl
//                                  << "Other objects should implement the method getAsACreateObjectParameter(). " << msgendl
//                                  << "This function should return a string usable as a parameter in createObject()." << msgendl
//                                  << "So to remove this message you must add a method getAsCreateObjectParameter(self) "
//                                     "to the object you are passing the createObject function." << msgendl
//                                  << PythonEnvironment::getStackAsString() ;
//    }


//    PyObject* tmpstr=PyObject_Repr(value);
//    out << PyString_AsString(tmpstr) ;
//    Py_DECREF(tmpstr) ;
//    return out ;
//}



#ifdef LOG_UNIMPLEMENTED_METHODS
static inline PythonScriptDataEngine* get_dataengine(PyObject* obj) {
    return down_cast<PythonScriptDataEngine>( get_baseobject( obj ) );
}
#endif

static DataEngine* get_dataengine(PyObject* self) {
    return sofa::py::unwrap<DataEngine>(self);
}

static PyObject * PythonScriptDataEngine_update(PyObject * self, PyObject * /*args*/) {
    (void) self;

#ifdef LOG_UNIMPLEMENTED_METHODS
    PythonScriptDataEngine* obj = get_getdataengine(self);
    msg_error("PythonScriptDataEngine") << obj->m_classname.getValueString()
                                        << ".update not implemented in "
                                        << obj->name.getValueString() << std::endl;
#endif

    Py_RETURN_NONE;
}

static PyObject * PythonScriptDataEngine_init(PyObject * self, PyObject * /*args*/) {
    (void) self;
    msg_error("PythonScriptDataEngine")<< "awa, actually called from Py";

#ifdef LOG_UNIMPLEMENTED_METHODS
    PythonScriptDataEngine* obj = get_getdataengine(self);
    msg_error("PythonScriptDataEngine") << obj->m_classname.getValueString()
                                        << ".init not implemented in "
                                        << obj->name.getValueString() << std::endl;
#endif

    Py_RETURN_NONE;
}

static PyObject * PythonScriptDataEngine_addField(PyObject *self, PyObject* args, PyObject * kw)
{
//    msg_warning("Binding") << "ok, can trigger call from within python";
//    DataEngine* engine = get_dataengine( self );
//    char *FieldType; // Input or Output
//    char *type;

//    if (!PyArg_ParseTuple(args, "s",&FieldType))
//    {
//        return NULL;
//    }

//    BaseObjectDescription desc(FieldType,FieldType);

//    if (kw && PyDict_Size(kw)>0)
//    {
//        PyObject* keys = PyDict_Keys(kw);
//        PyObject* values = PyDict_Values(kw);
//        for (int i=0; i<PyDict_Size(kw); i++)
//        {
//            PyObject *key = PyList_GetItem(keys,i);
//            PyObject *value = PyList_GetItem(values,i);

////            if( !strcmp( PyString_AsString(key), "warning") )
////            {
////                if PyBool_Check(value)
////                        warning = (value==Py_True);
////            }
////            else
////            {
//                std::stringstream s;
//                pythonToSofaDataString_(value, s) ;
//                desc.setAttribute(PyString_AsString(key),s.str().c_str());
////            }
//        }
//        Py_DecRef(keys);
//        Py_DecRef(values);
//    }

//    // TODO: After "parsing" create Input or output here!
//    //BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(context,&desc);

//    Py_RETURN_NONE;
}



struct error { };

template<class T>
static inline T* operator || (T* obj, error e) {
    if(!obj) throw e;
    return obj;
}

static PyObject * PythonScriptDataEngine_new(PyTypeObject * cls, PyObject * args, PyObject* /*kwargs*/) {

    try {
        PyObject* py_node = PyTuple_GetItem(args, 0) || error();

        BaseContext* ctx = sofa::py::unwrap<BaseContext>(py_node) || error();

        using controller_type = PythonScriptDataEngine;
        controller_type::SPtr controller = New<controller_type>();

        // note: original bindings **require** the controller to be wrapped as a
        // Base. virtual inheritance between Base and PythonScriptDataEngine
        // have been cleaned since then, so is should be safe to 1. wrap
        // directly as a PythonScriptDataEngine and 2. static_cast wrapped
        // pointers
        PyObject* instance = BuildPySPtr<Base>(controller.get(), cls);
        controller->setInstance(instance);

        ctx->addObject( controller );

        return instance;

    } catch (error e) {
        PyErr_SetString(PyExc_TypeError,
                        "PythonScriptDataEngine.__new__ needs a Sofa.BaseContext as first argument");
        return NULL;
    };
}


SP_CLASS_METHODS_BEGIN(PythonScriptDataEngine)
SP_CLASS_METHOD(PythonScriptDataEngine,update)
SP_CLASS_METHOD(PythonScriptDataEngine,init)
SP_CLASS_METHOD_KW_DOC(PythonScriptDataEngine,addField,
               "Creates a Sofa object and then adds it to the node. "
               "First argument is the type name, parameters are passed as subsequent keyword arguments.\n"
               "Automatic conversion is performed for Scalar, Integer, String, List & Sequence as well as \n"
               "object with a getAsCreateObjectParameter(self)."
               "example:\n"
               "   object = node.createObject('MechanicalObject',name='mObject', dx=1, dy=2, dz=3)"
               )
SP_CLASS_METHODS_END


namespace {
static struct patch {

    patch() {
        // because i can
        SP_SOFAPYTYPEOBJECT(PythonScriptDataEngine).tp_new = PythonScriptDataEngine_new;
    }

} patcher;
}

SP_CLASS_TYPE_SPTR(PythonScriptDataEngine, PythonScriptDataEngine, DataEngine);

