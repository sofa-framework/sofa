#include "Binding_PythonScriptDataEngine.h"
#include "Binding_BaseObject.h"
#include "Binding_Base.h"


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

BaseData * helper_addNewIO(PyObject * self, PyObject * args, PyObject * kw)
{
    DataEngine* engine = get_dataengine( self );
    BaseData* NewData;

    NewData = helper_addNewDataKW(args,kw,engine);

    if(NewData==nullptr)
    {
        msg_error("SofaPython") << "Adding new IO failed!";
        return nullptr;
    }
    NewData->setGroup(""); // Needs to be empty before it can be set to Input or Output ...

    return NewData;

}

static PyObject * PythonScriptDataEngine_addNewInput(PyObject *self, PyObject* args, PyObject * kw)
{
     DataEngine* engine = get_dataengine( self );

     BaseData * NewData = helper_addNewIO(self, args, kw);

     if (NewData == nullptr)
     {
         Py_RETURN_NONE;
     }

     // Check IO stuff
     // TODO (Stefan Escaida 29.01.2018): maybe in the long term enforce that an Input can either be constant or only linked to an Output (for dat that Simulink feelz)
     BaseData* Parent = NewData->getParent();
     char * ParentGroup;
     if (Parent!=nullptr && strcmp(Parent->getGroup(), "Outputs")!=0)
     {
        msg_warning("SofaPython") << "Linking a Data defined as Input to a Data that is not an Output";
     }

     engine->addInput(NewData);
     Py_RETURN_NONE;
}

static PyObject * PythonScriptDataEngine_addNewOutput(PyObject *self, PyObject* args, PyObject * kw)
{
    DataEngine* engine = get_dataengine( self );

    BaseData * NewData = helper_addNewIO(self,args, kw);

    if (NewData == nullptr)
    {
        Py_RETURN_NONE;
    }

    engine->addOutput(NewData);
    Py_RETURN_NONE;
}

//static PyObject * PythonScriptDataEngine_testKwargs(PyObject * self, PyObject* args, PyObject *kw)
//{
//}


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
SP_CLASS_METHOD_KW_DOC(PythonScriptDataEngine,addNewInput,
               "Creates a new sofa Data of the desired type and adds it as input to the PSDE-object. "
               )
SP_CLASS_METHOD_KW_DOC(PythonScriptDataEngine,addNewOutput,
               "Creates a new sofa Data of the desired type and adds it as output to the PSDE-object. "
               )
//SP_CLASS_METHOD_KW_DOC(PythonScriptDataEngine,testKwargs,"help!")
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

