#include "PythonScriptDataEngine.h"
#include "PythonMacros.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>
using sofa::helper::AdvancedTimer;


using sofa::core::objectmodel::Base;
using sofa::simulation::Node;

#include "Binding_PythonScriptDataEngine.h"
using sofa::simulation::PythonEnvironment;

#include "PythonScriptEvent.h"
using sofa::core::objectmodel::PythonScriptEvent;

#include <sofa/helper/system/FileMonitor.h>
using sofa::helper::system::FileMonitor ;
using sofa::helper::system::FileEventListener ;

#include <sofa/core/objectmodel/IdleEvent.h>
using sofa::core::objectmodel::IdleEvent ;

#include "PythonFactory.h"

namespace sofa
{

namespace component
{

namespace controller
{

class MyyFileEventListener : public FileEventListener
{
    PythonScriptDataEngine* m_dataengine ;
public:
    MyyFileEventListener(PythonScriptDataEngine* psc){
        m_dataengine = psc ;
    }

    virtual ~MyyFileEventListener(){}

    virtual void fileHasChanged(const std::string& filepath){
        PythonEnvironment::gil lock {__func__} ;

        /// This function is called when the file has changed. Two cases have
        /// to be considered if the script was already loaded once or not.
        if(!m_dataengine->scriptDataEngineInstance()){
            m_dataengine->doLoadScript();
        }else{
            std::string file=filepath;
            SP_CALL_FILEFUNC(const_cast<char*>("onReimpAFile"),
                             const_cast<char*>("s"),
                             const_cast<char*>(file.data()));

            m_dataengine->refreshBinding();
        }
    }
};

int PythonScriptDataEngineClass = core::RegisterObject("A Sofa DataEngine scripted in python")
        .add< PythonScriptDataEngine>()
        ;

SOFA_DECL_CLASS(PythonScriptController)


PythonScriptDataEngine::PythonScriptDataEngine()
    :ScriptDataEngine()
    , m_filename(initData(&m_filename, "filename",
                          "Python script filename"))
    , m_classname(initData(&m_classname, "classname",
                           "Python class implemented in the script to instanciate for the controller"))
    , m_variables(initData(&m_variables, "variables",
                           "Array of string variables (equivalent to a c-like argv)" ) )
    , m_timingEnabled(initData(&m_timingEnabled, true, "timingEnabled",
                               "Set this attribute to true or false to activate/deactivate the gathering"
                               " of timing statistics on the python execution time. Default value is set"
                               "to true." ))
    , m_doAutoReload( initData( &m_doAutoReload, false, "autoreload",
                                "Automatically reload the file when the source code is changed. "
                                "Default value is set to false" ) )
    , m_ScriptDataEngineClass(0)
    , m_ScriptDataEngineInstance(0)
{
    m_filelistener = new MyyFileEventListener(this) ;
    msg_warning() << "in constructor ";
}
PythonScriptDataEngine::~PythonScriptDataEngine()
{

}

void PythonScriptDataEngine::script_update()
{
    msg_warning() << "wee, passing in script_update()";
}


void PythonScriptDataEngine::refreshBinding()
{
    //BIND_OBJECT_METHOD_DATA_ENGINE(update)
            //BIND_OBJECT_METHOD(update)
}

void PythonScriptDataEngine::doLoadScript()
{
    loadScript() ;
    msg_warning() << "wee, loading script in DataEngine";
}

void PythonScriptDataEngine::loadScript()
{
    PythonEnvironment::gil lock(__func__);
    if(m_doAutoReload.getValue())
    {
        FileMonitor::addFile(m_filename.getFullPath(), m_filelistener) ;
    }

    // if the filename is empty, the DataEngine is supposed to be in an already loaded file
    // otherwise load the DataEngine's file
    if( m_filename.isSet() && !m_filename.getRelativePath().empty() && !PythonEnvironment::runFile(m_filename.getFullPath().c_str()) )
    {
        msg_error() << " load error (file '"<<m_filename.getFullPath().c_str()<<"' not parsable)" ;
        return;
    }

    // classe
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));
    m_ScriptDataEngineClass = PyDict_GetItemString(pDict,m_classname.getValueString().c_str());
    if (!m_ScriptDataEngineClass)
    {
        msg_error() << " load error (class '"<<m_classname.getValueString()<<"' not found)." ;
        return;
    }

    // verify that the class is a subclass of PythonScriptDataEngine
    if (1!=PyObject_IsSubclass(m_ScriptDataEngineClass,(PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptDataEngine)))
    {
        msg_error() << " load error (class '"<<m_classname.getValueString()<<"' does not inherit from 'Sofa.PythonScriptDataEngine')." ;
        return;
    }

    // créer l'instance de la classe
    m_ScriptDataEngineInstance = BuildPySPtr<Base>(this,(PyTypeObject*)m_ScriptDataEngineClass);

    if (!m_ScriptDataEngineInstance)
    {
        msg_error() << " load error (class '" <<m_classname.getValueString()<<"' instanciation error)." ;
        return;
    }

    msg_warning() << " DataEngine Script loaded successfully.";

    refreshBinding();
}


void PythonScriptDataEngine::setInstance(PyObject* instance) {
    PythonEnvironment::gil lock(__func__);

    // "trust me i'm an engineer"
    if( m_ScriptDataEngineInstance ) {
        Py_DECREF( m_ScriptDataEngineInstance );
    }

    m_ScriptDataEngineInstance = instance;

    // note: we don't use PyObject_Type as it returns a new reference which is
    // not handled correctly in loadScript
    m_ScriptDataEngineClass = (PyObject*)instance->ob_type;

    Py_INCREF( instance );

    refreshBinding();
}



}
}
}
