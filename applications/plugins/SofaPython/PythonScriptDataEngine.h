#ifndef PYTHONSCRIPTDATAENGINE_H
#define PYTHONSCRIPTDATAENGINE_H

#include "PythonEnvironment.h"
#include <sofa/core/objectmodel/DataFileName.h>
#include "ScriptDataEngine.h"


namespace sofa {
    namespace core{
        namespace objectmodel{ class IdleEvent ;}
    }
    namespace helper{
        namespace system{ class FileEventListener; }
    }
}


namespace sofa
{

namespace component
{

namespace controller
{

class SOFA_SOFAPYTHON_API PythonScriptDataEngine: public ScriptDataEngine
{
public:
    SOFA_CLASS(PythonScriptDataEngine,ScriptDataEngine);
    PyObject* scriptDataEngineInstance() const {return m_ScriptDataEngineInstance;}
    void setInstance(PyObject* instance);
    void refreshBinding();
    void doLoadScript();

protected:
    PythonScriptDataEngine();
    virtual ~PythonScriptDataEngine();
    //sofa::helper::system::FileEventListener* m_filelistener {nullptr} ;

    sofa::helper::system::FileEventListener* m_filelistener {nullptr} ;

    PyObject *m_ScriptDataEngineClass      {nullptr} ;   // class implemented in the script to use
                                                         // to instanciate the python controller
    PyObject *m_ScriptDataEngineInstance   {nullptr} ;   // instance of m_ScriptControllerClass
    PyObject *m_Func_update                {nullptr} ;

    virtual void script_update() override;
    virtual void loadScript() override;


public:
    sofa::core::objectmodel::DataFileName       m_filename;
    sofa::core::objectmodel::Data<std::string>  m_classname;
    sofa::core::objectmodel::Data< helper::vector< std::string > >  m_variables; /// array of string variables (equivalent to a c-like argv), while waiting to have a better way to share variables
    sofa::core::objectmodel::Data<bool>         m_timingEnabled;
    sofa::core::objectmodel::Data<bool>         m_doAutoReload;
};

}
}
}

#endif // PYTHONSCRIPTDATAENGINE_H
