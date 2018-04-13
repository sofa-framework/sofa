#ifndef PYTHONSCRIPTDATAENGINE_H
#define PYTHONSCRIPTDATAENGINE_H

#include "PythonEnvironment.h"
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include "ScriptDataEngine.h"
#include <sofa/defaulttype/Vec3Types.h>
#include "PythonFactory.h"
#include "PythonToSofa.inl"



namespace sofa {
    namespace core{
        namespace objectmodel{ class IdleEvent ;}
    }
    namespace helper{
        namespace system{ class FileEventListener; }
    }
}

using sofa::core::objectmodel::Event;
using sofa::core::objectmodel::BaseObjectDescription ;
using sofa::core::objectmodel::BaseData;
using sofa::core::behavior::MechanicalState ;
using sofa::core::topology::BaseMeshTopology ;
using sofa::core::behavior::MechanicalState ;
using sofa::core::objectmodel::BaseContext ;
using sofa::core::objectmodel::BaseObject ;
using sofa::core::visual::VisualParams ;
using sofa::core::objectmodel::Event ;
using sofa::core::ExecParams ;
using sofa::core::DataEngine ;
using sofa::helper::vector ;
using std::string ;

namespace sofa
{

namespace component
{

namespace controller
{

//template <typename DataTypes>
class SOFA_SOFAPYTHON_API PythonScriptDataEngine: public ScriptDataEngine
{


public:
    typedef BaseMeshTopology::Tetra Tetra;
    typedef BaseMeshTopology::SetIndex SetIndex;

public:

    SOFA_CLASS(PythonScriptDataEngine,ScriptDataEngine);
    PyObject* scriptDataEngineInstance() const {return m_ScriptDataEngineInstance;}
    void setInstance(PyObject* instance);
    void refreshBinding();
    void doLoadScript();
    virtual void handleEvent(Event *event) override;
    virtual void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override ;

protected:
    PythonScriptDataEngine();
    virtual ~PythonScriptDataEngine();
    //sofa::helper::system::FileEventListener* m_filelistener {nullptr} ;

    sofa::helper::system::FileEventListener* m_filelistener {nullptr} ;

    PyObject *m_ScriptDataEngineClass      {nullptr} ;   // class implemented in the script to use
                                                         // to instanciate the python DataEngine
    PyObject *m_ScriptDataEngineInstance   {nullptr} ;   // instance of m_ScriptDataEngineClass
    PyObject *m_Func_update                {nullptr} ;
    PyObject *m_Func_init                  {nullptr} ;
    PyObject *m_Func_parse                {nullptr} ;

    virtual void script_update() override;
    virtual void script_init() override;
    virtual void script_parse() override;
    virtual void loadScript() override;    
    void init() override;


public:
    sofa::core::objectmodel::DataFileName       m_filename;
    sofa::core::objectmodel::Data<std::string>  m_classname;
    sofa::core::objectmodel::Data< helper::vector< std::string > >  m_variables; /// array of string variables (equivalent to a c-like argv), while waiting to have a better way to share variables
    sofa::core::objectmodel::Data<bool>         m_timingEnabled;
    sofa::core::objectmodel::Data<bool>         m_doAutoReload;
    sofa::core::objectmodel::Data<bool>         d_doUpdate;    
    sofa::core::objectmodel::Data<float> d_Test;
};

}
}
}

#endif // PYTHONSCRIPTDATAENGINE_H
