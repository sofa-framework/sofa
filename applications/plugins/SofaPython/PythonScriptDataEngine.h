#ifndef PYTHONSCRIPTDATAENGINE_H
#define PYTHONSCRIPTDATAENGINE_H

#include "PythonEnvironment.h"
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include "ScriptDataEngine.h"
#include <sofa/defaulttype/Vec3Types.h>



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
   // typedef typename DataTypes::VecCoord VecCoord; TODO: template this class??
public:
//    SOFA_CLASS(SOFA_TEMPLATE(PythonScriptDataEngine,DataTypes),ScriptDataEngine);
      SOFA_CLASS(PythonScriptDataEngine,ScriptDataEngine);
    PyObject* scriptDataEngineInstance() const {return m_ScriptDataEngineInstance;}
    void setInstance(PyObject* instance);
    void refreshBinding();
    void doLoadScript();
    virtual void handleEvent(Event *event) override;

protected:
    PythonScriptDataEngine();
    virtual ~PythonScriptDataEngine();
    //sofa::helper::system::FileEventListener* m_filelistener {nullptr} ;

    sofa::helper::system::FileEventListener* m_filelistener {nullptr} ;

    PyObject *m_ScriptDataEngineClass      {nullptr} ;   // class implemented in the script to use
                                                         // to instanciate the python controller
    PyObject *m_ScriptDataEngineInstance   {nullptr} ;   // instance of m_ScriptControllerClass
    PyObject *m_Func_update                {nullptr} ;
    PyObject *m_Func_init                  {nullptr} ;

    virtual void script_update() override;
    virtual void script_init() override;
    virtual void loadScript() override;
    void init() override;


public:
    sofa::core::objectmodel::DataFileName       m_filename;
    sofa::core::objectmodel::Data<std::string>  m_classname;
    sofa::core::objectmodel::Data< helper::vector< std::string > >  m_variables; /// array of string variables (equivalent to a c-like argv), while waiting to have a better way to share variables
    sofa::core::objectmodel::Data<bool>         m_timingEnabled;
    sofa::core::objectmodel::Data<bool>         m_doAutoReload;
    sofa::core::objectmodel::Data<bool>         d_doUpdate;
    // Inputs
    sofa::core::objectmodel::Data<vector<Tetra> > d_tetrahedra;
    sofa::core::objectmodel::Data<defaulttype::Vec3dTypes::VecCoord> d_X0;
    // Outputs
    sofa::core::objectmodel::Data<vector<Tetra> > d_tetrahedraComputed; // Consider that the goal is to use the script to compute a set of tetrahedra
    sofa::core::objectmodel::Data<vector<Tetra> > d_tetrahedraOutliers; // Consider that the goal is to use the script to compute a set of tetrahedra
    sofa::core::objectmodel::Data<SetIndex> d_tetrahedronIndices;
};

}
}
}

#endif // PYTHONSCRIPTDATAENGINE_H
