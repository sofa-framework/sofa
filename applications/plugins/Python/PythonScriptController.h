#ifndef PYTHONCONTROLLER_H
#define PYTHONCONTROLLER_H

#include "PythonEnvironment.h"
#include "ScriptController.h"
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/loader/BaseLoader.h>

namespace sofa
{

namespace component
{

namespace controller
{

class PythonScriptController : public ScriptController
{
public:
    SOFA_CLASS(PythonScriptController,ScriptController);

protected:
    PythonScriptController();

    /// @name Script interface
    ///   Function that need to be implemented for each script language
    /// Typically, all "script_*" functions call the corresponding "*" function of the script, if it exists
    /// @{

    virtual void loadScript();

    virtual void script_onLoaded(sofa::simulation::tree::GNode* node);   // called once, immediately after the script is loaded
    virtual void script_createGraph(sofa::simulation::tree::GNode* node);       // called when the script must create its graph
    virtual void script_initGraph(sofa::simulation::tree::GNode* node);         // called when the script must init its graph, once all the graph has been create

    virtual void script_onBeginAnimationStep(const double dt);
    /// @}


public:
    sofa::core::objectmodel::DataFileName m_filename;


protected:
    PyObject *m_Script;         // python script module
    PyObject *m_ScriptDict;     // functions dictionnary

    // optionnal script entry points:
    PyObject *m_Func_onBeginAnimationStep;
    PyObject *m_Func_onLoaded;
    PyObject *m_Func_createGraph;
    PyObject *m_Func_initGraph;
public:

};


} // namespace controller

} // namespace component

} // namespace sofa

#endif // PYTHONCONTROLLER_H
