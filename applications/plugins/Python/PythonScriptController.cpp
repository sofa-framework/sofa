#include "PythonScriptController.h"
#include <sofa/core/ObjectFactory.h>

#include <boost/python.hpp>
using namespace boost::python;

namespace sofa
{

namespace component
{

namespace controller
{


int PythonScriptControllerClass = core::RegisterObject("A python component experiment")
        .add< PythonScriptController >()
        .addAlias("Python")
        ;

SOFA_DECL_CLASS(PythonController)



PythonScriptController::PythonScriptController()
    : ScriptController()
    , m_filename(initData(&m_filename, "filename","Python script filename"))
    , m_Script(0)
{
    // various initialization stuff here...
}




void PythonScriptController::loadScript()
{
    if (m_Script)
    {
        std::cout << getName() << "load ignored: script already loaded." << std::endl;
    }
    m_Script = sofa::simulation::PythonEnvironment::importScript(m_filename.getFullPath().c_str());
    if (!m_Script)
    {
        // LOAD ERROR
        return;
    }

    // TODO
    // binder les diférents points d'entrée du script

    // pDict is a borrowed reference; no need to release it
    m_ScriptDict = PyModule_GetDict(m_Script);

    // pFunc is also a borrowed reference

#define BIND_SCRIPT_FUNC(funcName) { m_Func_##funcName = PyDict_GetItemString(m_ScriptDict,#funcName); if (!PyCallable_Check(m_Func_##funcName)) m_Func_##funcName=0; }

    m_Func_onBeginAnimationStep = PyDict_GetItemString(m_ScriptDict, "onBeginAnimationStep");
    if (!PyCallable_Check(m_Func_onBeginAnimationStep)) m_Func_onBeginAnimationStep=0;

    m_Func_onLoaded = PyDict_GetItemString(m_ScriptDict, "onLoaded");
    if (!PyCallable_Check(m_Func_onLoaded)) m_Func_onLoaded=0;

    m_Func_createGraph = PyDict_GetItemString(m_ScriptDict, "createGraph");
    if (!PyCallable_Check(m_Func_createGraph)) m_Func_createGraph=0;

    //m_Func_initGraph = PyDict_GetItemString(m_ScriptDict, "initGraph");
    //if (!PyCallable_Check(m_Func_initGraph)) m_Func_initGraph=0;

    BIND_SCRIPT_FUNC(initGraph)

}



void PythonScriptController::script_onLoaded(simulation::tree::GNode *node)
{
    if (m_Func_onLoaded)
        try
        {
            boost::python::call<int>(m_Func_onLoaded,boost::ref(*node));
        }
        catch (const error_already_set e)
        {
            printf("<PYTHON> exception\n");
            PyErr_Print();
        }
}

void PythonScriptController::script_createGraph(simulation::tree::GNode *node)
{
    if (m_Func_createGraph)
        try
        {
            boost::python::call<int>(m_Func_createGraph,boost::ref(*node));
        }
        catch (const error_already_set e)
        {
            printf("<PYTHON> exception\n");
            PyErr_Print();
        }
}

void PythonScriptController::script_initGraph(simulation::tree::GNode *node)
{
    if (m_Func_initGraph)
        try
        {
            boost::python::call<int>(m_Func_initGraph,boost::ref(*node));
        }
        catch (const error_already_set e)
        {
            printf("<PYTHON> exception\n");
            PyErr_Print();
        }
}

void PythonScriptController::script_onBeginAnimationStep(const double dt)
{
    if (m_Func_onBeginAnimationStep)
        try
        {
            boost::python::call<int>(m_Func_onBeginAnimationStep,dt);
        }
        catch (const error_already_set e)
        {
            printf("<PYTHON> exception\n");
            PyErr_Print();
        }

}



} // namespace controller

} // namespace component

} // namespace sofa

