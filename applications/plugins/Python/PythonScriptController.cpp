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

    // binder les différents points d'entrée du script

    // pDict is a borrowed reference; no need to release it
    m_ScriptDict = PyModule_GetDict(m_Script);

    // functions are also borrowed references

#define BIND_SCRIPT_FUNC(funcName) { m_Func_##funcName = PyDict_GetItemString(m_ScriptDict,#funcName); if (!PyCallable_Check(m_Func_##funcName)) m_Func_##funcName=0; }

    BIND_SCRIPT_FUNC(onLoaded)
    BIND_SCRIPT_FUNC(createGraph)
    BIND_SCRIPT_FUNC(initGraph)
    BIND_SCRIPT_FUNC(onBeginAnimationStep)
    BIND_SCRIPT_FUNC(storeResetState)
    BIND_SCRIPT_FUNC(reset)
    BIND_SCRIPT_FUNC(cleanup)

}

#define CALL_SCRIPT_FUNC0(funcName) {if(m_Func_##funcName) try{boost::python::call<int>(m_Func_##funcName); } catch (const error_already_set e) { printf("<PYTHON> exception\n"); PyErr_Print(); } }
#define CALL_SCRIPT_FUNC1(funcName,param) {if(m_Func_##funcName) try{boost::python::call<int>(m_Func_##funcName,param); } catch (const error_already_set e) { printf("<PYTHON> exception\n"); PyErr_Print(); } }

void PythonScriptController::script_onLoaded(simulation::tree::GNode *node)
{
    CALL_SCRIPT_FUNC1(onLoaded,boost::ref(*node))
}

void PythonScriptController::script_createGraph(simulation::tree::GNode *node)
{
    CALL_SCRIPT_FUNC1(createGraph,boost::ref(*node))
}

void PythonScriptController::script_initGraph(simulation::tree::GNode *node)
{
    CALL_SCRIPT_FUNC1(initGraph,boost::ref(*node))
}

void PythonScriptController::script_onBeginAnimationStep(const double dt)
{
    CALL_SCRIPT_FUNC1(onBeginAnimationStep,dt)
}

void PythonScriptController::script_storeResetState()
{
    printf("PythonScriptController::script_storeResetState\n");
    CALL_SCRIPT_FUNC0(storeResetState)
}

void PythonScriptController::script_reset()
{
    printf("PythonScriptController::script_reset\n");
    CALL_SCRIPT_FUNC0(reset)
}

void PythonScriptController::script_cleanup()
{
    printf("PythonScriptController::script_cleanup\n");
    CALL_SCRIPT_FUNC0(cleanup)
}



} // namespace controller

} // namespace component

} // namespace sofa

