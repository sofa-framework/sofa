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
        std::cout << getName() << " load ignored: script already loaded." << std::endl;
    }
    m_Script = sofa::simulation::PythonEnvironment::importScript(m_filename.getFullPath().c_str());
    if (!m_Script)
    {
        // LOAD ERROR
        std::cout << "<PYTHON> ERROR : "<<getName() << " object - "<<m_filename.getFullPath().c_str()<<" script load error." << std::endl;
        return;
    }

    // binder les différents points d'entrée du script

    // pDict is a borrowed reference; no need to release it
    m_ScriptDict = PyModule_GetDict(m_Script);

    // functions are also borrowed references
    if (!m_ScriptDict)
    {
        // LOAD ERROR
        std::cout << getName() << " load error (dictionnary not found)." << std::endl;
        return;
    }

#define BIND_SCRIPT_FUNC(funcName) \
    { \
        m_Func_##funcName = PyDict_GetItemString(m_ScriptDict,#funcName); \
        if (!PyCallable_Check(m_Func_##funcName)) m_Func_##funcName=0; \
    }

//    std::cout << "Binding functions of script \"" << m_filename.getFullPath().c_str() << "\"" << std::endl;
//    std::cout << "Number of dictionnay entries: "<< PyDict_Size(m_ScriptDict) << std::endl;
    BIND_SCRIPT_FUNC(onLoaded)
    BIND_SCRIPT_FUNC(createGraph)
    BIND_SCRIPT_FUNC(initGraph)
    BIND_SCRIPT_FUNC(onBeginAnimationStep)
    BIND_SCRIPT_FUNC(storeResetState)
    BIND_SCRIPT_FUNC(reset)
    BIND_SCRIPT_FUNC(cleanup)
    BIND_SCRIPT_FUNC(onGUIEvent)

}

#define CALL_SCRIPT_FUNC0(funcName) {if(m_Func_##funcName) try{boost::python::call<int>(m_Func_##funcName); } catch (const error_already_set e) { printf("<PYTHON> exception\n"); PyErr_Print(); } }
#define CALL_SCRIPT_FUNC1(funcName,param) {if(m_Func_##funcName) try{boost::python::call<int>(m_Func_##funcName,param); } catch (const error_already_set e) { printf("<PYTHON> exception\n"); PyErr_Print(); } }
#define CALL_SCRIPT_FUNC2(funcName,param1,param2) {if(m_Func_##funcName) try{boost::python::call<int>(m_Func_##funcName,param1,param2); } catch (const error_already_set e) { printf("<PYTHON> exception\n"); PyErr_Print(); } }
#define CALL_SCRIPT_FUNC3(funcName,param1,param2,param3) {if(m_Func_##funcName) try{boost::python::call<int>(m_Func_##funcName,param1,param2,param3); } catch (const error_already_set e) { printf("<PYTHON> exception\n"); PyErr_Print(); } }

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
    CALL_SCRIPT_FUNC0(storeResetState)
}

void PythonScriptController::script_reset()
{
    CALL_SCRIPT_FUNC0(reset)
}

void PythonScriptController::script_cleanup()
{
    CALL_SCRIPT_FUNC0(cleanup)
}

void PythonScriptController::script_onGUIEvent(const char* controlID, const char* valueName, const char* value)
{
    CALL_SCRIPT_FUNC3(onGUIEvent,controlID,valueName,value)
}



} // namespace controller

} // namespace component

} // namespace sofa

