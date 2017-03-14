#include "PythonEnvironment.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{


class PythonModuleReload : public core::objectmodel::BaseObject
{

public:
    SOFA_CLASS(PythonModuleReload, core::objectmodel::BaseObject);

    virtual void cleanup()
    {
        PyRun_SimpleString("SofaPython.unloadModules()");
    }

};

int PythonModuleReloadClass = core::RegisterObject("Force reloading python modules when (re)loading a scene")
        .add< PythonModuleReload >()
        ;

SOFA_DECL_CLASS(PythonModuleReload)

} // namespace sofa
