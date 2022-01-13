#include <sofa/component/utility/init.h>

namespace sofa::component::utility
{

void init()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

extern "C" {
    SOFA_COMPONENT_UTILITY_API void initExternalModule();
    SOFA_COMPONENT_UTILITY_API const char* getModuleName();
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return sofa::component::utility::MODULE_NAME;
}

} // namespace sofa::component::utility
