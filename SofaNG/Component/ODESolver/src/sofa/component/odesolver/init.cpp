#include <sofa/component/odesolver/config.h>

namespace sofa::component::odesolver
{
	
extern "C" {
    SOFACOMPONENTODESOLVER_API void initExternalModule();
    SOFACOMPONENTODESOLVER_API const char* getModuleName();
}

void initExternalModule()
{
	static bool first = true;
	if (first)
	{
		first = false;
	}
}

const char* getModuleName()
{
	return MODULE_NAME;
}

} // namespace sofa::component::odesolver
