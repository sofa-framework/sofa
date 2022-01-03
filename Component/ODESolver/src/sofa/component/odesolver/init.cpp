#include <sofa/component/odesolver/init.h>

namespace sofa::component::odesolver
{
	
extern "C" {
    SOFA_EXPORT_DYNAMIC_LIBRARY void initExternalModule();
	SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName();
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

void init()
{
	initExternalModule();
}

} // namespace sofa::component::odesolver
