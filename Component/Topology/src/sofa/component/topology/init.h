#include <sofa/component/topology/config.h>

namespace sofa::component::topology
{

extern "C" {
	SOFACOMPONENTTOPOLOGY_API void initExternalModule();
	SOFACOMPONENTTOPOLOGY_API const char* getModuleName();
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

} // namespace sofa::component::topology
