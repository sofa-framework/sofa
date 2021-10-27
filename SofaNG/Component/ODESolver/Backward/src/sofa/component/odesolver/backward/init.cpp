#include <sofa/component/odesolver/backward/init.h>

namespace sofa::component::odesolver::backward
{
	
extern "C" {
    SOFACOMPONENTODESOLVERBACKWARD_API void initExternalModule();
    SOFACOMPONENTODESOLVERBACKWARD_API const char* getModuleName();
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

} // namespace sofa::component::odesolver::backward
