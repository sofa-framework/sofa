#include <sofa/component/visual/config.h>

namespace sofa::component::visual
{

extern "C" {
    SOFACOMPONENTVISUAL_API void initExternalModule();
    SOFACOMPONENTVISUAL_API const char* getModuleName();
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

} // namespace sofa::component::visual
