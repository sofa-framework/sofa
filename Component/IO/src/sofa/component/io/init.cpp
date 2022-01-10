#include <sofa/component/io/config.h>

namespace sofa::component::io
{

extern "C" {
    SOFACOMPONENTIO_API void initExternalModule();
    SOFACOMPONENTIO_API const char* getModuleName();
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

} // namespace sofa::component::io
