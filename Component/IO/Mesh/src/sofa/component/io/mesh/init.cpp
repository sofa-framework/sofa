#include <sofa/component/io/mesh/config.h>

namespace sofa::component::io::mesh
{

extern "C" {
    SOFACOMPONENTIOMESH_API void initExternalModule();
    SOFACOMPONENTIOMESH_API const char* getModuleName();
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

} // namespace sofa::component::io::mesh
