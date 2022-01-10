#include <sofa/component/io/mesh/config.h>

namespace sofa::component::io::mesh
{

extern "C" {
	SOFA_COMPONENT_IO_MESH_API void initExternalModule();
	SOFA_COMPONENT_IO_MESH_API const char* getModuleName();
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
