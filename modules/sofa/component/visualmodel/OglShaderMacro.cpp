#include "OglShaderMacro.h"
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(OglShaderDefineMacro)

//Register OglIntVariable in the Object Factory
int OglShaderDefineMacroClass = core::RegisterObject("OglShaderDefineMacro")
        .add< OglShaderDefineMacro >();

OglShaderMacro::OglShaderMacro()
{

}

OglShaderMacro::~OglShaderMacro()
{
}

void OglShaderMacro::init()
{
    OglShaderElement::init();
}

OglShaderDefineMacro::OglShaderDefineMacro()
    : value(initData(&value, (std::string) "", "value", "Set a value for define macro"))
{

}

OglShaderDefineMacro::~OglShaderDefineMacro()
{
}

void OglShaderDefineMacro::init()
{
    OglShaderMacro::init();

    shader->addDefineMacro(id.getValue(), value.getValue());
}

}

}

}
