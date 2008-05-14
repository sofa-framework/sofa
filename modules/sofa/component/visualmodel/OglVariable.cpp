#include <sofa/component/visualmodel/OglVariable.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(OglFloatVariable)
SOFA_DECL_CLASS(OglFloat2Variable)
SOFA_DECL_CLASS(OglFloat3Variable)

//Register OglFloatVariable in the Object Factory
int OglFloatVariableClass = core::RegisterObject("OglFloatVariable")
        .add< OglFloatVariable >()
        ;
//Register OglFloat2Variable in the Object Factory
int OglFloat2VariableClass = core::RegisterObject("OglFloat2Variable")
        .add< OglFloat2Variable >()
        ;
//Register OglFloat3Variable in the Object Factory
int OglFloat3VariableClass = core::RegisterObject("OglFloat3Variable")
        .add< OglFloat3Variable >()
        ;

OglFloatVariable::OglFloatVariable()
    : value(initData(&value, (float) 0.0, "value", "Set a float value"))
{

}

OglFloat2Variable::OglFloat2Variable()
    : value(initData(&value, (defaulttype::Vec2f) defaulttype::Vec2f(0.0,0.0) , "value", "Set a Vec2f value"))
{

}

OglFloat3Variable::OglFloat3Variable()
    : value(initData(&value, (defaulttype::Vec3f) defaulttype::Vec3f(0.0,0.0,0.0) , "value", "Set a Vec3f value"))
{

}

void OglFloatVariable::initVisual()
{
    shader->setFloat(id.getValue().c_str(), value.getValue());
}


void OglFloat2Variable::initVisual()
{
    shader->setFloat2(id.getValue().c_str(), value.getValue()[0], value.getValue()[1]);
}

void OglFloat3Variable::initVisual()
{
    shader->setFloat3(id.getValue().c_str(), value.getValue()[0], value.getValue()[1], value.getValue()[2]);
}

}

}

}
