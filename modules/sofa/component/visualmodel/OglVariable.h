#ifndef OGLVARIABLE_H_
#define OGLVARIABLE_H_

#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>
#include <sofa/component/visualmodel/OglShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{


class OglVariable : public core::VisualModel, public OglShaderElement
{
private:

public:
    OglVariable() { };
    virtual ~OglVariable() { };

    virtual void initVisual() { }
    virtual void reinit() { initVisual(); }
};

class OglFloatVariable : public OglVariable
{
private:
    Data<float> value;

public:
    OglFloatVariable();
    virtual ~OglFloatVariable() { };

    void initVisual();
};

class OglFloat2Variable : public OglVariable
{
private:
    Data<defaulttype::Vec2f> value;

public:
    OglFloat2Variable();
    virtual ~OglFloat2Variable() { };

    void initVisual();
};

class OglFloat3Variable : public OglVariable
{
private:
    Data<defaulttype::Vec3f> value;

public:
    OglFloat3Variable();
    virtual ~OglFloat3Variable() { };

    void initVisual();
};

}

}

}

#endif /*OGLVARIABLE_H_*/
