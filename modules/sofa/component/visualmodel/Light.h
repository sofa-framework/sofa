#ifndef SOFA_COMPONENT_LIGHT
#define SOFA_COMPONENT_LIGHT

#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/VisualModel.h>
namespace sofa
{

namespace component
{

namespace visualmodel
{

using sofa::defaulttype::Vector3;

class Light : public virtual sofa::core::VisualModel
{
protected:
    Data<Vector3> color;
    GLint lightID;

public:

    Light();
    virtual ~Light();

    void setID(const GLint& id);

    virtual void initVisual() ;
    void init();
    virtual void drawLight();
    void draw() { } ;
    virtual void reinit();
    void update() {} ;
};

class DirectionalLight : public Light
{
private:
    Data<Vector3> direction;

public:

    DirectionalLight();
    virtual ~DirectionalLight();
    virtual void initVisual() ;
    virtual void drawLight();
    virtual void reinit();


};

class PositionalLight : public Light
{
protected:
    Data<Vector3> position;
    Data<float> attenuation;

public:

    PositionalLight();
    virtual ~PositionalLight();
    virtual void initVisual() ;
    virtual void drawLight();
    virtual void reinit();

};

class SpotLight : public PositionalLight
{
protected:
    Data<Vector3> direction;
    Data<float> cutoff;
    Data<float> exponent;

public:
    SpotLight();
    virtual ~SpotLight();
    virtual void initVisual() ;
    virtual void drawLight();
    virtual void reinit();


};

} //namespace visualmodel

} //namespace component

} //namespace sofa

#endif //SOFA_COMPONENT_LIGHT
