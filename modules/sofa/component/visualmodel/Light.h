// Author: The SOFA team </www.sofa-framework.org>, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_LIGHT
#define SOFA_COMPONENT_LIGHT

#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

using sofa::defaulttype::Vector3;

//Singleton class
class LightTable
{
private:
    static const unsigned int MAX_NUMBER_OF_LIGHTS = GL_MAX_LIGHTS;
    std::map<std::string, GLint> lightTable;
    static LightTable* instance;

    LightTable() { };

public:

    static LightTable* getInstance()
    {
        if (instance == NULL)
            instance = new LightTable();
        return instance;
    }

    GLint getAvailableLightID(const std::string& name)
    {
        if (lightTable.size() >= MAX_NUMBER_OF_LIGHTS)
            return 0;

        GLint temp = GL_LIGHT0 + lightTable.size();
        lightTable[name] = temp;
        return lightTable[name];
    }

    void removeLightID(const std::string& name)
    {
        lightTable.erase(name);
    }

};

class Light : public core::VisualModel
{
protected:
    Data<Vector3> color;
    GLint lightID;

public:

    Light();
    virtual ~Light();

    virtual void initTextures() ;
    void init();
    virtual void draw();
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
    virtual void initTextures() ;
    virtual void draw();
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
    virtual void initTextures() ;
    virtual void draw();
    virtual void reinit();

};

class SpotLight : public PositionalLight
{
protected:
    Data<Vector3> direction;
    Data<float> cutoff;

public:
    SpotLight();
    virtual ~SpotLight();
    virtual void initTextures() ;
    virtual void draw();
    virtual void reinit();


};

} //namespace component

} //namespace sofa

#endif //SOFA_COMPONENT_LIGHT
