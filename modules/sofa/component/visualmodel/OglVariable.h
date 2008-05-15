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

    virtual void init() { OglShaderElement::init(); };
    virtual void initVisual() { }
    virtual void reinit() { init(); initVisual(); }
};

/** SINGLE INT VARIABLE **/
class OglIntVariable : public OglVariable
{
private:
    Data<int> value;

public:
    OglIntVariable();
    virtual ~OglIntVariable() { };

    void initVisual();
};

class OglInt2Variable : public OglVariable
{
private:
    Data<defaulttype::Vec<2, int> > value;

public:
    OglInt2Variable();
    virtual ~OglInt2Variable() { };

    void initVisual();
};

class OglInt3Variable : public OglVariable
{
private:
    Data<defaulttype::Vec<3, int> > value;

public:
    OglInt3Variable();
    virtual ~OglInt3Variable() { };

    void initVisual();
};

class OglInt4Variable : public OglVariable
{
private:
    Data<defaulttype::Vec<4, int> > value;

public:
    OglInt4Variable();
    virtual ~OglInt4Variable() { };

    void initVisual();
};

/** SINGLE FLOAT VARIABLE **/

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

class OglFloat4Variable : public OglVariable
{
private:
    Data<defaulttype::Vec4f> value;

public:
    OglFloat4Variable();
    virtual ~OglFloat4Variable() { };

    void initVisual();
};

/** INT VECTOR VARIABLE **/
class OglIntVectorVariable : public OglVariable
{
protected:
    Data<helper::vector<GLint> > iv;

public:
    OglIntVectorVariable();
    virtual ~OglIntVectorVariable() { };

    virtual void init();
    virtual void initVisual();
};

class OglIntVector2Variable : public OglIntVectorVariable
{

public:
    OglIntVector2Variable();
    virtual ~OglIntVector2Variable() { };

    virtual void init();
    virtual void initVisual();
};

class OglIntVector3Variable : public OglIntVectorVariable
{
public:
    OglIntVector3Variable();
    virtual ~OglIntVector3Variable() { };

    virtual void init();
    virtual void initVisual();
};

class OglIntVector4Variable : public OglIntVectorVariable
{
public:
    OglIntVector4Variable();
    virtual ~OglIntVector4Variable() { };

    virtual void init();
    virtual void initVisual();
};

/** FLOAT VECTOR VARIABLE **/
class OglFloatVectorVariable : public OglVariable
{
protected:
    Data<helper::vector<float> > fv;

public:
    OglFloatVectorVariable();
    virtual ~OglFloatVectorVariable() { };

    virtual void init();
    virtual void initVisual();
};

class OglFloatVector2Variable : public OglFloatVectorVariable
{
public:
    OglFloatVector2Variable();
    virtual ~OglFloatVector2Variable() { };

    virtual void init();
    virtual void initVisual();
};

class OglFloatVector3Variable : public OglFloatVectorVariable
{
public:
    OglFloatVector3Variable();
    virtual ~OglFloatVector3Variable() { };

    virtual void init();
    virtual void initVisual();
};

class OglFloatVector4Variable : public OglFloatVectorVariable
{
public:
    OglFloatVector4Variable();
    virtual ~OglFloatVector4Variable() { };

    virtual void init();
    virtual void initVisual();
};

}

}

}

#endif /*OGLVARIABLE_H_*/
