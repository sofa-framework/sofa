#ifndef OGLSHADERMACRO_H_
#define OGLSHADERMACRO_H_

#include <sofa/component/visualmodel/OglShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class OglShaderMacro : public OglShaderElement
{
protected:

public:
    OglShaderMacro();
    virtual ~OglShaderMacro();
    virtual void init();

};


class OglShaderDefineMacro : public OglShaderMacro
{
protected:
    Data<std::string> value;
public:
    OglShaderDefineMacro();
    virtual ~OglShaderDefineMacro();
    virtual void init();
};

}

}

}

#endif /*OGLSHADERMACRO_H_*/
