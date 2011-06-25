/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/visualmodel/OglVariable.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

/** SINGLE INT VARIABLE **/
SOFA_DECL_CLASS(OglIntVariable)
SOFA_DECL_CLASS(OglInt2Variable)
SOFA_DECL_CLASS(OglInt3Variable)
SOFA_DECL_CLASS(OglInt4Variable)

//Register OglIntVariable in the Object Factory
int OglIntVariableClass = core::RegisterObject("OglIntVariable")
        .add< OglIntVariable >()
        ;
//Register OglInt2Variable in the Object Factory
int OglInt2VariableClass = core::RegisterObject("OglInt2Variable")
        .add< OglInt2Variable >()
        ;
//Register OglInt3Variable in the Object Factory
int OglInt3VariableClass = core::RegisterObject("OglInt3Variable")
        .add< OglInt3Variable >()
        ;
//Register OglInt4Variable in the Object Factory
int OglInt4VariableClass = core::RegisterObject("OglInt4Variable")
        .add< OglInt4Variable >()
        ;

/** SINGLE FLOAT VARIABLE **/

SOFA_DECL_CLASS(OglFloatVariable)
SOFA_DECL_CLASS(OglFloat2Variable)
SOFA_DECL_CLASS(OglFloat3Variable)
SOFA_DECL_CLASS(OglFloat4Variable)

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
//Register OglFloat4Variable in the Object Factory
int OglFloat4VariableClass = core::RegisterObject("OglFloat4Variable")
        .add< OglFloat4Variable >()
        ;

/** INT VECTOR VARIABLE **/
SOFA_DECL_CLASS(OglIntVectorVariable)
SOFA_DECL_CLASS(OglIntVector2Variable)
SOFA_DECL_CLASS(OglIntVector3Variable)
SOFA_DECL_CLASS(OglIntVector4Variable)

//Register OglIntVectorVariable in the Object Factory
int OglIntVectorVariableClass = core::RegisterObject("OglIntVectorVariable")
        .add< OglIntVectorVariable >()
        ;

//Register OglIntVector2Variable in the Object Factory
int OglIntVector2VariableClass = core::RegisterObject("OglIntVector2Variable")
        .add< OglIntVector2Variable >()
        ;

//Register OglIntVector3Variable in the Object Factory
int OglIntVector3VariableClass = core::RegisterObject("OglIntVector3Variable")
        .add< OglIntVector3Variable >()
        ;

//Register OglIntVector4Variable in the Object Factory
int OglIntVector4VariableClass = core::RegisterObject("OglIntVector4Variable")
        .add< OglIntVector4Variable >()
        ;


/** FLOAT VECTOR VARIABLE **/
SOFA_DECL_CLASS(OglFloatVectorVariable)
SOFA_DECL_CLASS(OglFloatVector2Variable)
SOFA_DECL_CLASS(OglFloatVector3Variable)
SOFA_DECL_CLASS(OglFloatVector4Variable)

//Register OglFloatVectorVariable in the Object Factory
int OglFloatVectorVariableClass = core::RegisterObject("OglFloatVectorVariable")
        .add< OglFloatVectorVariable >()
        ;

//Register OglFloatVector2Variable in the Object Factory
int OglFloatVector2VariableClass = core::RegisterObject("OglFloatVector2Variable")
        .add< OglFloatVector2Variable >()
        ;

//Register OglFloatVector3Variable in the Object Factory
int OglFloatVector3VariableClass = core::RegisterObject("OglFloatVector3Variable")
        .add< OglFloatVector3Variable >()
        ;
//Register OglFloatVector4Variable in the Object Factory
int OglFloatVector4VariableClass = core::RegisterObject("OglFloatVector4Variable")
        .add< OglFloatVector4Variable >()
        ;

/** FLOAT VECTOR VARIABLE **/
SOFA_DECL_CLASS(OglMatrix2Variable)
SOFA_DECL_CLASS(OglMatrix3Variable)
SOFA_DECL_CLASS(OglMatrix4Variable)
SOFA_DECL_CLASS(OglMatrix2x3Variable)
SOFA_DECL_CLASS(OglMatrix3x2Variable)
SOFA_DECL_CLASS(OglMatrix2x4Variable)
SOFA_DECL_CLASS(OglMatrix4x2Variable)
SOFA_DECL_CLASS(OglMatrix3x4Variable)
SOFA_DECL_CLASS(OglMatrix4x3Variable)

//Register OglMatrix2Variable in the Object Factory
int OglMatrix2VariableClass = core::RegisterObject("OglMatrix2Variable")
        .add< OglMatrix2Variable >()
        ;

//Register OglMatrix3Variable in the Object Factory
int OglMatrix3VariableClass = core::RegisterObject("OglMatrix3Variable")
        .add< OglMatrix3Variable >()
        ;

//Register OglMatrix4Variable in the Object Factory
int OglMatrix4VariableClass = core::RegisterObject("OglMatrix4Variable")
        .add< OglMatrix4Variable >()
        ;

//Register OglMatrix2x3Variable in the Object Factory
int OglMatrix2x3VariableClass = core::RegisterObject("OglMatrix2x3Variable")
        .add< OglMatrix2x3Variable >()
        ;

//Register OglMatrix3x2Variable in the Object Factory
int OglMatrix3x2VariableClass = core::RegisterObject("OglMatrix3x2Variable")
        .add< OglMatrix3x2Variable >()
        ;

//Register OglMatrix2x4Variable in the Object Factory
int OglMatrix2x4VariableClass = core::RegisterObject("OglMatrix2x4Variable")
        .add< OglMatrix2x4Variable >()
        ;

//Register OglMatrix4x2Variable in the Object Factory
int OglMatrix4x2VariableClass = core::RegisterObject("OglMatrix4x2Variable")
        .add< OglMatrix4x2Variable >()
        ;

//Register OglMatrix2x4Variable in the Object Factory
int OglMatrix3x4VariableClass = core::RegisterObject("OglMatrix3x4Variable")
        .add< OglMatrix3x4Variable >()
        ;

//Register OglMatrix4x3Variable in the Object Factory
int OglMatrix4x3VariableClass = core::RegisterObject("OglMatrix4x3Variable")
        .add< OglMatrix4x3Variable >()
        ;



OglIntVariable::OglIntVariable()
{

}

OglInt2Variable::OglInt2Variable()
{

}

OglInt3Variable::OglInt3Variable()
{

}

OglInt4Variable::OglInt4Variable()
{

}

void OglIntVariable::initVisual()
{
    shader->setInt(indexShader.getValue(), id.getValue().c_str(), value.getValue());
}


void OglInt2Variable::initVisual()
{
    shader->setInt2(indexShader.getValue(), id.getValue().c_str(), value.getValue()[0], value.getValue()[1]);
}

void OglInt3Variable::initVisual()
{
    shader->setInt3(indexShader.getValue(), id.getValue().c_str(), value.getValue()[0], value.getValue()[1], value.getValue()[2]);
}

void OglInt4Variable::initVisual()
{
    shader->setInt4(indexShader.getValue(), id.getValue().c_str(), value.getValue()[0], value.getValue()[1], value.getValue()[2], value.getValue()[3]);
}


OglFloatVariable::OglFloatVariable()
{

}

OglFloat2Variable::OglFloat2Variable()
{

}

OglFloat3Variable::OglFloat3Variable()
{

}

OglFloat4Variable::OglFloat4Variable()
{

}

void OglFloatVariable::initVisual()
{
    shader->setFloat(indexShader.getValue(), id.getValue().c_str(), value.getValue());
}


void OglFloat2Variable::initVisual()
{
    shader->setFloat2(indexShader.getValue(), id.getValue().c_str(), value.getValue()[0], value.getValue()[1]);
}

void OglFloat3Variable::initVisual()
{
    shader->setFloat3(indexShader.getValue(), id.getValue().c_str(), value.getValue()[0], value.getValue()[1], value.getValue()[2]);
}

void OglFloat4Variable::initVisual()
{
    shader->setFloat4(indexShader.getValue(), id.getValue().c_str(), value.getValue()[0], value.getValue()[1], value.getValue()[2], value.getValue()[3]);
}


OglIntVectorVariable::OglIntVectorVariable()
{

}

OglIntVector2Variable::OglIntVector2Variable()
{

}

OglIntVector3Variable::OglIntVector3Variable()
{

}

OglIntVector4Variable::OglIntVector4Variable()
{

}


void OglIntVectorVariable::init()
{
    OglVariable<helper::vector<GLint> >::init();
}

void OglIntVector2Variable::init()
{
    OglIntVectorVariable::init();
    helper::vector<GLint> temp = value.getValue();
    if (value.getValue().size() %2 != 0)
    {
        serr << "The number of values is not even ; padding with one zero" << sendl;
        temp.push_back(0);
        value.setValue(temp);

    }
}

void OglIntVector3Variable::init()
{
    OglIntVectorVariable::init();
    helper::vector<GLint> temp = value.getValue();

    if (value.getValue().size() %3 != 0)
    {
        serr << "The number of values is not a multiple of 3 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %3 != 0)
            temp.push_back(0);
        value.setValue(temp);
    }
}

void OglIntVector4Variable::init()
{
    OglIntVectorVariable::init();
    helper::vector<GLint> temp = value.getValue();

    if (value.getValue().size() %4 != 0)
    {
        serr << "The number of values is not a multiple of 4 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %4 != 0)
            temp.push_back(0);
        value.setValue(temp);
    }
}

void OglIntVectorVariable::initVisual()
{
    shader->setIntVector(indexShader.getValue(), id.getValue().c_str(), value.getValue().size(), &(value.getValue()[0]));
}

void OglIntVector2Variable::initVisual()
{
    shader->setIntVector2(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/2, &(value.getValue()[0]));
}

void OglIntVector3Variable::initVisual()
{
    shader->setIntVector3(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/3, &(value.getValue()[0]));
}

void OglIntVector4Variable::initVisual()
{
    shader->setIntVector4(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/4, &(value.getValue()[0]));
}


OglFloatVectorVariable::OglFloatVectorVariable()
{

}

OglFloatVector2Variable::OglFloatVector2Variable()
{

}

OglFloatVector3Variable::OglFloatVector3Variable()
{

}

OglFloatVector4Variable::OglFloatVector4Variable()
{

}


void OglFloatVectorVariable::init()
{
    OglVariable<helper::vector<float> >::init();
}

void OglFloatVector2Variable::init()
{
    OglFloatVectorVariable::init();
    helper::vector<float> temp = value.getValue();
    if (value.getValue().size() %2 != 0)
    {
        serr << "The number of values is not even ; padding with one zero" << sendl;
        temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglFloatVector3Variable::init()
{
    OglFloatVectorVariable::init();
    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %3 != 0)
    {
        serr << "The number of values is not a multiple of 3 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %3 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglFloatVector4Variable::init()
{
    OglFloatVectorVariable::init();
    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %4 != 0)
    {
        serr << "The number of values is not a multiple of 4 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %4 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglFloatVectorVariable::initVisual()
{
    shader->setFloatVector(indexShader.getValue(), id.getValue().c_str(), value.getValue().size(), &(value.getValue()[0]));
}

void OglFloatVector2Variable::initVisual()
{
    shader->setFloatVector2(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/2, &(value.getValue()[0]));
}

void OglFloatVector3Variable::initVisual()
{
    shader->setFloatVector3(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/3, &(value.getValue()[0]));
}

void OglFloatVector4Variable::initVisual()
{
    shader->setFloatVector4(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/4, &(value.getValue()[0]));
}


//// Matrix /////
OglMatrix2Variable::OglMatrix2Variable()
{

}

void OglMatrix2Variable::init()
{
    OglVariable<helper::vector<float> >::init();

    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %4 != 0)
    {
        serr << "The number of values is not a multiple of 4 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %4 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix2Variable::initVisual()
{
    shader->setMatrix2(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/4, transpose.getValue(), &(value.getValue()[0]));
}

OglMatrix3Variable::OglMatrix3Variable()
{

}

void OglMatrix3Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %9 != 0)
    {
        serr << "The number of values is not a multiple of 9 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %9 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix3Variable::initVisual()
{
    shader->setMatrix3(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/9, transpose.getValue(), &(value.getValue()[0]));
}

OglMatrix4Variable::OglMatrix4Variable()
{

}

void OglMatrix4Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %16 != 0)
    {
        serr << "The number of values is not a multiple of 16 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %16 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix4Variable::initVisual()
{
    shader->setMatrix4(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/16, transpose.getValue(), &(value.getValue()[0]));
}

OglMatrix2x3Variable::OglMatrix2x3Variable()
{

}

void OglMatrix2x3Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %6 != 0)
    {
        serr << "The number of values is not a multiple of 6 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %6 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix2x3Variable::initVisual()
{
    shader->setMatrix2x3(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/6, transpose.getValue(), &(value.getValue()[0]));
}


OglMatrix3x2Variable::OglMatrix3x2Variable()
{

}

void OglMatrix3x2Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %6 != 0)
    {
        serr << "The number of values is not a multiple of 6 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %6 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix3x2Variable::initVisual()
{
    shader->setMatrix3x2(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/6, transpose.getValue(), &(value.getValue()[0]));
}

OglMatrix2x4Variable::OglMatrix2x4Variable()
{

}

void OglMatrix2x4Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %8 != 0)
    {
        serr << "The number of values is not a multiple of 8 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %8 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix2x4Variable::initVisual()
{
    shader->setMatrix2x4(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/8, transpose.getValue(), &(value.getValue()[0]));
}

OglMatrix4x2Variable::OglMatrix4x2Variable()
{

}

void OglMatrix4x2Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %8 != 0)
    {
        serr << "The number of values is not a multiple of 8 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %8 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix4x2Variable::initVisual()
{
    shader->setMatrix4x2(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/8, transpose.getValue(), &(value.getValue()[0]));
}

OglMatrix3x4Variable::OglMatrix3x4Variable()
{

}

void OglMatrix3x4Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %12 != 0)
    {
        serr << "The number of values is not a multiple of 12 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %12 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix3x4Variable::initVisual()
{
    shader->setMatrix3x4(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/12, transpose.getValue(), &(value.getValue()[0]));
}

OglMatrix4x3Variable::OglMatrix4x3Variable()
{

}

void OglMatrix4x3Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (value.getValue().size() %12 != 0)
    {
        serr << "The number of values is not a multiple of 12 ; padding with zero(s)" << sendl;
        while (value.getValue().size() %12 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix4x3Variable::initVisual()
{
    shader->setMatrix4x3(indexShader.getValue(), id.getValue().c_str(), value.getValue().size()/12, transpose.getValue(), &(value.getValue()[0]));
}

} // namespace visual

} // namespace component

} // namespace sofa
