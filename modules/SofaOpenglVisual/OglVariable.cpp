/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaOpenglVisual/OglVariable.inl>
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

/** Matrix VARIABLE **/
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

/** Matrix vector VARIABLE **/
//Register OglMatrix4VectorVariable in the Object Factory
int OglMatrix4VectorVariableClass = core::RegisterObject("OglMatrix4VectorVariable")
        .add< OglMatrix4VectorVariable >()
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
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const int v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setInt(idShader, idstr.c_str(), v);
}


void OglInt2Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const defaulttype::Vec<2,int>& v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setInt2(idShader, idstr.c_str(), v[0], v[1]);
}

void OglInt3Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const defaulttype::Vec<3,int>& v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setInt3(idShader, idstr.c_str(), v[0], v[1], v[2]);
}

void OglInt4Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const defaulttype::Vec<4,int>& v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setInt4(idShader, idstr.c_str(), v[0], v[1], v[2], v[3]);
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
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const float v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloat(idShader, idstr.c_str(), v);
}


void OglFloat2Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const defaulttype::Vec2f& v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloat2(idShader, idstr.c_str(), v[0], v[1]);
}

void OglFloat3Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const defaulttype::Vec3f& v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloat3(idShader, idstr.c_str(), v[0], v[1], v[2]);
}

void OglFloat4Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const defaulttype::Vec4f& v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloat4(idShader, idstr.c_str(), v[0], v[1], v[2], v[3]);
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
    if (temp.size() %2 != 0)
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

    if (temp.size() %3 != 0)
    {
        serr << "The number of values is not a multiple of 3 ; padding with zero(s)" << sendl;
        while (temp.size() %3 != 0)
            temp.push_back(0);
        value.setValue(temp);
    }
}

void OglIntVector4Variable::init()
{
    OglIntVectorVariable::init();
    helper::vector<GLint> temp = value.getValue();

    if (temp.size() %4 != 0)
    {
        serr << "The number of values is not a multiple of 4 ; padding with zero(s)" << sendl;
        while (temp.size() %4 != 0)
            temp.push_back(0);
        value.setValue(temp);
    }
}

void OglIntVectorVariable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<GLint>& v = value.getValue();
    const GLint* vptr = v.empty() ? NULL : &(v[0]);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setIntVector(idShader, idstr.c_str(), v.size(), vptr);
}

void OglIntVector2Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<GLint>& v = value.getValue();
    const GLint* vptr = v.empty() ? NULL : &(v[0]);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setIntVector2(idShader, idstr.c_str(), v.size()/2, vptr);
}

void OglIntVector3Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<GLint>& v = value.getValue();
    const GLint* vptr = v.empty() ? NULL : &(v[0]);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setIntVector3(idShader, idstr.c_str(), v.size()/3, vptr);
}

void OglIntVector4Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<GLint>& v = value.getValue();
    const GLint* vptr = v.empty() ? NULL : &(v[0]);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setIntVector4(idShader, idstr.c_str(), v.size()/4, vptr);
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
    OglVariable<helper::vector<defaulttype::Vec2f> >::init();
}

void OglFloatVector3Variable::init()
{
    OglVariable<helper::vector<defaulttype::Vec3f> >::init();
}

void OglFloatVector4Variable::init()
{
    OglVariable<helper::vector<defaulttype::Vec4f> >::init();
}

void OglFloatVectorVariable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0]);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloatVector(idShader, idstr.c_str(), v.size(), vptr);
}

void OglFloatVector2Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<defaulttype::Vec2f>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0][0]);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloatVector2(idShader, idstr.c_str(), (int)v.size(), vptr);
}

void OglFloatVector3Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<defaulttype::Vec3f>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0][0]);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloatVector3(idShader, idstr.c_str(), (int)v.size(), vptr);
}

void OglFloatVector4Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<defaulttype::Vec4f>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0][0]);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloatVector4(idShader, idstr.c_str(), (int)v.size(), vptr);
}


//// Matrix /////
OglMatrix2Variable::OglMatrix2Variable()
    : transpose(initData(&transpose,false,"transpose","Transpose the matrix (e.g. to use row-dominant matrices in OpenGL"))
{

}

void OglMatrix2Variable::init()
{
    OglVariable<helper::vector<float> >::init();

    helper::vector<float> temp = value.getValue();

    if (temp.size() %4 != 0)
    {
        serr << "The number of values is not a multiple of 4 ; padding with zero(s)" << sendl;
        while (temp.size() %4 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix2Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0]);
    const bool transp = transpose.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix2(idShader, idstr.c_str(), v.size()/4, transp, vptr);
}

OglMatrix3Variable::OglMatrix3Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix3Variable::init()
{
    OglVariable<helper::vector<float> >::init();

    helper::vector<float> temp = value.getValue();

    if (temp.size() %9 != 0)
    {
        serr << "The number of values is not a multiple of 9 ; padding with zero(s)" << sendl;
        while (temp.size() %9 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix3Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0]);
    const bool transp = transpose.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix3(idShader, idstr.c_str(), v.size()/9, transp, vptr);
}

OglMatrix4Variable::OglMatrix4Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix4Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (temp.size() %16 != 0)
    {
        serr << "The number of values is not a multiple of 16 ; padding with zero(s)" << sendl;
        while (temp.size() %16 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix4Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0]);
    const bool transp = transpose.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix4(idShader, idstr.c_str(), v.size()/16, transp, vptr);
}

OglMatrix2x3Variable::OglMatrix2x3Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix2x3Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (temp.size() %6 != 0)
    {
        serr << "The number of values is not a multiple of 6 ; padding with zero(s)" << sendl;
        while (temp.size() %6 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix2x3Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0]);
    const bool transp = transpose.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix2x3(idShader, idstr.c_str(), v.size()/6, transp, vptr);
}


OglMatrix3x2Variable::OglMatrix3x2Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix3x2Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (temp.size() %6 != 0)
    {
        serr << "The number of values is not a multiple of 6 ; padding with zero(s)" << sendl;
        while (temp.size() %6 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix3x2Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0]);
    const bool transp = transpose.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix3x2(idShader, idstr.c_str(), v.size()/6, transp, vptr);
}

OglMatrix2x4Variable::OglMatrix2x4Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix2x4Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (temp.size() %8 != 0)
    {
        serr << "The number of values is not a multiple of 8 ; padding with zero(s)" << sendl;
        while (temp.size() %8 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix2x4Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0]);
    const bool transp = transpose.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix2x4(idShader, idstr.c_str(), v.size()/8, transp, vptr);
}

OglMatrix4x2Variable::OglMatrix4x2Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix4x2Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (temp.size() %8 != 0)
    {
        serr << "The number of values is not a multiple of 8 ; padding with zero(s)" << sendl;
        while (temp.size() %8 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix4x2Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0]);
    const bool transp = transpose.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix4x2(idShader, idstr.c_str(), v.size()/8, transp, vptr);
}

OglMatrix3x4Variable::OglMatrix3x4Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix3x4Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (temp.size() %12 != 0)
    {
        serr << "The number of values is not a multiple of 12 ; padding with zero(s)" << sendl;
        while (temp.size() %12 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix3x4Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0]);
    const bool transp = transpose.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix3x4(idShader, idstr.c_str(), v.size()/12, transp, vptr);
}

OglMatrix4x3Variable::OglMatrix4x3Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix4x3Variable::init()
{
    OglMatrix2Variable::init();

    helper::vector<float> temp = value.getValue();

    if (temp.size() %12 != 0)
    {
        serr << "The number of values is not a multiple of 12 ; padding with zero(s)" << sendl;
        while (temp.size() %12 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix4x3Variable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? NULL : &(v[0]);
    const bool transp = transpose.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix4x3(idShader, idstr.c_str(), v.size()/12, transp, vptr);
}

OglMatrix4VectorVariable::OglMatrix4VectorVariable()
    : transpose(initData(&transpose,false,"transpose","Transpose the matrix (e.g. to use row-dominant matrices in OpenGL"))
{
}

void OglMatrix4VectorVariable::init()
{
    OglVariable<helper::vector<defaulttype::Mat4x4f> >::init();
}
void OglMatrix4VectorVariable::initVisual()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const helper::vector<defaulttype::Mat4x4f>& v = value.getValue();
    //serr << "OglMatrix4VectorVariable::initVisual(), v = " << v << sendl;
    const float* vptr = v.empty() ? NULL : &(v[0][0][0]);
    bool transp = transpose.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix4(idShader, idstr.c_str(), v.size(), transp, vptr);
}


} // namespace visual

} // namespace component

} // namespace sofa
