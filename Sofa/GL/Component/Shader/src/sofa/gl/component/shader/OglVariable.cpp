/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/gl/component/shader/OglVariable.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gl::component::shader
{

void registerOglVariable(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("OglIntVariable").add< OglIntVariable >());
    factory->registerObjects(core::ObjectRegistrationData("OglInt2Variable").add< OglInt2Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglInt3Variable").add< OglInt3Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglInt4Variable").add< OglInt4Variable >());

    factory->registerObjects(core::ObjectRegistrationData("OglFloatVariable").add< OglFloatVariable >());
    factory->registerObjects(core::ObjectRegistrationData("OglFloat2Variable").add< OglFloat2Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglFloat3Variable").add< OglFloat3Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglFloat4Variable").add< OglFloat4Variable >());

    factory->registerObjects(core::ObjectRegistrationData("OglIntVectorVariable").add< OglIntVectorVariable >());
    factory->registerObjects(core::ObjectRegistrationData("OglIntVector2Variable").add< OglIntVector2Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglIntVector3Variable").add< OglIntVector3Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglIntVector4Variable").add< OglIntVector4Variable >());


    factory->registerObjects(core::ObjectRegistrationData("OglFloatVectorVariable").add< OglFloatVectorVariable >());
    factory->registerObjects(core::ObjectRegistrationData("OglFloatVector2Variable").add< OglFloatVector2Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglFloatVector3Variable").add< OglFloatVector3Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglFloatVector4Variable").add< OglFloatVector4Variable >());


    factory->registerObjects(core::ObjectRegistrationData("OglMatrix2Variable").add< OglMatrix2Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglMatrix3Variable").add< OglMatrix3Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglMatrix4Variable").add< OglMatrix4Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglMatrix2x3Variable").add< OglMatrix2x3Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglMatrix3x2Variable").add< OglMatrix3x2Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglMatrix2x4Variable").add< OglMatrix2x4Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglMatrix4x2Variable").add< OglMatrix4x2Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglMatrix3x4Variable").add< OglMatrix3x4Variable >());
    factory->registerObjects(core::ObjectRegistrationData("OglMatrix4x3Variable").add< OglMatrix4x3Variable >());

    factory->registerObjects(core::ObjectRegistrationData("OglMatrix4VectorVariable").add< OglMatrix4VectorVariable >());

}

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

void OglIntVariable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const int v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setInt(idShader, idstr.c_str(), v);
}


void OglInt2Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::Vec<2,int>& v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setInt2(idShader, idstr.c_str(), v[0], v[1]);
}

void OglInt3Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::Vec<3,int>& v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setInt3(idShader, idstr.c_str(), v[0], v[1], v[2]);
}

void OglInt4Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::Vec<4,int>& v = value.getValue();
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

void OglFloatVariable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const float v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloat(idShader, idstr.c_str(), v);
}


void OglFloat2Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::Vec2f& v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloat2(idShader, idstr.c_str(), v[0], v[1]);
}

void OglFloat3Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::Vec3f& v = value.getValue();
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloat3(idShader, idstr.c_str(), v[0], v[1], v[2]);
}

void OglFloat4Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::Vec4f& v = value.getValue();
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
    OglVariable<type::vector<GLint> >::init();
}

void OglIntVector2Variable::init()
{
    OglIntVectorVariable::init();
    type::vector<GLint> temp = value.getValue();
    if (temp.size() %2 != 0)
    {
        msg_error() << "The number of values is not even ; padding with one zero";
        temp.push_back(0);
        value.setValue(temp);

    }
}

void OglIntVector3Variable::init()
{
    OglIntVectorVariable::init();
    type::vector<GLint> temp = value.getValue();

    if (temp.size() %3 != 0)
    {
        msg_error() << "The number of values is not a multiple of 3 ; padding with zero(s)";
        while (temp.size() %3 != 0)
            temp.push_back(0);
        value.setValue(temp);
    }
}

void OglIntVector4Variable::init()
{
    OglIntVectorVariable::init();
    type::vector<GLint> temp = value.getValue();

    if (temp.size() %4 != 0)
    {
        msg_error() << "The number of values is not a multiple of 4 ; padding with zero(s)";
        while (temp.size() %4 != 0)
            temp.push_back(0);
        value.setValue(temp);
    }
}

void OglIntVectorVariable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<GLint>& v = value.getValue();
    const GLint* vptr = v.empty() ? nullptr : &(v[0]);
    const int count = int(v.size());
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setIntVector(idShader, idstr.c_str(), count, vptr);
}

void OglIntVector2Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<GLint>& v = value.getValue();
    const GLint* vptr = v.empty() ? nullptr : &(v[0]);
    const int count = int(v.size()/2);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setIntVector2(idShader, idstr.c_str(), count, vptr);
}

void OglIntVector3Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<GLint>& v = value.getValue();
    const GLint* vptr = v.empty() ? nullptr : &(v[0]);
    const int count = int(v.size()/3);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setIntVector3(idShader, idstr.c_str(), count, vptr);
}

void OglIntVector4Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<GLint>& v = value.getValue();
    const GLint* vptr = v.empty() ? nullptr : &(v[0]);
    const int count = int(v.size()/4);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setIntVector4(idShader, idstr.c_str(), count, vptr);
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
    OglVariable<type::vector<float> >::init();
}

void OglFloatVector2Variable::init()
{
    OglVariable<type::vector<type::Vec2f> >::init();
}

void OglFloatVector3Variable::init()
{
    OglVariable<type::vector<type::Vec3f> >::init();
}

void OglFloatVector4Variable::init()
{
    OglVariable<type::vector<type::Vec4f> >::init();
}

void OglFloatVectorVariable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0]);
    const int count = int(v.size());
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloatVector(idShader, idstr.c_str(), count, vptr);
}

void OglFloatVector2Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<type::Vec2f>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0][0]);
    const int count = int(v.size());
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloatVector2(idShader, idstr.c_str(), count, vptr);
}

void OglFloatVector3Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<type::Vec3f>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0][0]);
    const int count = int(v.size());
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloatVector3(idShader, idstr.c_str(), count, vptr);
}

void OglFloatVector4Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<type::Vec4f>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0][0]);
    const int count = int(v.size());
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setFloatVector4(idShader, idstr.c_str(), count, vptr);
}


//// Matrix /////
OglMatrix2Variable::OglMatrix2Variable()
    : transpose(initData(&transpose,false,"transpose","Transpose the matrix (e.g. to use row-dominant matrices in OpenGL"))
{

}

void OglMatrix2Variable::init()
{
    OglVariable<type::vector<float> >::init();

    type::vector<float> temp = value.getValue();

    if (temp.size() %4 != 0)
    {
        msg_error() << "The number of values is not a multiple of 4 ; padding with zero(s)";
        while (temp.size() %4 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix2Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0]);
    const bool transp = transpose.getValue();
    const int count = int(v.size()/4);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix2(idShader, idstr.c_str(), count, transp, vptr);
}

OglMatrix3Variable::OglMatrix3Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix3Variable::init()
{
    OglVariable<type::vector<float> >::init();

    type::vector<float> temp = value.getValue();

    if (temp.size() %9 != 0)
    {
        msg_error() << "The number of values is not a multiple of 9 ; padding with zero(s)";
        while (temp.size() %9 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix3Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0]);
    const bool transp = transpose.getValue();
    const int count = int(v.size() / 9);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix3(idShader, idstr.c_str(), count, transp, vptr);
}

OglMatrix4Variable::OglMatrix4Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix4Variable::init()
{
    OglMatrix2Variable::init();

    type::vector<float> temp = value.getValue();

    if (temp.size() %16 != 0)
    {
        msg_error() << "The number of values is not a multiple of 16 ; padding with zero(s)";
        while (temp.size() %16 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix4Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0]);
    const bool transp = transpose.getValue();
    const int count = int(v.size() / 16);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix4(idShader, idstr.c_str(), count, transp, vptr);
}

OglMatrix2x3Variable::OglMatrix2x3Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix2x3Variable::init()
{
    OglMatrix2Variable::init();

    type::vector<float> temp = value.getValue();

    if (temp.size() %6 != 0)
    {
        msg_error() << "The number of values is not a multiple of 6 ; padding with zero(s)";
        while (temp.size() %6 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix2x3Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0]);
    const bool transp = transpose.getValue();
    const int count = int(v.size() / 6);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix2x3(idShader, idstr.c_str(), count, transp, vptr);
}


OglMatrix3x2Variable::OglMatrix3x2Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix3x2Variable::init()
{
    OglMatrix2Variable::init();

    type::vector<float> temp = value.getValue();

    if (temp.size() %6 != 0)
    {
        msg_error() << "The number of values is not a multiple of 6 ; padding with zero(s)";
        while (temp.size() %6 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix3x2Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0]);
    const bool transp = transpose.getValue();
    const int count = int(v.size() / 6);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix3x2(idShader, idstr.c_str(), count, transp, vptr);
}

OglMatrix2x4Variable::OglMatrix2x4Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix2x4Variable::init()
{
    OglMatrix2Variable::init();

    type::vector<float> temp = value.getValue();

    if (temp.size() %8 != 0)
    {
        msg_error() << "The number of values is not a multiple of 8 ; padding with zero(s)";
        while (temp.size() %8 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix2x4Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0]);
    const bool transp = transpose.getValue();
    const int count = int(v.size() / 8);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix2x4(idShader, idstr.c_str(), count, transp, vptr);
}

OglMatrix4x2Variable::OglMatrix4x2Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix4x2Variable::init()
{
    OglMatrix2Variable::init();

    type::vector<float> temp = value.getValue();

    if (temp.size() %8 != 0)
    {
        msg_error() << "The number of values is not a multiple of 8 ; padding with zero(s)";
        while (temp.size() %8 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix4x2Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0]);
    const bool transp = transpose.getValue();
    const int count = int(v.size() / 8);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix4x2(idShader, idstr.c_str(), count, transp, vptr);
}

OglMatrix3x4Variable::OglMatrix3x4Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix3x4Variable::init()
{
    OglMatrix2Variable::init();

    type::vector<float> temp = value.getValue();

    if (temp.size() %12 != 0)
    {
        msg_error() << "The number of values is not a multiple of 12 ; padding with zero(s)";
        while (temp.size() %12 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix3x4Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0]);
    const bool transp = transpose.getValue();
    const int count = int(v.size() / 12);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix3x4(idShader, idstr.c_str(), count, transp, vptr);
}

OglMatrix4x3Variable::OglMatrix4x3Variable()
    : OglMatrix2Variable()
{

}

void OglMatrix4x3Variable::init()
{
    OglMatrix2Variable::init();

    type::vector<float> temp = value.getValue();

    if (temp.size() %12 != 0)
    {
        msg_error() << "The number of values is not a multiple of 12 ; padding with zero(s)";
        while (temp.size() %12 != 0)
            temp.push_back(0.0);
        value.setValue(temp);
    }
}

void OglMatrix4x3Variable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<float>& v = value.getValue();
    const float* vptr = v.empty() ? nullptr : &(v[0]);
    const bool transp = transpose.getValue();
    const int count = int(v.size() / 12);
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix4x3(idShader, idstr.c_str(), count, transp, vptr);
}

OglMatrix4VectorVariable::OglMatrix4VectorVariable()
    : transpose(initData(&transpose,false,"transpose","Transpose the matrix (e.g. to use row-dominant matrices in OpenGL"))
{
}

void OglMatrix4VectorVariable::init()
{
    OglVariable<type::vector<type::Mat4x4f> >::init();
}
void OglMatrix4VectorVariable::pushValue()
{
    const unsigned int idShader = indexShader.getValue();
    const std::string& idstr = id.getValue();
    const type::vector<type::Mat4x4f>& v = value.getValue();

    const float* vptr = v.empty() ? nullptr : v[0].data();
    const bool transp = transpose.getValue();
    const int count = int(v.size());
    for(std::set<OglShader*>::iterator it = shaders.begin(), iend = shaders.end(); it!=iend; ++it)
        (*it)->setMatrix4(idShader, idstr.c_str(), count, transp, vptr);
}


} // namespace sofa::gl::component::shader
