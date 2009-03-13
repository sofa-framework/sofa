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
#include <sofa/component/visualmodel/OglVariable.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

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


}

}

}
