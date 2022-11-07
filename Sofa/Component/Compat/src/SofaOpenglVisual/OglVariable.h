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
#pragma once

#include <sofa/config.h>

#if __has_include(<sofa/gl/component/shader/OglVariable.h>)
#include <sofa/gl/component/shader/OglVariable.h>
#define SOFAGL_COMPONENT_OGLVARIABLE

SOFA_DEPRECATED_HEADER("v22.06", "v23.06", "sofa/gl/component/shader/OglVariable.h")

#else
#error "SofaOpenglVisual contents has been moved to Sofa.GL.Component. Include <sofa/gl/component/shader/OglVariable.h> instead of this one."
#endif

#ifdef SOFAGL_COMPONENT_OGLVARIABLE

namespace sofa::component::visualmodel
{
    template<class DataTypes>
    using OglVariable = sofa::gl::component::shader::OglVariable<DataTypes>;

    using OglIntVariable = sofa::gl::component::shader::OglIntVariable;
    using OglInt2Variable = sofa::gl::component::shader::OglInt2Variable;
    using OglInt3Variable = sofa::gl::component::shader::OglInt3Variable;
    using OglInt4Variable = sofa::gl::component::shader::OglInt4Variable;
    using OglFloatVariable = sofa::gl::component::shader::OglFloatVariable;
    using OglFloat2Variable = sofa::gl::component::shader::OglFloat2Variable;
    using OglFloat3Variable = sofa::gl::component::shader::OglFloat3Variable;
    using OglFloat4Variable = sofa::gl::component::shader::OglFloat4Variable;
    using OglIntVectorVariable = sofa::gl::component::shader::OglIntVectorVariable;
    using OglIntVector2Variable = sofa::gl::component::shader::OglIntVector2Variable;
    using OglIntVector3Variable = sofa::gl::component::shader::OglIntVector3Variable;
    using OglIntVector4Variable = sofa::gl::component::shader::OglIntVector4Variable;
    using OglFloatVectorVariable = sofa::gl::component::shader::OglFloatVectorVariable;
    using OglFloatVector2Variable = sofa::gl::component::shader::OglFloatVector2Variable;
    using OglFloatVector3Variable = sofa::gl::component::shader::OglFloatVector3Variable;
    using OglFloatVector4Variable = sofa::gl::component::shader::OglFloatVector4Variable;
    using OglMatrix2Variable = sofa::gl::component::shader::OglMatrix2Variable;
    using OglMatrix3Variable = sofa::gl::component::shader::OglMatrix3Variable;
    using OglMatrix4Variable = sofa::gl::component::shader::OglMatrix4Variable;
    using OglMatrix2x3Variable = sofa::gl::component::shader::OglMatrix2x3Variable;
    using OglMatrix3x2Variable = sofa::gl::component::shader::OglMatrix3x2Variable;
    using OglMatrix2x4Variable = sofa::gl::component::shader::OglMatrix2x4Variable;
    using OglMatrix4x2Variable = sofa::gl::component::shader::OglMatrix4x2Variable;
    using OglMatrix3x4Variable = sofa::gl::component::shader::OglMatrix3x4Variable;
    using OglMatrix4x3Variable = sofa::gl::component::shader::OglMatrix4x3Variable;
    using OglMatrix4VectorVariable = sofa::gl::component::shader::OglMatrix4VectorVariable;

} // namespace sofa::component::visualmodel

#endif // SOFAGL_COMPONENT_OGLVARIABLE
