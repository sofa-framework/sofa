/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/testing/config.h>

#include <sofa/helper/Utils.h>
#include <gtest/gtest.h>

#include <sofa/gl/GLSLShader.h>
#include <sofa/helper/system/FileRepository.h>
#include "test.cppglsl"

using namespace sofa;
using sofa::helper::Utils;

/// NOTE: We can only test non-OPenGL operations as the context is not created

struct GLSLShader_test : public ::testing::Test
{
    GLSLShader_test() {
    }

    void SetUp() override
    {
        sofa::helper::system::DataRepository.addFirstPath(SOFA_TESTING_RESOURCES_DIR);
    }
    void TearDown() override
    {
        sofa::helper::system::DataRepository.removePath(SOFA_TESTING_RESOURCES_DIR);
    }

};

TEST(GLSLShader_test, GLSLShader_SetFiles)
{
    gl::GLSLShader glshader;
    std::string vs("shader/test.vs");
    std::string fs("shader/test.fs");
    std::string gs("shader/test.gs");
    std::string tcs("shader/test.tcs");
    std::string tes("shader/test.tes");
    glshader.SetVertexShaderFileName(vs);
    glshader.SetFragmentShaderFileName(fs);
#ifdef GL_GEOMETRY_SHADER_EXT
    glshader.SetGeometryShaderFileName(gs);
#endif
#ifdef GL_TESS_CONTROL_SHADER
    glshader.SetTessellationControlShaderFileName(tcs);
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    glshader.SetTessellationEvaluationShaderFileName(tes);
#endif

    EXPECT_TRUE(glshader.IsSet(GL_VERTEX_SHADER_ARB));
    EXPECT_TRUE(glshader.GetShaderFileName(GL_VERTEX_SHADER_ARB).compare(vs) == 0);

    EXPECT_TRUE(glshader.IsSet(GL_FRAGMENT_SHADER_ARB));
    EXPECT_TRUE(glshader.GetShaderFileName(GL_FRAGMENT_SHADER_ARB).compare(fs) == 0);

#ifdef GL_GEOMETRY_SHADER_EXT
    EXPECT_TRUE(glshader.IsSet(GL_GEOMETRY_SHADER_ARB));
    EXPECT_TRUE(glshader.GetShaderFileName(GL_GEOMETRY_SHADER_ARB).compare(gs) == 0);
#endif
#ifdef GL_TESS_CONTROL_SHADER
    EXPECT_TRUE(glshader.IsSet(GL_TESS_CONTROL_SHADER));
    EXPECT_TRUE(glshader.GetShaderFileName(GL_TESS_CONTROL_SHADER).compare(tcs) == 0);
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    EXPECT_TRUE(glshader.IsSet(GL_TESS_EVALUATION_SHADER));
    EXPECT_TRUE(glshader.GetShaderFileName(GL_TESS_EVALUATION_SHADER).compare(tes) == 0);
#endif
}

TEST(GLSLShader_test, GLSLShader_SetStrings)
{
    gl::GLSLShader glshader;
    std::string vs = sofa::gl::shader::testvs;
    std::string fs = sofa::gl::shader::testfs;
    std::string gs = sofa::gl::shader::testgs;
    std::string tcs = sofa::gl::shader::testtcs;
    std::string tes = sofa::gl::shader::testtes;
    glshader.SetVertexShaderFromString(vs);
    glshader.SetFragmentShaderFromString(fs);
#ifdef GL_GEOMETRY_SHADER_EXT
    glshader.SetGeometryShaderFromString(gs);
#endif
#ifdef GL_TESS_CONTROL_SHADER
    glshader.SetTessellationControlShaderFromString(tcs);
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    glshader.SetTessellationEvaluationShaderFromString(tes);
#endif

    EXPECT_TRUE(glshader.IsSet(GL_VERTEX_SHADER_ARB));
    EXPECT_TRUE(glshader.IsSet(GL_FRAGMENT_SHADER_ARB));
#ifdef GL_GEOMETRY_SHADER_EXT
    EXPECT_TRUE(glshader.IsSet(GL_GEOMETRY_SHADER_ARB));
#endif
#ifdef GL_TESS_CONTROL_SHADER
    EXPECT_TRUE(glshader.IsSet(GL_TESS_CONTROL_SHADER));
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    EXPECT_TRUE(glshader.IsSet(GL_TESS_EVALUATION_SHADER));
#endif
}

TEST(GLSLShader_test, GLSLShader_AddHeader)
{
    gl::GLSLShader glshader;

    const std::string header = "#HEADER#";
    const std::string expectedHeader = "#HEADER#\n";

    glshader.AddHeader(header);
    EXPECT_TRUE(glshader.GetHeader().compare(expectedHeader) == 0);
}

TEST(GLSLShader_test, GLSLShader_SetStage)
{
    gl::GLSLShader glshader;
    EXPECT_TRUE(glshader.GetShaderStageName(GL_VERTEX_SHADER_ARB).compare("Vertex") == 0);
    EXPECT_TRUE(glshader.GetShaderStageName(GL_FRAGMENT_SHADER_ARB).compare("Fragment") == 0);
    EXPECT_TRUE(glshader.GetShaderStageName(GL_GEOMETRY_SHADER_EXT).compare("Geometry") == 0);
    EXPECT_TRUE(glshader.GetShaderStageName(GL_TESS_CONTROL_SHADER).compare("TessellationControl") == 0);
    EXPECT_TRUE(glshader.GetShaderStageName(GL_TESS_EVALUATION_SHADER).compare("TessellationEvaluation") == 0);
    EXPECT_TRUE(glshader.GetShaderStageName(123456789).compare("Unknown") == 0);
}

TEST(GLSLShader_test, GLSLShader_AddDefineMacro)
{
    gl::GLSLShader glshader;

    const std::string define1 = "NUMBER_OF_THINGS";
    const std::string value1 = "5";
    const std::string define2 = "NUMBER_OF_STUFF";
    const std::string value2 = "42";
    const std::string expectedHeader1 = "#define NUMBER_OF_THINGS 5\n";
    const std::string expectedHeader2 = "#define NUMBER_OF_THINGS 5\n#define NUMBER_OF_STUFF 42\n";

    glshader.AddDefineMacro(define1, value1);
    EXPECT_TRUE(glshader.GetHeader().compare(expectedHeader1) == 0);
    glshader.AddDefineMacro(define2, value2);
    EXPECT_TRUE(glshader.GetHeader().compare(expectedHeader2) == 0);
}
