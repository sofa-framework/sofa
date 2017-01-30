#include <sofa/helper/Utils.h>
#include <gtest/gtest.h>

#include <sofa/helper/gl/GLSLShader.h>
#include <sofa/helper/system/FileRepository.h>
#include "test.glsl"

using sofa::helper::Utils;


struct GLSLShader_test : public ::testing::Test
{
    GLSLShader_test() {
    }

    void SetUp()
    {
        sofa::helper::system::DataRepository.addFirstPath(FRAMEWORK_TEST_RESOURCES_DIR);
    }
    void TearDown()
    {
        sofa::helper::system::DataRepository.removePath(FRAMEWORK_TEST_RESOURCES_DIR);
    }
    
};

TEST(GLSLShader_test, GLSLShader_SetFiles)
{
    sofa::helper::gl::GLSLShader glshader;
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
    sofa::helper::gl::GLSLShader glshader;
    std::string vs = sofa::helper::gl::shader::testvs;
    std::string fs = sofa::helper::gl::shader::testfs;
    std::string gs = sofa::helper::gl::shader::testgs;
    std::string tcs = sofa::helper::gl::shader::testtcs;
    std::string tes = sofa::helper::gl::shader::testtes;
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