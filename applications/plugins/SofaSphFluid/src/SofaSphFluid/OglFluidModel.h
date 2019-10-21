#pragma once

#include <SofaSphFluid/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/helper/gl/FrameBufferObject.h>
#include <sofa/helper/gl/GLSLShader.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseTopology//TopologyData.inl>

namespace sofa
{
namespace component
{
namespace visualmodel
{
using namespace sofa::defaulttype;
/**
 *  \brief Render volume using particles
 *
 */

// http://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf

template<class DataTypes>
class SOFA_SPH_FLUID_API OglFluidModel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglFluidModel, core::visual::VisualModel);
    //typedef ExtVec3fTypes DataTypes;
    typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Real Real;

private:
    topology::PointData< VecCoord > m_positions;
	VecCoord m_previousPositions;

    GLuint m_posVBO;
    helper::gl::FrameBufferObject* m_spriteDepthFBO;
    helper::gl::FrameBufferObject* m_spriteThicknessFBO;
    helper::gl::FrameBufferObject* m_spriteNormalFBO;
    helper::gl::FrameBufferObject* m_spriteBlurDepthHFBO;
    helper::gl::FrameBufferObject* m_spriteBlurDepthVFBO;
	helper::gl::FrameBufferObject* m_spriteBlurThicknessHFBO;
	helper::gl::FrameBufferObject* m_spriteBlurThicknessVFBO;
    helper::gl::FrameBufferObject* m_spriteShadeFBO;

    helper::gl::GLSLShader m_spriteShader;
	helper::gl::GLSLShader m_spriteNormalShader;
	helper::gl::GLSLShader m_spriteBlurDepthShader;
	helper::gl::GLSLShader m_spriteBlurThicknessShader;
	helper::gl::GLSLShader m_spriteShadeShader;

    void drawSprites(const core::visual::VisualParams* vparams);
    void updateVertexBuffer();
protected:
    OglFluidModel();
    virtual ~OglFluidModel();
public:
    Data<unsigned int> d_debugFBO;
    Data<float> d_spriteRadius;
    Data<float> d_spriteThickness;
    Data<unsigned int> d_spriteBlurRadius;
    Data<float> d_spriteBlurScale;
    Data<float> d_spriteBlurDepthFalloff;
    Data<sofa::defaulttype::RGBAColor> d_spriteDiffuseColor;


    void init();
    void initVisual();
    void fwdDraw(core::visual::VisualParams*);
    void bwdDraw(core::visual::VisualParams*);
    void drawVisual(const core::visual::VisualParams* vparams);
    void computeBBox(const core::ExecParams* params, bool onlyVisible = false);

    virtual void updateVisual();

    static std::string templateName(const OglFluidModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

};

}

}

}

