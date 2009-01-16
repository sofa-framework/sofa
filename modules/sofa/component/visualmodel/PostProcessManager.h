/*
 * PostProcessManager.h
 *
 *  Created on: 12 janv. 2009
 *      Author: froy
 */

#ifndef SOFA_COMPONENT_POSTPROCESSMANAGER_H_
#define SOFA_COMPONENT_POSTPROCESSMANAGER_H_

#include <sofa/core/VisualManager.h>
#include <sofa/helper/gl/FrameBufferObject.h>
#include <sofa/component/visualmodel/OglShader.h>
namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_COMPONENT_VISUALMODEL_API PostProcessManager : public core::VisualManager
{
private:
    static const std::string DEPTH_OF_FIELD_VERTEX_SHADER;
    static const std::string DEPTH_OF_FIELD_FRAGMENT_SHADER;
    helper::gl::FrameBufferObject fbo;
    OglShader* dofShader;
    bool postProcessEnabled;

public:
    ///Files where vertex shader is defined
    Data<std::string> vertFilename;
    ///Files where fragment shader is defined
    Data<std::string> fragFilename;

    PostProcessManager();
    virtual ~PostProcessManager();

    void init() ;
    void reinit() { };
    void initVisual();
    void update() { };

    void preDrawScene(helper::gl::VisualParameters* vp);
    bool drawScene(helper::gl::VisualParameters* vp);
    void postDrawScene(helper::gl::VisualParameters* vp);

    void handleEvent(sofa::core::objectmodel::Event* event);
};

} //visualmodel

} //component

} //sofa

#endif /* SOFA_COMPONENT_POSTPROCESSMANAGER_H_ */
