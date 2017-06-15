/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_VISUALMODEL_OGLMODEL_H
#define SOFA_COMPONENT_VISUALMODEL_OGLMODEL_H
#include "config.h"

#include <vector>
#include <string>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <SofaBaseVisual/VisualModelImpl.h>

#define   NB_MAX_TEXTURES 16

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief Main class for rendering 3D model in SOFA.
 *
 *  This class implements VisuelModelImpl with rendering functions
 *  using OpenGL.
 *
 */

class SOFA_OPENGL_VISUAL_API OglModel : public VisualModelImpl
{
public:
    SOFA_CLASS(OglModel, VisualModelImpl);

    Data<bool> blendTransparency;
protected:
    Data<bool> premultipliedAlpha, useVBO, writeZTransparent, alphaBlend, depthTest;
    Data<int> cullFace;
    Data<GLfloat> lineWidth;
    Data<GLfloat> pointSize;
    Data<bool> lineSmooth;
    Data<bool> pointSmooth;
    /// Suppress field for save as function
    Data < bool > isToPrint;

    // primitive types
    Data<sofa::helper::OptionsGroup> primitiveType;

    //alpha blend function
    Data<sofa::helper::OptionsGroup> blendEquation;
    Data<sofa::helper::OptionsGroup> sourceFactor;
    Data<sofa::helper::OptionsGroup> destFactor;
    GLenum blendEq, sfactor, dfactor;

    helper::gl::Texture *tex; //this texture is used only if a texture name is specified in the scn
    GLuint vbo, iboEdges, iboTriangles, iboQuads;
    bool canUseVBO, VBOGenDone, initDone, useEdges, useTriangles, useQuads, canUsePatches;
    unsigned int oldVerticesSize, oldNormalsSize, oldTexCoordsSize, oldTangentsSize, oldBitangentsSize, oldEdgesSize, oldTrianglesSize, oldQuadsSize;
    void internalDraw(const core::visual::VisualParams* vparams, bool transparent);

    void drawGroup(int ig, bool transparent);
    void drawGroups(bool transparent);

    virtual void pushTransformMatrix(float* matrix) { glPushMatrix(); glMultMatrixf(matrix); }
    virtual void popTransformMatrix() { glPopMatrix(); }

    std::vector<helper::gl::Texture*> textures;

    std::map<int, int> materialTextureIdMap; //link between a material and a texture

    GLenum getGLenum(const char* c ) const;


    OglModel();

    ~OglModel();
public:

    bool loadTexture(const std::string& filename);
    bool loadTextures() ;

    void initTextures();
    virtual void initVisual();

    virtual void init() { VisualModelImpl::init(); }

    virtual void updateBuffers();

    bool hasTransparent();
    bool hasTexture();

public:
    bool isUseEdges()	{ return useEdges; }
    bool isUseTriangles()	{ return useTriangles; }
    bool isUseQuads()	{ return useQuads; }
    bool isUseVbo()	{ return useVBO.getValue(); }

    helper::gl::Texture* getTex() const	{ return tex; }
    GLuint getVbo()	{ return vbo;	}
    GLuint getIboEdges() { return iboEdges; }
    GLuint getIboTriangles() { return iboTriangles; }
    GLuint getIboQuads()    { return iboQuads; }
    const std::vector<helper::gl::Texture*>& getTextures() const { return textures;	}

#ifdef SOFA_HAVE_GLEW
    void createVertexBuffer();
    void createEdgesIndicesBuffer();
    void createTrianglesIndicesBuffer();
    void createQuadsIndicesBuffer();
    void initVertexBuffer();
    void initEdgesIndicesBuffer();
    void initTrianglesIndicesBuffer();
    void initQuadsIndicesBuffer();
    void updateVertexBuffer();
    void updateEdgesIndicesBuffer();
    void updateTrianglesIndicesBuffer();
    void updateQuadsIndicesBuffer();
#endif
};

typedef sofa::defaulttype::Vec<3,GLfloat> GLVec3f;
typedef sofa::defaulttype::ExtVectorTypes<GLVec3f,GLVec3f> GLExtVec3fTypes;

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
