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
#include <SofaOpenglVisual/OglModel.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/RAII.h>
#include <sofa/helper/vector.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <string.h>
#include <sofa/helper/types/RGBAColor.h>
//#ifdef SOFA_HAVE_GLEW
//#include <sofa/helper/gl/GLSLShader.h>
//#endif // SOFA_HAVE_GLEW

//#define NO_VBO
//#define DEBUG_DRAW
namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;

SOFA_DECL_CLASS(OglModel)

int OglModelClass = core::RegisterObject("Generic visual model for OpenGL display")
        .add< OglModel >()
        ;

template<class T>
const T* getData(const defaulttype::ResizableExtVector<T>& v) { return v.getData(); }

template<class T>
const T* getData(const sofa::helper::vector<T>& v) { return &v[0]; }


OglModel::OglModel()
    : blendTransparency(initData(&blendTransparency, (bool) true, "blendTranslucency", "Blend transparent parts"))
    , premultipliedAlpha(initData(&premultipliedAlpha, (bool) false, "premultipliedAlpha", "is alpha premultiplied ?"))
#ifndef SOFA_HAVE_GLEW
    , useVBO(initData(&useVBO, (bool) false, "useVBO", "Use VBO for rendering"))
#else
    , useVBO(initData(&useVBO, (bool) true, "useVBO", "Use VBO for rendering"))
#endif
    , writeZTransparent(initData(&writeZTransparent, (bool) false, "writeZTransparent", "Write into Z Buffer for Transparent Object"))
    , alphaBlend(initData(&alphaBlend, (bool) false, "alphaBlend", "Enable alpha blending"))
    , depthTest(initData(&depthTest, (bool) true, "depthTest", "Enable depth testing"))
    , cullFace(initData(&cullFace, (int) 0, "cullFace", "Face culling (0 = no culling, 1 = cull back faces, 2 = cull front faces)"))
    , lineWidth(initData(&lineWidth, (GLfloat) 1, "lineWidth", "Line width (set if != 1, only for lines rendering)"))
    , pointSize(initData(&pointSize, (GLfloat) 1, "pointSize", "Point size (set if != 1, only for points rendering)"))
    , lineSmooth(initData(&lineSmooth, (bool) false, "lineSmooth", "Enable smooth line rendering"))
    , pointSmooth(initData(&pointSmooth, (bool) false, "pointSmooth", "Enable smooth point rendering"))
    , isToPrint( initData(&isToPrint, false, "isToPrint", "suppress somes data before using save as function"))
    , primitiveType( initData(&primitiveType, "primitiveType", "Select types of primitives to send (necessary for some shader types such as geometry or tesselation)"))
    , blendEquation( initData(&blendEquation, "blendEquation", "if alpha blending is enabled this specifies how source and destination colors are combined") )
    , sourceFactor( initData(&sourceFactor, "sfactor", "if alpha blending is enabled this specifies how the red, green, blue, and alpha source blending factors are computed") )
    , destFactor( initData(&destFactor, "dfactor", "if alpha blending is enabled this specifies how the red, green, blue, and alpha destination blending factors are computed") )
    , tex(NULL)
    , vbo(0), iboEdges(0), iboTriangles(0), iboQuads(0)
    , canUseVBO(false), VBOGenDone(false), initDone(false), useEdges(false), useTriangles(false), useQuads(false), canUsePatches(false)
    , oldVerticesSize(0), oldNormalsSize(0), oldTexCoordsSize(0), oldTangentsSize(0), oldBitangentsSize(0), oldEdgesSize(0), oldTrianglesSize(0), oldQuadsSize(0)
{

    textures.clear();

    sofa::helper::OptionsGroup* blendEquationOptions = blendEquation.beginEdit();
    blendEquationOptions->setNames(4,"GL_FUNC_ADD", "GL_FUNC_SUBTRACT", "GL_MIN", "GL_MAX"); // .. add other options
    blendEquationOptions->setSelectedItem(0);
    blendEquation.endEdit();

    // alpha blend values
    sofa::helper::OptionsGroup* sourceFactorOptions = sourceFactor.beginEdit();
    sourceFactorOptions->setNames(4,"GL_ZERO", "GL_ONE", "GL_SRC_ALPHA", "GL_ONE_MINUS_SRC_ALPHA"); // .. add other options
    sourceFactorOptions->setSelectedItem(2);
    sourceFactor.endEdit();

    sofa::helper::OptionsGroup* destFactorOptions = destFactor.beginEdit();
    destFactorOptions->setNames(4,"GL_ZERO", "GL_ONE", "GL_SRC_ALPHA", "GL_ONE_MINUS_SRC_ALPHA"); // .. add other options
    destFactorOptions->setSelectedItem(3);
    destFactor.endEdit();

    sofa::helper::OptionsGroup* primitiveTypeOptions = primitiveType.beginEdit();
    primitiveTypeOptions->setNames(4, "DEFAULT", "LINES_ADJACENCY", "PATCHES", "POINTS");
    primitiveTypeOptions->setSelectedItem(0);
    primitiveType.endEdit();
}

OglModel::~OglModel()
{
    if (tex!=NULL) delete tex;

    for (unsigned int i = 0 ; i < textures.size() ; i++)
    {
        delete textures[i];
    }

#ifdef GL_ARB_vertex_buffer_object
    // NB fjourdes : I don t know why gDEBugger still reports
    // graphics memory leaks after destroying the GLContext
    // even if the vbos destruction is claimed with the following
    // lines...
    if( vbo > 0 )
    {
        glDeleteBuffersARB(1,&vbo);
    }
    if( iboEdges > 0)
    {
        glDeleteBuffersARB(1,&iboEdges);
    }
    if( iboTriangles > 0)
    {
        glDeleteBuffersARB(1,&iboTriangles);
    }
    if( iboQuads > 0 )
    {
        glDeleteBuffersARB(1,&iboQuads);
    }
#endif

}

void OglModel::drawGroup(int ig, bool transparent)
{
    glEnable(GL_NORMALIZE);

    const ResizableExtVector<Edge>& edges = this->getEdges();
    const ResizableExtVector<Triangle>& triangles = this->getTriangles();
    const ResizableExtVector<Quad>& quads = this->getQuads();
    const VecCoord& vertices = this->getVertices();
    const ResizableExtVector<Deriv>& vnormals = this->getVnormals();

    FaceGroup g;
    if (ig < 0)
    {
        g.materialId = -1;
        g.edge0 = 0;
        g.nbe = edges.size();
        g.tri0 = 0;
        g.nbt = triangles.size();
        g.quad0 = 0;
        g.nbq = quads.size();
    }
    else
    {
        g = this->groups.getValue()[ig];
    }
    Material m;
    if (g.materialId < 0)
        m = this->material.getValue();
    else
        m = this->materials.getValue()[g.materialId];

    bool isTransparent = (m.useDiffuse && m.diffuse[3] < 1.0) || hasTransparent();
    if (transparent ^ isTransparent) return;


    if (!tex && m.useTexture && m.activated)
    {
        //get the texture id corresponding to the current material
        int indexInTextureArray = materialTextureIdMap[g.materialId];
        if (textures[indexInTextureArray])
        {
            textures[indexInTextureArray]->bind();
        }

        glEnable(GL_TEXTURE_2D);
#ifdef SOFA_HAVE_GLEW
        if(VBOGenDone && useVBO.getValue())
        {
            glBindBufferARB(GL_ARRAY_BUFFER, vbo);
            glTexCoordPointer(2, GL_FLOAT, 0, (char*)NULL + (vertices.size()*sizeof(vertices[0]))
                    + (vnormals.size()*sizeof(vnormals[0]))
                             );
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
        }
        else
#endif // SOFA_HAVE_GLEW
        {
            //get the texture coordinates
            const VecTexCoord& vtexcoords = this->getVtexcoords();
            glTexCoordPointer(2, GL_FLOAT, 0, getData(vtexcoords));
        }
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
//
//        if (hasTangents)
//        {
//            glClientActiveTexture(GL_TEXTURE1);
//            glEnableClientState(GL_TEXTURE_COORD_ARRAY);
//            if(VBOGenDone && useVBO.getValue())
//            {
//                glBindBufferARB(GL_ARRAY_BUFFER, vbo);
//                glTexCoordPointer(3, GL_FLOAT, 0,
//                                  (char*)NULL + (vertices.size()*sizeof(vertices[0])) +
//                                  (vnormals.size()*sizeof(vnormals[0])) +
//                                  (vtexcoords.size()*sizeof(vtexcoords[0])));
//                glBindBufferARB(GL_ARRAY_BUFFER, 0);
//            }
//            else
//                glTexCoordPointer(3, GL_FLOAT, 0, vtangents.getData());
//
//            glClientActiveTexture(GL_TEXTURE2);
//            glEnableClientState(GL_TEXTURE_COORD_ARRAY);
//            if(VBOGenDone && useVBO.getValue())
//            {
//                glBindBufferARB(GL_ARRAY_BUFFER, vbo);
//                glTexCoordPointer(3, GL_FLOAT, 0,
//                                  (char*)NULL + (vertices.size()*sizeof(vertices[0])) +
//                                  (vnormals.size()*sizeof(vnormals[0])) +
//                                  (vtexcoords.size()*sizeof(vtexcoords[0])) +
//                                  (vtangents.size()*sizeof(vtangents[0])));
//                glBindBufferARB(GL_ARRAY_BUFFER, 0);
//            }
//            else
//                glTexCoordPointer(3, GL_FLOAT, 0, vbitangents.getData());
//
//            glClientActiveTexture(GL_TEXTURE0);
//        }
    }

    RGBAColor ambient = m.useAmbient?m.ambient:RGBAColor::black();
    RGBAColor diffuse = m.useDiffuse?m.diffuse:RGBAColor::black();
    RGBAColor specular = m.useSpecular?m.specular:RGBAColor::black();
    RGBAColor emissive = m.useEmissive?m.emissive:RGBAColor::black();
    float shininess = m.useShininess?m.shininess:45;
    if( shininess > 128.0f ) shininess = 128.0f;

    if (shininess == 0.0f)
    {
        specular = RGBAColor::black() ;
        shininess = 1;
    }

    if (isTransparent)
    {
        emissive[3] = 0; //diffuse[3];
        ambient[3] = 0; //diffuse[3];
        //diffuse[3] = 0;
        specular[3] = 0;
    }
    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT, ambient.data());
    glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse.data());
    glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular.data());
    glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive.data());
    glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, shininess);
    const bool useBufferObjects = (VBOGenDone && useVBO.getValue());
    const bool drawPoints = (primitiveType.getValue().getSelectedId() == 3);
    if (drawPoints)
    {
        //Disable lighting if we draw points
        glDisable(GL_LIGHTING);
        glColor4fv(diffuse.data());
        glDrawArrays(GL_POINTS, 0, vertices.size());
        glEnable(GL_LIGHTING);
        glColor4f(1.0,1.0,1.0,1.0);
    }
    if (g.nbe > 0 && !drawPoints)
    {
        const Edge* indices = NULL;
#ifdef SOFA_HAVE_GLEW
        if (useBufferObjects)
            glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, iboEdges);
        else
#endif
        indices = edges.getData();

        GLenum prim = GL_LINES;
        switch (primitiveType.getValue().getSelectedId())
        {
        case 1:
            serr << "LINES_ADJACENCY primitive type invalid for edge topologies" << sendl;
            break;
        case 2:
#if defined(GL_PATCHES) && defined(SOFA_HAVE_GLEW)
            if (canUsePatches)
            {
                prim = GL_PATCHES;
                glPatchParameteri(GL_PATCH_VERTICES,2);
            }
#endif
            break;
        default:
            break;
        }

        glDrawElements(prim, g.nbe * 2, GL_UNSIGNED_INT, indices + g.edge0);

#ifdef SOFA_HAVE_GLEW
        if (useBufferObjects)
            glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
#endif
    }
    if (g.nbt > 0 && !drawPoints)
    {
        const Triangle* indices = NULL;
#ifdef SOFA_HAVE_GLEW
        if (useBufferObjects)
            glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);
        else
#endif
            indices = triangles.getData();

        GLenum prim = GL_TRIANGLES;
        switch (primitiveType.getValue().getSelectedId())
        {
        case 1:
            serr << "LINES_ADJACENCY primitive type invalid for triangular topologies" << sendl;
            break;
        case 2:
#if defined(GL_PATCHES) && defined(SOFA_HAVE_GLEW)
            if (canUsePatches)
            {
                prim = GL_PATCHES;
                glPatchParameteri(GL_PATCH_VERTICES,3);
            }
#endif
            break;
        default:
            break;
        }

        glDrawElements(prim, g.nbt * 3, GL_UNSIGNED_INT, indices + g.tri0);

#ifdef SOFA_HAVE_GLEW
        if (useBufferObjects)
            glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
#endif
    }
    if (g.nbq > 0 && !drawPoints)
    {
        const Quad* indices = NULL;
#ifdef SOFA_HAVE_GLEW
        if (useBufferObjects)
            glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
        else
#endif
            indices = quads.getData();


        GLenum prim = GL_QUADS;
        switch (primitiveType.getValue().getSelectedId())
        {
        case 1:
#ifndef GL_LINES_ADJACENCY_EXT
            serr << "GL_LINES_ADJACENCY_EXT not defined, please activage GLEW" << sendl;
#else
            {
                prim = GL_LINES_ADJACENCY_EXT;
            }
#endif
            break;
        case 2:
#if defined(GL_PATCHES) && defined(SOFA_HAVE_GLEW)
            if (canUsePatches)
            {
                prim = GL_PATCHES;
                glPatchParameteri(GL_PATCH_VERTICES,4);
            }
#endif
            break;
        default:
            break;
        }

        glDrawElements(prim, g.nbq * 4, GL_UNSIGNED_INT, indices + g.quad0);

#ifdef SOFA_HAVE_GLEW
        if (useBufferObjects)
            glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
#endif
    }

    if (!tex && m.useTexture && m.activated)
    {
        int indexInTextureArray = materialTextureIdMap[g.materialId];
        if (textures[indexInTextureArray])
        {
            textures[indexInTextureArray]->unbind();
        }
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glDisable(GL_TEXTURE_2D);
    }
}

void OglModel::drawGroups(bool transparent)
{
    if(isToPrint.getValue()==true) {
    m_positions.setPersistent(false);
    m_vnormals.setPersistent(false);
    m_vtexcoords.setPersistent(false);
    m_triangles.setPersistent(false);}

    helper::ReadAccessor< Data< helper::vector<FaceGroup> > > groups = this->groups;

    //for (unsigned int i=0; i<xforms.size(); i++)
    {
        //    float matrix[16];
        //    xforms[i].writeOpenGlMatrix(matrix);
        //    pushTransformMatrix(matrix);

        if (groups.empty())
            drawGroup(-1, transparent);
        else
        {
            for (unsigned int i=0; i<groups.size(); ++i)
                drawGroup(i, transparent);
        }

        //    popTransformMatrix();
    }
}

void OglModel::internalDraw(const core::visual::VisualParams* vparams, bool transparent)
{
//    m_vtexcoords.updateIfDirty();
//    serr<<" OglModel::internalDraw()"<<sendl;
    if (!vparams->displayFlags().getShowVisualModels()) return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& vertices = this->getVertices();
    const ResizableExtVector<Deriv>& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();
    const VecCoord& vtangents= this->getVtangents();
    const VecCoord& vbitangents= this->getVbitangents();
    bool hasTangents = vtangents.size() && vbitangents.size();

    glEnable(GL_LIGHTING);

    //Enable<GL_BLEND> blending;
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glColor3f(1.0 , 1.0, 1.0);

#ifdef SOFA_HAVE_GLEW
    if(VBOGenDone && useVBO.getValue())
    {
        glBindBufferARB(GL_ARRAY_BUFFER, vbo);

        glVertexPointer(3, GL_FLOAT, 0, (char*)NULL + 0);
        glNormalPointer(GL_FLOAT, 0, (char*)NULL + (vertices.size()*sizeof(vertices[0])));

        glBindBufferARB(GL_ARRAY_BUFFER, 0);
    }
    else
#endif // SOFA_HAVE_GLEW
    {
        glVertexPointer (3, GL_FLOAT, 0, vertices.getData());
        glNormalPointer (GL_FLOAT, 0, vnormals.getData());
    }

    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    if ((tex || putOnlyTexCoords.getValue()) )//&& !numberOfTextures)
    {
        if(tex)
        {
            glEnable(GL_TEXTURE_2D);
            tex->bind();
        }
#ifdef SOFA_HAVE_GLEW
        if(VBOGenDone && useVBO.getValue())
        {
            glBindBufferARB(GL_ARRAY_BUFFER, vbo);
            glTexCoordPointer(2, GL_FLOAT, 0, (char*)NULL + (vertices.size()*sizeof(vertices[0])) + (vnormals.size()*sizeof(vnormals[0])) );
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
        }
        else
#endif // SOFA_HAVE_GLEW
        {
            glTexCoordPointer(2, GL_FLOAT, 0, getData(vtexcoords));
        }
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        if (hasTangents)
        {
#ifdef SOFA_HAVE_GLEW
            glClientActiveTexture(GL_TEXTURE1);
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);
            if(VBOGenDone && useVBO.getValue())
            {
                glBindBufferARB(GL_ARRAY_BUFFER, vbo);
                glTexCoordPointer(3, GL_FLOAT, 0,
                        (char*)NULL + (vertices.size()*sizeof(vertices[0])) +
                        (vnormals.size()*sizeof(vnormals[0])) +
                        (vtexcoords.size()*sizeof(vtexcoords[0])));
                glBindBufferARB(GL_ARRAY_BUFFER, 0);
            }
            else
                glTexCoordPointer(3, GL_FLOAT, 0, vtangents.getData());

            glClientActiveTexture(GL_TEXTURE2);
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);
            if(VBOGenDone && useVBO.getValue())
            {
                glBindBufferARB(GL_ARRAY_BUFFER, vbo);
                glTexCoordPointer(3, GL_FLOAT, 0,
                        (char*)NULL + (vertices.size()*sizeof(vertices[0])) +
                        (vnormals.size()*sizeof(vnormals[0])) +
                        (vtexcoords.size()*sizeof(vtexcoords[0])) +
                        (vtangents.size()*sizeof(vtangents[0])));
                glBindBufferARB(GL_ARRAY_BUFFER, 0);
            }
            else
                glTexCoordPointer(3, GL_FLOAT, 0, vbitangents.getData());

            glClientActiveTexture(GL_TEXTURE0);
#endif //  SOFA_HAVE_GLEW
        }
    }

    if (transparent && blendTransparency.getValue())
    {
        glEnable(GL_BLEND);
        if (writeZTransparent.getValue())
            glDepthMask(GL_TRUE);
        else glDepthMask(GL_FALSE);

        glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);

        drawGroups(transparent);

        if (premultipliedAlpha.getValue())
            glBlendFunc(GL_ONE, GL_ONE);
        else
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    }

    if (alphaBlend.getValue())
    {
        glDepthMask(GL_FALSE);
#ifdef SOFA_HAVE_GLEW
        glBlendEquation( blendEq );
#endif // SOFA_HAVE_GLEW
        glBlendFunc( sfactor, dfactor );
        glEnable(GL_BLEND);
    }

    if (!depthTest.getValue())
        glDisable(GL_DEPTH_TEST);

    switch (cullFace.getValue())
    {
    case 1:
        glCullFace(GL_BACK);
        glEnable(GL_CULL_FACE);
        break;
    case 2:
        glCullFace(GL_FRONT);
        glEnable(GL_CULL_FACE);
        break;
    }

    if (lineWidth.isSet())
    {
        glLineWidth(lineWidth.getValue());
    }

    if (pointSize.isSet())
    {
        glPointSize(pointSize.getValue());
    }

    if (pointSmooth.getValue())
    {
        glEnable(GL_POINT_SMOOTH);
    }

    if (lineSmooth.getValue())
    {
        glEnable(GL_LINE_SMOOTH);
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
    }

    drawGroups(transparent);

    if (lineSmooth.getValue())
    {
        glDisable(GL_LINE_SMOOTH);
    }

    if (pointSmooth.getValue())
    {
        glDisable(GL_POINT_SMOOTH);
    }

    if (lineWidth.isSet())
    {
        glLineWidth((GLfloat)1);
    }

    if (pointSize.isSet())
    {
        glPointSize((GLfloat)1);
    }

    switch (cullFace.getValue())
    {
    case 1:
    case 2:
        glDisable(GL_CULL_FACE);
        break;
    }

    if (!depthTest.getValue())
        glEnable(GL_DEPTH_TEST);

    if (alphaBlend.getValue())
    {
        // restore Default value
#ifdef SOFA_HAVE_GLEW
        glBlendEquation( GL_FUNC_ADD );
#endif // SOFA_HAVE_GLEW
        glBlendFunc( GL_ONE, GL_ONE );
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }

    if ( (tex || putOnlyTexCoords.getValue()) )//&& !numberOfTextures)
    {
        if (tex)
        {
            tex->unbind();
            glDisable(GL_TEXTURE_2D);
        }
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
#ifdef SOFA_HAVE_GLEW
        if (hasTangents)
        {
            glClientActiveTexture(GL_TEXTURE1);
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
            glClientActiveTexture(GL_TEXTURE2);
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
            glClientActiveTexture(GL_TEXTURE0);
        }
#endif // SOFA_HAVE_GLEW
    }
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisable(GL_LIGHTING);

    if (transparent && blendTransparency.getValue())
    {
        glDisable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glBlendFunc(GL_ONE, GL_ZERO);
        glDepthMask(GL_TRUE);
    }

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (vparams->displayFlags().getShowNormals())
    {
//#ifdef SOFA_HAVE_GLEW
//        GLhandleARB currentShader = sofa::helper::gl::GLSLShader::GetActiveShaderProgram();
//        sofa::helper::gl::GLSLShader::SetActiveShaderProgram(0);
//#endif // SOFA_HAVE_GLEW
        glColor3f (1.0, 1.0, 1.0);
        for (unsigned int i=0; i<xforms.size(); i++)
        {
            float matrix[16];
            xforms[i].writeOpenGlMatrix(matrix);
            glPushMatrix();
            glMultMatrixf(matrix);

            glBegin(GL_LINES);
            for (unsigned int i = 0; i < vertices.size(); i++)
            {
                glVertex3fv (vertices[i].ptr());
                Coord p = vertices[i] + vnormals[i];
                glVertex3fv (p.ptr());
            }
            glEnd();

            glPopMatrix();
        }
//#ifdef SOFA_HAVE_GLEW
//        sofa::helper::gl::GLSLShader::SetActiveShaderProgram(currentShader);
//#endif // SOFA_HAVE_GLEW
    }
//    m_vtexcoords.updateIfDirty();
}

bool OglModel::hasTransparent()
{
    if(alphaBlend.getValue())
        return true;
    return VisualModelImpl::hasTransparent();
}

bool OglModel::hasTexture()
{
    return !textures.empty() || tex;
}

bool OglModel::loadTexture(const std::string& filename)
{
    helper::io::Image *img = helper::io::Image::Create(filename);
    if (!img)
        return false;
    tex = new helper::gl::Texture(img, true, true, false, srgbTexturing.getValue());
    return true;
}

// a false result indicated problems during textures loading
bool OglModel::loadTextures()
{
    bool result = true;
    textures.clear();

    //count the total number of activated textures
    std::vector<unsigned int> activatedTextures;
    for (unsigned int i = 0 ; i < this->materials.getValue().size() ; ++i)
        if (this->materials.getValue()[i].useTexture && this->materials.getValue()[i].activated)
            activatedTextures.push_back(i);

    for (std::vector< unsigned int>::iterator i = activatedTextures.begin() ; i < activatedTextures.end(); ++i)
    {
        std::string textureFile(this->materials.getValue()[*i].textureFilename);

        if (!sofa::helper::system::DataRepository.findFile(textureFile))
        {
            textureFile = this->fileMesh.getFullPath();
            std::size_t position = textureFile.rfind("/");
            textureFile.replace (position+1,textureFile.length() - position, this->materials.getValue()[*i].textureFilename);

            if (!sofa::helper::system::DataRepository.findFile(textureFile))
            {
                serr   << "ERROR: Texture \"" << this->materials.getValue()[*i].textureFilename << "\" not found"
                        << " in material " << this->materials.getValue()[*i].name <<  sendl;
                result = false;
                continue;
            }
        }

        helper::io::Image *img = helper::io::Image::Create(textureFile);
        if (!img)
        {
            serr << "ERROR: couldn't create an image from file " << this->materials.getValue()[*i].textureFilename << sendl;
            result = false;
            continue;
        }
        helper::gl::Texture * text = new helper::gl::Texture(img, true, true, false, srgbTexturing.getValue());
        materialTextureIdMap.insert(std::pair<int, int>(*i,textures.size()));
        textures.push_back( text );
    }

    if (textures.size() != activatedTextures.size())
        serr << "ERROR: " << (activatedTextures.size() - textures.size()) << " textures couldn't be loaded" <<  sendl;



    /**********************************************
     * Load textures for bump mapping
     *********************************************/
//
//    for (unsigned int i = 0 ; i < this->materials.getValue().size() ; i++)
//    {
//       //we count only the bump texture with an activated material
//       if (this->materials.getValue()[i].useBumpMapping && this->materials.getValue()[i].activated)
//       {
//            std::string textureFile(this->materials.getValue()[i].bumpTextureFilename);
//
//            if (!sofa::helper::system::DataRepository.findFile(textureFile))
//            {
//                textureFile = this->fileMesh.getFullPath();
//                unsigned int position = textureFile.rfind("/");
//                textureFile.replace (position+1,textureFile.length() - position, this->materials.getValue()[i].bumpTextureFilename);
//
//                if (!sofa::helper::system::DataRepository.findFile(textureFile))
//                {
//                    serr << "Texture \"" << this->materials.getValue()[i].bumpTextureFilename << "\" not found"
//                            << " in material " << this->materials.getValue()[i].name << " for OglModel " << this->name
//                            << "(\""<< this->fileMesh.getFullPath() << "\")" << sendl;
//                    break;
//                }
//            }
//
//            helper::io::Image *img = helper::io::Image::Create(textureFile);
//            if (!img)
//            {
//               msg_error() << "Error:OglModel:loadTextures: couldn't create an image from file " << this->materials.getValue()[i].bumpTextureFilename << std::endl;
//               return false;
//            }
//            helper::gl::Texture * text = new helper::gl::Texture(img, true, true, false, srgbTexturing.getValue());
//            materialTextureIdMap.insert(std::pair<int, int>(i,textures.size()));
//            textures.push_back( text );
//
//            msg_info() << "\r\033[K" << i+1 << "/" << this->materials.getValue().size() << " textures loaded for bump mapping for OglModel " << this->getName()
//                    << "(loading "<<textureFile << ")"<< std::flush;
//       }
//    }
    return result;
}

void OglModel::initVisual()
{
    initTextures();

    initDone = true;
#ifdef NO_VBO
    canUseVBO = false;
#else
#if !defined(PS3)
    static bool vboAvailable = false; // check the vbo availability

    static bool init = false;
    if(!init)
    {
        vboAvailable = CanUseGlExtension( "GL_ARB_vertex_buffer_object" );
        init = true;
    }

    canUseVBO = vboAvailable;
#elif PS3
    canUseVBO = true;
#endif

    if (useVBO.getValue() && !canUseVBO)
    {
        serr << "OglModel : VBO is not supported by your GPU" << sendl;
    }

#endif

#if defined(SOFA_HAVE_GLEW) && !defined(PS3)
    if (primitiveType.getValue().getSelectedId() == 1 && !GLEW_EXT_geometry_shader4)
    {
        serr << "GL_EXT_geometry_shader4 not supported by your graphics card and/or OpenGL driver." << sendl;
    }

//#ifdef GL_ARB_tessellation_shader
    canUsePatches = (glewIsSupported("GL_ARB_tessellation_shader")!=0);
//#endif

    if (primitiveType.getValue().getSelectedId() == 2 && !canUsePatches)
    {
#ifdef GL_ARB_tessellation_shader
        serr << "GL_ARB_tessellation_shader not supported by your graphics card and/or OpenGL driver." << sendl;
#else
        serr << "GL_ARB_tessellation_shader not defined, please update GLEW to 1.5.4+" << sendl;
#endif
        serr << "GL Version: " << glGetString(GL_VERSION) << sendl;
        serr << "GL Vendor : " << glGetString(GL_VENDOR) << sendl;
        serr << "GL Extensions: " << glGetString(GL_EXTENSIONS) << sendl;
    }
#endif

    updateBuffers();

    // forcing the normal computation if we do not want to use the given ones
    if( !this->m_useNormals.getValue() ) { this->m_vnormals.beginWriteOnly()->clear(); this->m_vnormals.endEdit(); }
    computeNormals();

    if (m_updateTangents.getValue())
        computeTangents();

    if ( alphaBlend.getValue() )
    {
        blendEq = getGLenum( blendEquation.getValue().getSelectedItem().c_str() );
        sfactor = getGLenum( sourceFactor.getValue().getSelectedItem().c_str() );
        dfactor = getGLenum( destFactor.getValue().getSelectedItem().c_str() );
    }

}

void OglModel::initTextures()
{
    if (tex)
    {
        tex->init();
    }
    else
    {
        if (!textures.empty())
        {
            for (unsigned int i = 0 ; i < textures.size() ; i++)
            {
                textures[i]->init();
            }
        }
    }
}
#ifdef SOFA_HAVE_GLEW
void OglModel::createVertexBuffer()
{
    glGenBuffersARB(1, &vbo);
    initVertexBuffer();
    VBOGenDone = true;
}

void OglModel::createEdgesIndicesBuffer()
{
    glGenBuffersARB(1, &iboEdges);
    initEdgesIndicesBuffer();
    useEdges = true;
}

void OglModel::createTrianglesIndicesBuffer()
{
    glGenBuffersARB(1, &iboTriangles);
    initTrianglesIndicesBuffer();
    useTriangles = true;
}


void OglModel::createQuadsIndicesBuffer()
{
    glGenBuffersARB(1, &iboQuads);
    initQuadsIndicesBuffer();
    useQuads = true;
}


void OglModel::initVertexBuffer()
{
    unsigned positionsBufferSize, normalsBufferSize;
    unsigned textureCoordsBufferSize = 0, tangentsBufferSize = 0, bitangentsBufferSize = 0;
    const VecCoord& vertices = this->getVertices();
    const VecCoord& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();
    const VecCoord& vtangents= this->getVtangents();
    const VecCoord& vbitangents= this->getVbitangents();
    bool hasTangents = vtangents.size() && vbitangents.size();

    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));
    normalsBufferSize = (vnormals.size()*sizeof(vnormals[0]));
    if (tex || putOnlyTexCoords.getValue() || !textures.empty())
    {
        textureCoordsBufferSize = vtexcoords.size() * sizeof(vtexcoords[0]);

        if (hasTangents)
        {
            tangentsBufferSize = vtangents.size() * sizeof(vtangents[0]);
            bitangentsBufferSize = vbitangents.size() * sizeof(vbitangents[0]);
        }
    }

    unsigned int totalSize = positionsBufferSize + normalsBufferSize + textureCoordsBufferSize +
            tangentsBufferSize + bitangentsBufferSize;

    glBindBufferARB(GL_ARRAY_BUFFER, vbo);
    //Vertex Buffer creation
    glBufferDataARB(GL_ARRAY_BUFFER,
            totalSize,
            NULL,
            GL_DYNAMIC_DRAW);


    updateVertexBuffer();

    glBindBufferARB(GL_ARRAY_BUFFER, 0);
}


void OglModel::initEdgesIndicesBuffer()
{
    const ResizableExtVector<Edge>& edges = this->getEdges();

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, iboEdges);

    glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER, edges.size()*sizeof(edges[0]), NULL, GL_DYNAMIC_DRAW);
    updateEdgesIndicesBuffer();

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::initTrianglesIndicesBuffer()
{
    const ResizableExtVector<Triangle>& triangles = this->getTriangles();

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);

    glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER, triangles.size()*sizeof(triangles[0]), NULL, GL_DYNAMIC_DRAW);
    updateTrianglesIndicesBuffer();

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::initQuadsIndicesBuffer()
{
    const ResizableExtVector<Quad>& quads = this->getQuads();

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
    glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER, quads.size()*sizeof(quads[0]), NULL, GL_DYNAMIC_DRAW);
    updateQuadsIndicesBuffer();
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateVertexBuffer()
{
    const VecCoord& vertices = this->getVertices();
    const VecCoord& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();
    const VecCoord& vtangents= this->getVtangents();
    const VecCoord& vbitangents= this->getVbitangents();
    bool hasTangents = vtangents.size() && vbitangents.size();

    unsigned positionsBufferSize, normalsBufferSize;
    unsigned textureCoordsBufferSize = 0, tangentsBufferSize = 0, bitangentsBufferSize = 0;

    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));
    normalsBufferSize = (vnormals.size()*sizeof(vnormals[0]));
    if (tex || putOnlyTexCoords.getValue() || !textures.empty())
    {
        textureCoordsBufferSize = vtexcoords.size() * sizeof(vtexcoords[0]);

        if (hasTangents)
        {
            tangentsBufferSize = vtangents.size() * sizeof(vtangents[0]);
            bitangentsBufferSize = vbitangents.size() * sizeof(vbitangents[0]);
        }
    }

    glBindBufferARB(GL_ARRAY_BUFFER, vbo);
    //Positions
    glBufferSubDataARB(GL_ARRAY_BUFFER,
            0,
            positionsBufferSize,
            vertices.getData());

    //Normals
    glBufferSubDataARB(GL_ARRAY_BUFFER,
            positionsBufferSize,
            normalsBufferSize,
            vnormals.getData());

    //Texture coords
    if(tex || putOnlyTexCoords.getValue() ||!textures.empty())
    {
        glBufferSubDataARB(GL_ARRAY_BUFFER,
                positionsBufferSize + normalsBufferSize,
                textureCoordsBufferSize,
                getData(vtexcoords));

        if (hasTangents)
        {
            glBufferSubDataARB(GL_ARRAY_BUFFER,
                    positionsBufferSize + normalsBufferSize + textureCoordsBufferSize,
                    tangentsBufferSize,
                    vtangents.getData());

            glBufferSubDataARB(GL_ARRAY_BUFFER,
                    positionsBufferSize + normalsBufferSize + textureCoordsBufferSize + tangentsBufferSize,
                    bitangentsBufferSize,
                    vbitangents.getData());
        }
    }

    glBindBufferARB(GL_ARRAY_BUFFER, 0);

}

void OglModel::updateEdgesIndicesBuffer()
{
    const ResizableExtVector<Edge>& edges = this->getEdges();
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, iboEdges);
    glBufferSubDataARB(GL_ELEMENT_ARRAY_BUFFER, 0, edges.size()*sizeof(edges[0]), &edges[0]);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateTrianglesIndicesBuffer()
{
    const ResizableExtVector<Triangle>& triangles = this->getTriangles();
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);
    glBufferSubDataARB(GL_ELEMENT_ARRAY_BUFFER, 0, triangles.size()*sizeof(triangles[0]), &triangles[0]);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateQuadsIndicesBuffer()
{
    const ResizableExtVector<Quad>& quads = this->getQuads();
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
    glBufferSubDataARB(GL_ELEMENT_ARRAY_BUFFER, 0, quads.size()*sizeof(quads[0]), &quads[0]);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
}
#endif
void OglModel::updateBuffers()
{
    const ResizableExtVector<Edge>& edges = this->getEdges();
    const ResizableExtVector<Triangle>& triangles = this->getTriangles();
    const ResizableExtVector<Quad>& quads = this->getQuads();
    const VecCoord& vertices = this->getVertices();
    const VecDeriv& normals = this->getVnormals();
    const VecTexCoord& texCoords = this->getVtexcoords();
    const VecCoord& tangents = this->getVtangents();
    const VecCoord& bitangents = this->getVbitangents();

    if (initDone)
    {
#ifdef SOFA_HAVE_GLEW
        if (useVBO.getValue() && canUseVBO)
        {
            if(!VBOGenDone)
            {
                createVertexBuffer();
                //Index Buffer Object
                //Edges indices
                if(edges.size() > 0)
                    createEdgesIndicesBuffer();
                //Triangles indices
                if(triangles.size() > 0)
                    createTrianglesIndicesBuffer();
                //Quads indices
                if(quads.size() > 0)
                    createQuadsIndicesBuffer();
            }
            //Update VBO & IBO
            else
            {
                if(oldVerticesSize != vertices.size() ||
                   oldNormalsSize != normals.size() ||
                   oldTexCoordsSize != texCoords.size() ||
                   oldTangentsSize != tangents.size() ||
                   oldBitangentsSize != bitangents.size())
                    initVertexBuffer();
                else
                    updateVertexBuffer();
                //Indices
                //Edges
                if(useEdges)
                    if(oldEdgesSize != edges.size())
                        initEdgesIndicesBuffer();
                    else
                        updateEdgesIndicesBuffer();
                else if (edges.size() > 0)
                    createEdgesIndicesBuffer();

                //Triangles
                if(useTriangles)
                    if(oldTrianglesSize != triangles.size())
                        initTrianglesIndicesBuffer();
                    else
                        updateTrianglesIndicesBuffer();
                else if (triangles.size() > 0)
                    createTrianglesIndicesBuffer();

                //Quads
                if (useQuads)
                    if(oldQuadsSize != quads.size())
                        initQuadsIndicesBuffer();
                    else
                        updateQuadsIndicesBuffer();
                else if (quads.size() > 0)
                    createQuadsIndicesBuffer();
            }
            oldVerticesSize = vertices.size();
            oldNormalsSize = normals.size();
            oldTexCoordsSize = texCoords.size();
            oldTangentsSize = tangents.size();
            oldBitangentsSize = bitangents.size();
            oldEdgesSize = edges.size();
            oldTrianglesSize = triangles.size();
            oldQuadsSize = quads.size();
        }
#endif
    }

}


GLenum OglModel::getGLenum(const char* c ) const
{

    if ( strcmp( c, "GL_ZERO") == 0)
    {
        return GL_ZERO;
    }
    else if  ( strcmp( c, "GL_ONE") == 0)
    {
        return GL_ONE;
    }
    else if (strcmp( c, "GL_SRC_ALPHA") == 0 )
    {
        return GL_SRC_ALPHA;
    }
    else if (strcmp( c, "GL_ONE_MINUS_SRC_ALPHA") == 0 )
    {
        return GL_ONE_MINUS_SRC_ALPHA;
    }
#ifdef SOFA_HAVE_GLEW
    // .... add ohter OGL symbolic constants
    // glBlendEquation Value
    else if  ( strcmp( c, "GL_FUNC_ADD") == 0)
    {
        return GL_FUNC_ADD;
    }
    else if (strcmp( c, "GL_FUNC_SUBTRACT") == 0 )
    {
        return GL_FUNC_SUBTRACT;
    }
    else if (strcmp( c, "GL_MAX") == 0 )
    {
        return GL_MAX;
    }
    else if (strcmp( c, "GL_MIN") == 0 )
    {
        return GL_MIN;
    }
#endif // SOFA_HAVE_GLEW
    else
    {
        msg_warning()   << " OglModel - not valid or not supported openGL enum value: " << c ;
        return GL_ZERO;
    }


}


} // namespace visualmodel

} // namespace component

} // namespace sofa

