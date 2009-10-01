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
#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>
#include <sofa/helper/gl/RAII.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sstream>
#include <string.h>


namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(OglModel)

int OglModelClass = core::RegisterObject("Generic visual model for OpenGL display")
        .add< OglModel >()
        ;


OglModel::OglModel()
    : premultipliedAlpha(initData(&premultipliedAlpha, (bool) false, "premultipliedAlpha", "is alpha premultiplied ?"))
#ifdef SOFA_HAVE_GLEW
    , useVBO(initData(&useVBO, (bool) false, "useVBO", "Use VBO for rendering"))
#else
    , useVBO(initData(&useVBO, (bool) true, "useVBO", "Use VBO for rendering"))
#endif
    , writeZTransparent(initData(&writeZTransparent, (bool) false, "writeZTransparent", "Write into Z Buffer for Transparent Object"))
    , tex(NULL), canUseVBO(false), VBOGenDone(false), initDone(false), useTriangles(false), useQuads(false)
    , oldTrianglesSize(0), oldQuadsSize(0)
{
}

OglModel::~OglModel()
{
    if (tex!=NULL) delete tex;
}

void OglModel::internalDraw()
{
    //serr<<" OglModel::internalDraw()"<<sendl;
    if (!getContext()->getShowVisualModels()) return;

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glEnable(GL_LIGHTING);

    //Enable<GL_BLEND> blending;
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glColor3f(1.0 , 1.0, 1.0);
    Vec4f ambient = material.getValue().useAmbient?material.getValue().ambient:Vec4f();
    Vec4f diffuse = material.getValue().useDiffuse?material.getValue().diffuse:Vec4f();
    Vec4f specular = material.getValue().useSpecular?material.getValue().specular:Vec4f();
    Vec4f emissive = material.getValue().useEmissive?material.getValue().emissive:Vec4f();
    float shininess = material.getValue().useShininess?material.getValue().shininess:45;

    if (shininess == 0.0f)
    {
        specular.clear();
        shininess = 1;
    }

    if (isTransparent())
    {
        emissive[3] = 0; //diffuse[3];
        ambient[3] = 0; //diffuse[3];
        //diffuse[3] = 0;
        specular[3] = 0;
    }
    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT, ambient.ptr());
    glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse.ptr());
    glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular.ptr());
    glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive.ptr());
    glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, shininess);

    if(VBOGenDone && useVBO.getValue())
    {
#ifdef SOFA_HAVE_GLEW
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glVertexPointer(3, GL_FLOAT, 0, (char*)NULL + 0);
        glNormalPointer(GL_FLOAT, 0, (char*)NULL + (vertices.size()*sizeof(vertices[0])));

        glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
    }
    else
    {
        glVertexPointer (3, GL_FLOAT, 0, vertices.getData());
        glNormalPointer (GL_FLOAT, 0, vnormals.getData());
    }

    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    if (tex || putOnlyTexCoords.getValue())
    {
        glEnable(GL_TEXTURE_2D);
        if(tex)
            tex->bind();

        if(VBOGenDone && useVBO.getValue())
        {
#ifdef SOFA_HAVE_GLEW
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glTexCoordPointer(2, GL_FLOAT, 0, (char*)NULL + (vertices.size()*sizeof(vertices[0])) + (vnormals.size()*sizeof(vnormals[0])) );
            glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
        }
        else
        {
            glTexCoordPointer(2, GL_FLOAT, 0, vtexcoords.getData());
        }
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    if (isTransparent())
    {
        glEnable(GL_BLEND);
        if (writeZTransparent.getValue())
            glDepthMask(GL_TRUE);
        else glDepthMask(GL_FALSE);

        glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);

        for (unsigned int i=0; i<xforms.size(); i++)
        {
            float matrix[16];
            xforms[i].writeOpenGlMatrix(matrix);
            glPushMatrix();
            glMultMatrixf(matrix);
            if(VBOGenDone && useVBO.getValue())
            {
#ifdef SOFA_HAVE_GLEW
                if (!triangles.empty())
                {
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);
                    glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, (char*)NULL + 0);
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
                }
                if (!quads.empty())
                {
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
                    glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, (char*)NULL + 0);
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
                }
#endif
            }
            else
            {
                if (!triangles.empty())
                    glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, triangles.getData());
                if (!quads.empty())
                    glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, quads.getData());
            }
        }

        glPopMatrix();

        if (premultipliedAlpha.getValue())
            glBlendFunc(GL_ONE, GL_ONE);
        else
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    }

    for (unsigned int i=0; i<xforms.size(); i++)
    {
        //serr<<"OglModel::internalDraw() 4, quads.size() = "<<quads.size()<<sendl;
        float matrix[16];
        xforms[i].writeOpenGlMatrix(matrix);
        //for( int k=0; k<16; k++ ) serr<<matrix[k]<<" "; serr<<sendl;
        glPushMatrix();
        glMultMatrixf(matrix);

        //glutWireCube( 3 );
        if (VBOGenDone && useVBO.getValue())
        {
#ifdef SOFA_HAVE_GLEW
            if (!triangles.empty())
            {
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);
                glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, (char*)NULL + 0);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            }
            if (!quads.empty())
            {
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
                glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, (char*)NULL + 0);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            }
#endif
        }
        else
        {
            if (!triangles.empty())
                glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, triangles.getData());
            if (!quads.empty())
                glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, quads.getData());
        }

        glPopMatrix();
    }
    if (tex || putOnlyTexCoords.getValue())
    {
        if (tex)
            tex->unbind();
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glDisable(GL_TEXTURE_2D);
    }
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisable(GL_LIGHTING);
    if (isTransparent())
    {
        glDisable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glBlendFunc(GL_ONE, GL_ZERO);
        glDepthMask(GL_TRUE);
    }

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (getContext()->getShowNormals())
    {
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
    }

}

bool OglModel::loadTexture(const std::string& filename)
{
    helper::io::Image *img = helper::io::Image::Create(filename);
    if (!img)
        return false;
    tex = new helper::gl::Texture(img);
    return true;
}

void OglModel::initVisual()
{
    if (tex)
    {
        tex->init();
    }

    initDone = true;
#ifdef SOFA_HAVE_GLEW
    //This test is not enough to detect if we can enable the VBO.
    canUseVBO = (GLEW_ARB_vertex_buffer_object!=0);
#endif

    if (useVBO.getValue() && !canUseVBO)
    {
        std::cerr << "OglModel : VBO is not supported by your GPU ; will use display list instead" << std::endl;
    }

    updateBuffers();

}

void OglModel::initTextures()
{
    if (tex)
    {
        tex->init();
    }
}
#ifdef SOFA_HAVE_GLEW
void OglModel::createVertexBuffer()
{


    glGenBuffers(1, &vbo);
    initVertexBuffer();
    VBOGenDone = true;
}

void OglModel::createTrianglesIndicesBuffer()
{
    glGenBuffers(1, &iboTriangles);
    initTrianglesIndicesBuffer();
    useTriangles = true;
}


void OglModel::createQuadsIndicesBuffer()
{
    glGenBuffers(1, &iboQuads);
    initQuadsIndicesBuffer();
    useQuads = true;
}


void OglModel::initVertexBuffer()
{
    unsigned int positionsBufferSize, normalsBufferSize, textureCoordsBufferSize = 0;
    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));
    normalsBufferSize = (vnormals.size()*sizeof(vnormals[0]));
    if (tex || putOnlyTexCoords.getValue())
        textureCoordsBufferSize = vtexcoords.size() * sizeof(vtexcoords[0]);

    unsigned int totalSize = positionsBufferSize + normalsBufferSize + textureCoordsBufferSize;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //Vertex Buffer creation
    glBufferData(GL_ARRAY_BUFFER,
            totalSize,
            NULL,
            GL_DYNAMIC_DRAW);


    updateVertexBuffer();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void OglModel::initTrianglesIndicesBuffer()
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.size()*sizeof(triangles[0]), NULL, GL_DYNAMIC_DRAW);
    updateTrianglesIndicesBuffer();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::initQuadsIndicesBuffer()
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quads.size()*sizeof(quads[0]), NULL, GL_DYNAMIC_DRAW);
    updateQuadsIndicesBuffer();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateVertexBuffer()
{
    unsigned int positionsBufferSize, normalsBufferSize, textureCoordsBufferSize = 0;
    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));
    normalsBufferSize = (vnormals.size()*sizeof(vnormals[0]));
    if (tex || putOnlyTexCoords.getValue())
        textureCoordsBufferSize = vtexcoords.size() * sizeof(vtexcoords[0]);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //Positions
    glBufferSubData(GL_ARRAY_BUFFER,
            0,
            positionsBufferSize,
            vertices.getData());

    //Normals
    glBufferSubData(GL_ARRAY_BUFFER,
            positionsBufferSize,
            normalsBufferSize,
            vnormals.getData());
    //Texture coords
    if(tex || putOnlyTexCoords.getValue())
    {
        glBufferSubData(GL_ARRAY_BUFFER,
                positionsBufferSize + normalsBufferSize,
                textureCoordsBufferSize,
                vtexcoords.getData());
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void OglModel::updateTrianglesIndicesBuffer()
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, triangles.size()*sizeof(triangles[0]), &triangles[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateQuadsIndicesBuffer()
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, quads.size()*sizeof(quads[0]), &quads[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
#endif
void OglModel::updateBuffers()
{
    if (initDone)
    {
#ifdef SOFA_HAVE_GLEW
        if (useVBO.getValue() && canUseVBO)
        {
            if(!VBOGenDone)
            {
                createVertexBuffer();
                //Index Buffer Object
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
                if(oldVerticesSize != vertices.size())
                    initVertexBuffer();
                else
                    updateVertexBuffer();
                //Indices
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
            oldTrianglesSize = triangles.size();
            oldQuadsSize = quads.size();
        }
#endif
    }

}

} // namespace visualmodel

} // namespace component

} // namespace sofa

