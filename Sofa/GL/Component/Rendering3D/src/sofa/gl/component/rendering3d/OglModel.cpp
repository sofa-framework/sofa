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
#include <sofa/gl/component/rendering3d/OglModel.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/gl/gl.h>
#include <sofa/gl/RAII.h>
#include <sofa/type/vector.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <cstring>
#include <sofa/type/RGBAColor.h>

namespace sofa::gl::component::rendering3d
{

using sofa::type::RGBAColor;
using sofa::type::Material;
using namespace sofa::type;

int OglModelClass = core::RegisterObject("Generic visual model for OpenGL display")
    .add< OglModel >();


OglModel::OglModel()
    : blendTransparency(initData(&blendTransparency, true, "blendTranslucency", "Blend transparent parts"))
    , premultipliedAlpha(initData(&premultipliedAlpha, false, "premultipliedAlpha", "is alpha premultiplied ?"))
    , writeZTransparent(initData(&writeZTransparent, false, "writeZTransparent", "Write into Z Buffer for Transparent Object"))
    , alphaBlend(initData(&alphaBlend, false, "alphaBlend", "Enable alpha blending"))
    , depthTest(initData(&depthTest, true, "depthTest", "Enable depth testing"))
    , cullFace(initData(&cullFace, 0, "cullFace", "Face culling (0 = no culling, 1 = cull back faces, 2 = cull front faces)"))
    , lineWidth(initData(&lineWidth, 1.0f, "lineWidth", "Line width (set if != 1, only for lines rendering)"))
    , pointSize(initData(&pointSize, 1.0f, "pointSize", "Point size (set if != 1, only for points rendering)"))
    , lineSmooth(initData(&lineSmooth, false, "lineSmooth", "Enable smooth line rendering"))
    , pointSmooth(initData(&pointSmooth, false, "pointSmooth", "Enable smooth point rendering"))
    , isEnabled( initData(&isEnabled, true, "isEnabled", "Activate/deactive the component."))
    , primitiveType( initData(&primitiveType, "primitiveType", "Select types of primitives to send (necessary for some shader types such as geometry or tesselation)"))
    , blendEquation( initData(&blendEquation, "blendEquation", "if alpha blending is enabled this specifies how source and destination colors are combined") )
    , sourceFactor( initData(&sourceFactor, "sfactor", "if alpha blending is enabled this specifies how the red, green, blue, and alpha source blending factors are computed") )
    , destFactor( initData(&destFactor, "dfactor", "if alpha blending is enabled this specifies how the red, green, blue, and alpha destination blending factors are computed") )
    , tex(nullptr)
    , vbo(0), iboEdges(0), iboTriangles(0), iboQuads(0)
    , VBOGenDone(false), initDone(false), useEdges(false), useTriangles(false), useQuads(false), canUsePatches(false)
    , oldVerticesSize(0), oldNormalsSize(0), oldTexCoordsSize(0), oldTangentsSize(0), oldBitangentsSize(0), oldEdgesSize(0), oldTrianglesSize(0), oldQuadsSize(0)
    , edgesRevision(-1), trianglesRevision(-1), quadsRevision(-1)
{

    textures.clear();

    blendEquation.setValue({"GL_FUNC_ADD", "GL_FUNC_SUBTRACT", "GL_MIN", "GL_MAX"});
    sourceFactor.setValue(helper::OptionsGroup{"GL_ZERO", "GL_ONE", "GL_SRC_ALPHA", "GL_ONE_MINUS_SRC_ALPHA"}.setSelectedItem(2));
    destFactor.setValue(helper::OptionsGroup{"GL_ZERO", "GL_ONE", "GL_SRC_ALPHA", "GL_ONE_MINUS_SRC_ALPHA"}.setSelectedItem(3));
    primitiveType.setValue(helper::OptionsGroup{"DEFAULT", "LINES_ADJACENCY", "PATCHES", "POINTS"}.setSelectedItem(0));
}

void OglModel::deleteTextures()
{
    if (tex!=nullptr) delete tex;

    for (unsigned int i = 0 ; i < textures.size() ; i++)
    {
        delete textures[i];
    }
}

void OglModel::deleteBuffers()
{
    // NB fjourdes : I don t know why gDEBugger still reports
    // graphics memory leaks after destroying the GLContext
    // even if the vbos destruction is claimed with the following
    // lines...
    if( vbo > 0 )
    {
        glDeleteBuffers(1,&vbo);
    }
    if( iboEdges > 0)
    {
        glDeleteBuffers(1,&iboEdges);
    }
    if( iboTriangles > 0)
    {
        glDeleteBuffers(1,&iboTriangles);
    }
    if( iboQuads > 0 )
    {
        glDeleteBuffers(1,&iboQuads);
    }
}

OglModel::~OglModel()
{
    deleteTextures();
    deleteBuffers();
}

void OglModel::drawGroup(int ig, bool transparent)
{
    glEnable(GL_NORMALIZE);

    const Inherit::VecVisualEdge& edges = this->getEdges();
    const Inherit::VecVisualTriangle& triangles = this->getTriangles();
    const Inherit::VecVisualQuad& quads = this->getQuads();

    const VecCoord& vertices = this->getVertices();
    const VecDeriv& vnormals = this->getVnormals();

    FaceGroup g;
    if (ig < 0)
    {
        g.materialId = -1;
        g.edge0 = 0;
        g.nbe = int(edges.size());
        g.tri0 = 0;
        g.nbt = int(triangles.size());
        g.quad0 = 0;
        g.nbq = int(quads.size());
    }
    else
    {
        g = this->groups.getValue()[size_t(ig)];
    }
    Material m;
    if (g.materialId < 0)
        m = this->material.getValue();
    else
        m = this->materials.getValue()[size_t(g.materialId)];

    bool isTransparent = (m.useDiffuse && m.diffuse[3] < 1.0f) || hasTransparent();
    if (transparent ^ isTransparent) return;


    if (!tex && m.useTexture && m.activated)
    {
        //get the texture id corresponding to the current material
        size_t indexInTextureArray = size_t(materialTextureIdMap[g.materialId]);
        if (indexInTextureArray < textures.size() && textures[indexInTextureArray])
        {
            textures[indexInTextureArray]->bind();
        }

        glEnable(GL_TEXTURE_2D);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
	    uintptr_t pt = (vertices.size()*sizeof(vertices[0]))
                    + (vnormals.size()*sizeof(vnormals[0]));
        glTexCoordPointer(2, GL_FLOAT, 0, reinterpret_cast<void*>(pt));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
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
    const bool drawPoints = (primitiveType.getValue().getSelectedId() == 3);
    if (drawPoints)
    {
        //Disable lighting if we draw points
        glDisable(GL_LIGHTING);
        glColor4fv(diffuse.data());
        glDrawArrays(GL_POINTS, 0, GLsizei(vertices.size()));
        glEnable(GL_LIGHTING);
        glColor4f(1.0,1.0,1.0,1.0);
    }
    if (g.nbe > 0 && !drawPoints)
    {
        const VisualEdge* indices = nullptr;

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboEdges);

        GLenum prim = GL_LINES;
        switch (primitiveType.getValue().getSelectedId())
        {
        case 1:
            msg_warning() << "LINES_ADJACENCY primitive type invalid for edge topologies" ;
            break;
        case 2:
            prim = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES,2);
            break;
        default:
            break;
        }

        glDrawElements(prim, g.nbe * 2, GL_UNSIGNED_INT, indices + g.edge0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    if (g.nbt > 0 && !drawPoints)
    {
        const VisualTriangle* indices = nullptr;

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);

        GLenum prim = GL_TRIANGLES;
        switch (primitiveType.getValue().getSelectedId())
        {
        case 1:
            msg_warning() << "LINES_ADJACENCY primitive type invalid for triangular topologies" ;
            break;
        case 2:
            prim = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES,3);
            break;
        default:
            break;
        }

        glDrawElements(prim, g.nbt * 3, GL_UNSIGNED_INT, indices + g.tri0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    if (g.nbq > 0 && !drawPoints)
    {
        const VisualQuad* indices = nullptr;

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);

        GLenum prim = GL_QUADS;
        switch (primitiveType.getValue().getSelectedId())
        {
        case 1:
            prim = GL_LINES_ADJACENCY_EXT;
            break;
        case 2:
            prim = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES,4);
            break;
        default:
            break;
        }

        glDrawElements(prim, g.nbq * 4, GL_UNSIGNED_INT, indices + g.quad0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    if (!tex && m.useTexture && m.activated)
    {
        int indexInTextureArray = materialTextureIdMap[g.materialId];
        if (indexInTextureArray < int(textures.size()) && textures[size_t(indexInTextureArray)])
        {
            textures[size_t(indexInTextureArray)]->unbind();
        }
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glDisable(GL_TEXTURE_2D);
    }
}

void OglModel::drawGroups(bool transparent)
{
    const helper::ReadAccessor< Data< type::vector<FaceGroup> > > groups = this->groups;

    if (groups.empty())
    {
        drawGroup(-1, transparent);
    }
    else
    {
        for (size_t i=0; i<groups.size(); ++i)
            drawGroup(int(i), transparent);
    }
}


void glVertex3v(const float* d){ glVertex3fv(d); }
void glVertex3v(const double* d){ glVertex3dv(d); }

template<class T>
GLuint glType(){ return GL_FLOAT; }

template<>
GLuint glType<double>(){ return GL_DOUBLE; }

template<>
GLuint glType<float>(){ return GL_FLOAT; }

template<class InType, class OutType>
void copyVector(const InType& src, OutType& dst)
{
    unsigned int i=0;
    for(auto& item : src)
    {
        dst[i].set(item);
        ++i;
    }
}

void OglModel::internalDraw(const core::visual::VisualParams* vparams, bool transparent)
{
    if (!vparams->displayFlags().getShowVisualModels())
        return;

    if(!isEnabled.getValue())
        return;

    /// Checks that the VBO's are ready.
    if(!VBOGenDone)
        return;

    const VecCoord& vertices = this->getVertices();
    const VecDeriv& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();
    const VecCoord& vtangents= this->getVtangents();
    const VecCoord& vbitangents= this->getVbitangents();
    const bool hasTangents = vtangents.size() && vbitangents.size();


    glEnable(GL_LIGHTING);

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glColor3f(1.0 , 1.0, 1.0);

    /// Force the data to be of float type before sending to opengl...
    const GLuint datatype = GL_FLOAT;
    const GLuint vertexdatasize = sizeof(verticesTmpBuffer[0]);
    const GLuint normaldatasize = sizeof(normalsTmpBuffer[0]);

    const GLulong vertexArrayByteSize = vertices.size() * vertexdatasize;
    const GLulong normalArrayByteSize = vnormals.size() * normaldatasize;

    //// Update the vertex buffers.
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, datatype, 0, nullptr);
    glNormalPointer(datatype, 0, reinterpret_cast<void*>(vertexArrayByteSize));
    glBindBuffer(GL_ARRAY_BUFFER, 0);


    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    if ((tex || putOnlyTexCoords.getValue()) )
    {
        if(tex)
        {
            glEnable(GL_TEXTURE_2D);
            tex->bind();
        }

        const size_t textureArrayByteSize = vtexcoords.size()*sizeof(vtexcoords[0]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glTexCoordPointer(2, GL_FLOAT, 0, reinterpret_cast<void*>(vertexArrayByteSize + normalArrayByteSize ));
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        if (hasTangents)
        {
            const size_t tangentArrayByteSize = vtangents.size()*sizeof(vtangents[0]);

            glClientActiveTexture(GL_TEXTURE1);
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glTexCoordPointer(3, GL_DOUBLE, 0,
                              reinterpret_cast<void*>(vertexArrayByteSize + normalArrayByteSize + textureArrayByteSize));
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glClientActiveTexture(GL_TEXTURE2);
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glTexCoordPointer(3, GL_DOUBLE, 0,
                              reinterpret_cast<void*>(vertexArrayByteSize + normalArrayByteSize
                              + textureArrayByteSize + tangentArrayByteSize));
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glClientActiveTexture(GL_TEXTURE0);
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
        glBlendEquation( blendEq );
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
        glLineWidth(1.0f);
    }

    if (pointSize.isSet())
    {
        glPointSize(1.0f);
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
        glBlendEquation( GL_FUNC_ADD );
        glBlendFunc( GL_ONE, GL_ONE );
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }

    if ( (tex || putOnlyTexCoords.getValue()) )
    {
        if (tex)
        {
            tex->unbind();
            glDisable(GL_TEXTURE_2D);
        }
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        if (hasTangents)
        {
            glClientActiveTexture(GL_TEXTURE1);
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
            glClientActiveTexture(GL_TEXTURE2);
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
            glClientActiveTexture(GL_TEXTURE0);
        }
    }
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisable(GL_LIGHTING);

    if (transparent && blendTransparency.getValue())
    {
        glDisable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_TRUE);
    }

    if (vparams->displayFlags().getShowNormals())
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
                glVertex3v(vertices[i].ptr());
                Coord p = vertices[i] + vnormals[i];
                glVertex3v(p.ptr());
            }
            glEnd();

            glPopMatrix();
        }
    }
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
    tex = new sofa::gl::Texture(img, true, true, false, srgbTexturing.getValue());
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
            const std::size_t position = textureFile.rfind("/");
            textureFile.replace (position+1,textureFile.length() - position, this->materials.getValue()[*i].textureFilename);

            if (!sofa::helper::system::DataRepository.findFile(textureFile))
            {
                msg_error() << "Texture \"" << this->materials.getValue()[*i].textureFilename << "\" not found"
                            << " in material " << this->materials.getValue()[*i].name ;
                result = false;
                continue;
            }
        }

        helper::io::Image *img = helper::io::Image::Create(textureFile);
        if (!img)
        {
            msg_error() << "couldn't create an image from file " << this->materials.getValue()[*i].textureFilename ;
            result = false;
            continue;
        }
        sofa::gl::Texture * text = new sofa::gl::Texture(img, true, true, false, srgbTexturing.getValue());
        materialTextureIdMap.insert(std::pair<int, int>(*i,textures.size()));
        textures.push_back( text );
    }

    if (textures.size() != activatedTextures.size())
        msg_error() << (activatedTextures.size() - textures.size()) << " textures couldn't be loaded" ;

    return result;
}

void OglModel::initVisual()
{
    initTextures();

    initDone = true;
    static bool vboAvailable = false; // check the vbo availability

    static bool init = false;
    if(!init)
    {
        vboAvailable = CanUseGlExtension( "GL_ARB_vertex_buffer_object" );
        init = true;
    }

    if (!vboAvailable)
    {
        msg_warning() << "OglModel : VBO is not supported by your GPU" ;
    }

    if (primitiveType.getValue().getSelectedId() == 1 && !GLEW_EXT_geometry_shader4)
    {
        msg_warning() << "GL_EXT_geometry_shader4 not supported by your graphics card and/or OpenGL driver." ;
    }

    canUsePatches = (glewIsSupported("GL_ARB_tessellation_shader")!=0);

    if (primitiveType.getValue().getSelectedId() == 2 && !canUsePatches)
    {
        msg_warning() << "GL_ARB_tessellation_shader not supported by your graphics card and/or OpenGL driver." ;
        msg_warning() << "GL Version: " << glGetString(GL_VERSION) ;
        msg_warning() << "GL Vendor : " << glGetString(GL_VENDOR) ;
        msg_warning() << "GL Extensions: " << glGetString(GL_EXTENSIONS) ;
    }

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

void OglModel::createVertexBuffer()
{
    glGenBuffers(1, &vbo);
    initVertexBuffer();
    VBOGenDone = true;
}

void OglModel::createEdgesIndicesBuffer()
{
    glGenBuffers(1, &iboEdges);
    initEdgesIndicesBuffer();
    useEdges = true;
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
    size_t positionsBufferSize, normalsBufferSize;
    size_t textureCoordsBufferSize = 0, tangentsBufferSize = 0, bitangentsBufferSize = 0;
    const VecCoord& vertices = this->getVertices();
    const VecCoord& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();
    const VecCoord& vtangents= this->getVtangents();
    const VecCoord& vbitangents= this->getVbitangents();
    const bool hasTangents = vtangents.size() && vbitangents.size();

    positionsBufferSize = (vertices.size()*sizeof(Vec3f));
    normalsBufferSize = (vnormals.size()*sizeof(Vec3f));

    if (tex || putOnlyTexCoords.getValue() || !textures.empty())
    {
        textureCoordsBufferSize = vtexcoords.size() * sizeof(vtexcoords[0]);

        if (hasTangents)
        {
            tangentsBufferSize = vtangents.size() * sizeof(vtangents[0]);
            bitangentsBufferSize = vbitangents.size() * sizeof(vbitangents[0]);
        }
    }

    const size_t totalSize = positionsBufferSize + normalsBufferSize + textureCoordsBufferSize +
            tangentsBufferSize + bitangentsBufferSize;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 long(totalSize),
                 nullptr,
                 GL_DYNAMIC_DRAW);

    updateVertexBuffer();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void OglModel::initEdgesIndicesBuffer()
{
    const Inherit::VecVisualEdge& edges = this->getEdges();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboEdges);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, long(edges.size()*sizeof(edges[0])), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    updateEdgesIndicesBuffer();
}

void OglModel::initTrianglesIndicesBuffer()
{
    const Inherit::VecVisualTriangle& triangles = this->getTriangles();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, long(triangles.size()*sizeof(triangles[0])), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    updateTrianglesIndicesBuffer();
}

void OglModel::initQuadsIndicesBuffer()
{
    const Inherit::VecVisualQuad& quads = this->getQuads();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, long(quads.size()*sizeof(quads[0])), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    updateQuadsIndicesBuffer();
}

void OglModel::updateVertexBuffer()
{

    const VecCoord& vertices = this->getVertices();
    const VecCoord& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();
    const VecCoord& vtangents= this->getVtangents();
    const VecCoord& vbitangents= this->getVbitangents();
    const bool hasTangents = vtangents.size() && vbitangents.size();

    size_t positionsBufferSize, normalsBufferSize;
    size_t textureCoordsBufferSize = 0, tangentsBufferSize = 0, bitangentsBufferSize = 0;

    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));
    normalsBufferSize = (vnormals.size()*sizeof(vnormals[0]));
    const void* positionBuffer = vertices.data();
    const void* normalBuffer = vnormals.data();

    // use only temporary float buffers if vertices/normals are using double
    if constexpr(std::is_same_v<Coord, sofa::type::Vec3d>)
    {
        verticesTmpBuffer.resize( vertices.size() );
        normalsTmpBuffer.resize( vnormals.size() );

        copyVector(vertices, verticesTmpBuffer);
        copyVector(vnormals, normalsTmpBuffer);

        positionBuffer = verticesTmpBuffer.data();
        normalBuffer = normalsTmpBuffer.data();
    }

    positionsBufferSize = (vertices.size()*sizeof(Vec3f));
    normalsBufferSize = (vnormals.size()*sizeof(Vec3f));

    if (tex || putOnlyTexCoords.getValue() || !textures.empty())
    {
        textureCoordsBufferSize = (vtexcoords.size() * sizeof(vtexcoords[0]));

        if (hasTangents)
        {
            tangentsBufferSize = (vtangents.size() * sizeof(vtangents[0]));
            bitangentsBufferSize = (vbitangents.size() * sizeof(vbitangents[0]));
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    //Positions
    glBufferSubData(GL_ARRAY_BUFFER,
        0,
        positionsBufferSize,
        positionBuffer);

    //Normals
    glBufferSubData(GL_ARRAY_BUFFER,
        positionsBufferSize,
        normalsBufferSize,
        normalBuffer);

    ////Texture coords
    if (tex || putOnlyTexCoords.getValue() || !textures.empty())
    {
        glBufferSubData(GL_ARRAY_BUFFER,
            positionsBufferSize + normalsBufferSize,
            textureCoordsBufferSize,
            vtexcoords.data());

        if (hasTangents)
        {
            glBufferSubData(GL_ARRAY_BUFFER,
                positionsBufferSize + normalsBufferSize + textureCoordsBufferSize,
                tangentsBufferSize,
                vtangents.data());

            glBufferSubData(GL_ARRAY_BUFFER,
                positionsBufferSize + normalsBufferSize + textureCoordsBufferSize + tangentsBufferSize,
                bitangentsBufferSize,
                vbitangents.data());
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void OglModel::updateEdgesIndicesBuffer()
{
    const VecVisualEdge& edges = this->getEdges();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboEdges);

    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, long(edges.size()*sizeof(edges[0])), &edges[0]);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateTrianglesIndicesBuffer()
{
    const VecVisualTriangle& triangles = this->getTriangles();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);

    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, long(triangles.size() * sizeof(triangles[0])), &triangles[0]);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateQuadsIndicesBuffer()
{
    const VecVisualQuad& quads = this->getQuads();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);

    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, long(quads.size() * sizeof(quads[0])), &quads[0]);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
void OglModel::updateBuffers()
{
    const Inherit::VecVisualEdge& edges = this->getEdges();
    const Inherit::VecVisualTriangle& triangles = this->getTriangles();
    const Inherit::VecVisualQuad& quads = this->getQuads();

    const VecCoord& vertices = this->getVertices();
    const VecDeriv& normals = this->getVnormals();
    const VecTexCoord& texCoords = this->getVtexcoords();
    const VecCoord& tangents = this->getVtangents();
    const VecCoord& bitangents = this->getVbitangents();

    if (initDone)
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
            // if any topology change then resize buffer
            if (oldVerticesSize != vertices.size() ||
                oldNormalsSize != normals.size() ||
                oldTexCoordsSize != texCoords.size() ||
                oldTangentsSize != tangents.size() ||
                oldBitangentsSize != bitangents.size())
            {
                initVertexBuffer();
            }
            else
            {
                // if no topology change but vertices changes then update buffer
                if (this->modified)
                {
                    updateVertexBuffer();
                }
            }


            //Indices
            //Edges
            if (useEdges && !edges.empty())
            {

                if(oldEdgesSize != edges.size())
                    initEdgesIndicesBuffer();
                else
                    if(edgesRevision < m_edges.getCounter())
                        updateEdgesIndicesBuffer();

            }
            else if (edges.size() > 0)
                createEdgesIndicesBuffer();

            //Triangles
            if (useTriangles && !triangles.empty())
            {
                if (oldTrianglesSize != triangles.size())
                    initTrianglesIndicesBuffer();
                else
                    if (trianglesRevision < m_triangles.getCounter())
                        updateTrianglesIndicesBuffer();
            }
            else if (triangles.size() > 0)
                createTrianglesIndicesBuffer();

            //Quads
            if (useQuads && !quads.empty())
            {
                if(oldQuadsSize != quads.size())
                    initQuadsIndicesBuffer();
                else
                    if (quadsRevision < m_quads.getCounter())
                        updateQuadsIndicesBuffer();
            }
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

        edgesRevision = m_edges.getCounter();
        trianglesRevision = m_triangles.getCounter();
        quadsRevision = m_quads.getCounter();
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
    else
    {
        msg_warning() << " OglModel - not valid or not supported openGL enum value: " << c ;
        return GL_ZERO;
    }
}


} // namespace sofa::gl::component::rendering3d
