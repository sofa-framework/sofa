/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/RAII.h>
#include <sofa/helper/vector.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <cstring>
#include <sofa/helper/types/RGBAColor.h>

//#define DEBUG_DRAW
namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;

static int OglModelClass = core::RegisterObject("Generic visual model for OpenGL display")
        .add< OglModel >()
        ;

template<class T>
const T* getData(const sofa::helper::vector<T>& v) { return &v[0]; }


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
    if (tex!=nullptr) delete tex;

    for (unsigned int i = 0 ; i < textures.size() ; i++)
    {
        delete textures[i];
    }

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

void OglModel::parse(core::objectmodel::BaseObjectDescription* arg)
{
    if (arg->getAttribute("isToPrint")!=nullptr)
    {
        msg_deprecated() << "The 'isToPrint' data field has been deprecated in Sofa 19.06 due to lack of consistency in how it should work." << msgendl
                            "Please contact sofa-dev team in case you need similar.";
    }
    Inherit1::parse(arg);
}




void OglModel::drawGroup(int ig, bool transparent)
{
    glEnable(GL_NORMALIZE);

    const Inherit::VecEdge& edges = this->getEdges();
    const Inherit::VecTriangle& triangles = this->getTriangles();
    const Inherit::VecQuad& quads = this->getQuads();
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
        const Edge* indices = nullptr;

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
        const Triangle* indices = nullptr;

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
        const Quad* indices = nullptr;

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
    helper::ReadAccessor< Data< helper::vector<FaceGroup> > > groups = this->groups;

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

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& vertices = this->getVertices();
    const VecDeriv& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();
    const VecCoord& vtangents= this->getVtangents();
    const VecCoord& vbitangents= this->getVbitangents();
    bool hasTangents = vtangents.size() && vbitangents.size();


    glEnable(GL_LIGHTING);

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glColor3f(1.0 , 1.0, 1.0);

    /// Force the data to be of float type before sending to opengl...
    GLuint datatype = GL_FLOAT;
    GLuint vertexdatasize = sizeof(verticesTmpBuffer[0]);
    GLuint normaldatasize = sizeof(normalsTmpBuffer[0]);

    GLulong vertexArrayByteSize = vertices.size() * vertexdatasize;
    GLulong normalArrayByteSize = vnormals.size() * normaldatasize;

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

        size_t textureArrayByteSize = vtexcoords.size()*sizeof(vtexcoords[0]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glTexCoordPointer(2, GL_FLOAT, 0, reinterpret_cast<void*>(vertexArrayByteSize + normalArrayByteSize ));
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        if (hasTangents)
        {
            size_t tangentArrayByteSize = vtangents.size()*sizeof(vtangents[0]);

            glClientActiveTexture(GL_TEXTURE1);
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glTexCoordPointer(3, GL_FLOAT, 0,
                              reinterpret_cast<void*>(vertexArrayByteSize + normalArrayByteSize + textureArrayByteSize));
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glClientActiveTexture(GL_TEXTURE2);
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glTexCoordPointer(3, GL_FLOAT, 0,
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

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

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
        helper::gl::Texture * text = new helper::gl::Texture(img, true, true, false, srgbTexturing.getValue());
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
    bool hasTangents = vtangents.size() && vbitangents.size();

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

    size_t totalSize = positionsBufferSize + normalsBufferSize + textureCoordsBufferSize +
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
    const Inherit::VecEdge& edges = this->getEdges();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboEdges);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, long(edges.size()*sizeof(edges[0])), nullptr, GL_DYNAMIC_DRAW);

    updateEdgesIndicesBuffer();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::initTrianglesIndicesBuffer()
{
    const Inherit::VecTriangle& triangles = this->getTriangles();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, long(triangles.size()*sizeof(triangles[0])), nullptr, GL_DYNAMIC_DRAW);

    updateTrianglesIndicesBuffer();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::initQuadsIndicesBuffer()
{
    const Inherit::VecQuad& quads = this->getQuads();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, long(quads.size()*sizeof(quads[0])), nullptr, GL_DYNAMIC_DRAW);

    updateQuadsIndicesBuffer();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateVertexBuffer()
{

    const VecCoord& vertices = this->getVertices();
    const VecCoord& vnormals = this->getVnormals();
    const VecTexCoord& vtexcoords= this->getVtexcoords();
    const VecCoord& vtangents= this->getVtangents();
    const VecCoord& vbitangents= this->getVbitangents();
    bool hasTangents = vtangents.size() && vbitangents.size();

    size_t positionsBufferSize, normalsBufferSize;
    size_t textureCoordsBufferSize = 0, tangentsBufferSize = 0, bitangentsBufferSize = 0;

    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));
    normalsBufferSize = (vnormals.size()*sizeof(vnormals[0]));
    const void* positionBuffer = vertices.getData();
    const void* normalBuffer = vnormals.getData();


    verticesTmpBuffer.resize( vertices.size() );
    normalsTmpBuffer.resize( vnormals.size() );

    copyVector(vertices, verticesTmpBuffer);
    copyVector(vnormals, normalsTmpBuffer);

    positionsBufferSize = (vertices.size()*sizeof(Vec3f));
    normalsBufferSize = (vnormals.size()*sizeof(Vec3f));
    positionBuffer = verticesTmpBuffer.data();
    normalBuffer = normalsTmpBuffer.data();

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

    //Texture coords
    if(tex || putOnlyTexCoords.getValue() ||!textures.empty())
    {
        glBufferSubData(GL_ARRAY_BUFFER,
                        positionsBufferSize + normalsBufferSize,
                        textureCoordsBufferSize,
                        getData(vtexcoords));

        if (hasTangents)
        {
            glBufferSubData(GL_ARRAY_BUFFER,
                            positionsBufferSize + normalsBufferSize + textureCoordsBufferSize,
                            tangentsBufferSize,
                            vtangents.getData());

            glBufferSubData(GL_ARRAY_BUFFER,
                            positionsBufferSize + normalsBufferSize + textureCoordsBufferSize + tangentsBufferSize,
                            bitangentsBufferSize,
                            vbitangents.getData());
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void OglModel::updateEdgesIndicesBuffer()
{
    const ResizableExtVector<Edge>& edges = this->getEdges();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboEdges);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, long(edges.size()*sizeof(edges[0])), &edges[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateTrianglesIndicesBuffer()
{
    const ResizableExtVector<Triangle>& triangles = this->getTriangles();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTriangles);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, long(triangles.size()*sizeof(triangles[0])), &triangles[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void OglModel::updateQuadsIndicesBuffer()
{
    const ResizableExtVector<Quad>& quads = this->getQuads();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboQuads);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, long(quads.size()*sizeof(quads[0])), &quads[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
void OglModel::updateBuffers()
{
    const Inherit::VecEdge& edges = this->getEdges();
    const Inherit::VecTriangle& triangles = this->getTriangles();
    const Inherit::VecQuad& quads = this->getQuads();
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


} // namespace visualmodel

} // namespace component

} // namespace sofa

