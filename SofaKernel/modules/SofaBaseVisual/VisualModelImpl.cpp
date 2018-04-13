/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseVisual/VisualModelImpl.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/behavior/MechanicalState.h>

#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <SofaBaseTopology/QuadSetTopologyModifier.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>
#include <SofaBaseTopology/HexahedronSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseTopology/TopologyData.inl>

#include <SofaBaseTopology/SparseGridTopology.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/io/MeshOBJ.h>
#include <sofa/helper/io/MeshSTL.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/accessor.h>
#include <sstream>
#include <map>
#include <memory>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;
using namespace sofa::core::topology;
using namespace sofa::core::loader;
using helper::vector;

void VisualModelImpl::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::visual::VisualModel::parse(arg);

    VisualModelImpl* obj = this;

    if (arg->getAttribute("normals")!=NULL)
        obj->setUseNormals(arg->getAttributeAsInt("normals", 1)!=0);

    if (arg->getAttribute("castshadow")!=NULL)
        obj->setCastShadow(arg->getAttributeAsInt("castshadow", 1)!=0);

    if (arg->getAttribute("flip")!=NULL)
        obj->flipFaces();

    if (arg->getAttribute("color"))
        obj->setColor(arg->getAttribute("color"));

    if (arg->getAttribute("su")!=NULL || arg->getAttribute("sv")!=NULL)
        m_scaleTex = TexCoord(arg->getAttributeAsFloat("su",1.0),
                              arg->getAttributeAsFloat("sv",1.0));

    if (arg->getAttribute("du")!=NULL || arg->getAttribute("dv")!=NULL)
        m_translationTex = TexCoord(arg->getAttributeAsFloat("du",0.0),
                                    arg->getAttributeAsFloat("dv",0.0));

    if (arg->getAttribute("rx")!=NULL || arg->getAttribute("ry")!=NULL || arg->getAttribute("rz")!=NULL)
        m_rotation.setValue(Vec3Real((Real)arg->getAttributeAsFloat("rx",0.0),
                                     (Real)arg->getAttributeAsFloat("ry",0.0),
                                     (Real)arg->getAttributeAsFloat("rz",0.0)));

    if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
        m_translation.setValue(Vec3Real((Real)arg->getAttributeAsFloat("dx",0.0),
                                        (Real)arg->getAttributeAsFloat("dy",0.0),
                                        (Real)arg->getAttributeAsFloat("dz",0.0)));

    if (arg->getAttribute("scale")!=NULL)
    {
        m_scale.setValue(Vec3Real((Real)arg->getAttributeAsFloat("scale",1.0),
                                  (Real)arg->getAttributeAsFloat("scale",1.0),
                                  (Real)arg->getAttributeAsFloat("scale",1.0)));
    }
    else if (arg->getAttribute("sx")!=NULL || arg->getAttribute("sy")!=NULL || arg->getAttribute("sz")!=NULL)
    {
        m_scale.setValue(Vec3Real((Real)arg->getAttributeAsFloat("sx",1.0),
                                  (Real)arg->getAttributeAsFloat("sy",1.0),
                                  (Real)arg->getAttributeAsFloat("sz",1.0)));
    }
}

SOFA_DECL_CLASS(VisualModelImpl)

int VisualModelImplClass = core::RegisterObject("Generic visual model. If a viewer is active it will replace the VisualModel alias, otherwise nothing will be displayed.")
        .add< VisualModelImpl >()
        .addAlias("VisualModel")
        ;

VisualModelImpl::VisualModelImpl() //const std::string &name, std::string filename, std::string loader, std::string textureName)
    :  useTopology(false)
    , lastMeshRev(-1)
    , castShadow(true)
    , m_initRestPositions(initData  (&m_initRestPositions, false, "initRestPositions", "True if rest positions must be initialized with initial positions"))
    , m_useNormals		(initData	(&m_useNormals, true, "useNormals", "True if normal smoothing groups should be read from file"))
    , m_updateNormals   (initData   (&m_updateNormals, true, "updateNormals", "True if normals should be updated at each iteration"))
    , m_computeTangents (initData   (&m_computeTangents, false, "computeTangents", "True if tangents should be computed at startup"))
    , m_updateTangents  (initData   (&m_updateTangents, true, "updateTangents", "True if tangents should be updated at each iteration"))
    , m_handleDynamicTopology (initData   (&m_handleDynamicTopology, true, "handleDynamicTopology", "True if topological changes should be handled"))
    , m_fixMergedUVSeams (initData   (&m_fixMergedUVSeams, true, "fixMergedUVSeams", "True if UV seams should be handled even when duplicate UVs are merged"))
    , m_keepLines (initData   (&m_keepLines, false, "keepLines", "keep and draw lines (false by default)"))
    , m_vertices2       (initData   (&m_vertices2, "vertices", "vertices of the model (only if vertices have multiple normals/texcoords, otherwise positions are used)"))
    , m_vtexcoords      (initData   (&m_vtexcoords, "texcoords", "coordinates of the texture"))
    , m_vtangents       (initData   (&m_vtangents, "tangents", "tangents for normal mapping"))
    , m_vbitangents     (initData   (&m_vbitangents, "bitangents", "tangents for normal mapping"))
    , m_edges           (initData   (&m_edges, "edges", "edges of the model"))
    , m_triangles       (initData   (&m_triangles, "triangles", "triangles of the model"))
    , m_quads           (initData   (&m_quads, "quads", "quads of the model"))
    , m_vertPosIdx      (initData   (&m_vertPosIdx, "vertPosIdx", "If vertices have multiple normals/texcoords stores vertices position indices"))
    , m_vertNormIdx     (initData   (&m_vertNormIdx, "vertNormIdx", "If vertices have multiple normals/texcoords stores vertices normal indices"))
    , fileMesh          (initData   (&fileMesh, "fileMesh"," Path to the model"))
    , texturename       (initData   (&texturename, "texturename", "Name of the Texture"))
    , m_translation     (initData   (&m_translation, Vec3Real(), "translation", "Initial Translation of the object"))
    , m_rotation        (initData   (&m_rotation, Vec3Real(), "rotation", "Initial Rotation of the object"))
    , m_scale           (initData   (&m_scale, Vec3Real(1.0,1.0,1.0), "scale3d", "Initial Scale of the object"))
    , m_scaleTex        (initData   (&m_scaleTex, TexCoord(1.0,1.0), "scaleTex", "Scale of the texture"))
    , m_translationTex  (initData   (&m_translationTex, TexCoord(0.0,0.0), "translationTex", "Translation of the texture"))
    , material			(initData	(&material, "material", "Material")) // tex(NULL)
    , putOnlyTexCoords	(initData	(&putOnlyTexCoords, (bool) false, "putOnlyTexCoords", "Give Texture Coordinates without the texture binding"))
    , srgbTexturing		(initData	(&srgbTexturing, (bool) false, "srgbTexturing", "When sRGB rendering is enabled, is the texture in sRGB colorspace?"))
    , materials			(initData	(&materials, "materials", "List of materials"))
    , groups			(initData	(&groups, "groups", "Groups of triangles and quads using a given material"))
    , xformsModified(false)
{
    m_topology = 0;

    //material.setDisplayed(false);
    addAlias(&fileMesh, "filename");

    m_vertices2     .setGroup("Vector");
    m_vnormals      .setGroup("Vector");
    m_vtexcoords    .setGroup("Vector");
    m_vtangents     .setGroup("Vector");
    m_vbitangents   .setGroup("Vector");
    m_edges         .setGroup("Vector");
    m_triangles     .setGroup("Vector");
    m_quads         .setGroup("Vector");

    m_translation   .setGroup("Transformation");
    m_rotation      .setGroup("Transformation");
    m_scale         .setGroup("Transformation");

    m_edges.setAutoLink(false); // disable linking of edges by default

    // add one identity matrix
    xforms.resize(1);
}

VisualModelImpl::~VisualModelImpl()
{
}

bool VisualModelImpl::hasTransparent()
{
    const Material& material = this->material.getValue();
    helper::ReadAccessor< Data< helper::vector<FaceGroup> > > groups = this->groups;
    helper::ReadAccessor< Data< helper::vector<Material> > > materials = this->materials;
    if (groups.empty())
        return (material.useDiffuse && material.diffuse[3] < 1.0);
    else
    {
        for (unsigned int i = 0; i < groups.size(); ++i)
        {
            const Material& m = (groups[i].materialId == -1) ? material : materials[groups[i].materialId];
            if (m.useDiffuse && m.diffuse[3] < 1.0)
                return true;
        }
    }
    return false;
}

bool VisualModelImpl::hasOpaque()
{
    const Material& material = this->material.getValue();
    helper::ReadAccessor< Data< helper::vector<FaceGroup> > > groups = this->groups;
    helper::ReadAccessor< Data< helper::vector<Material> > > materials = this->materials;
    if (groups.empty())
        return !(material.useDiffuse && material.diffuse[3] < 1.0);
    else
    {
        for (unsigned int i = 0; i < groups.size(); ++i)
        {
            const Material& m = (groups[i].materialId == -1) ? material : materials[groups[i].materialId];
            if (!(m.useDiffuse && m.diffuse[3] < 1.0))
                return true;
        }
    }
    return false;
}

void VisualModelImpl::drawVisual(const core::visual::VisualParams* vparams)
{
    //Update external buffers (like VBO) if the mesh change AFTER doing the updateVisual() process
    if(m_vertices2.isDirty())
    {
        updateBuffers();
    }

    if (hasOpaque())
        internalDraw(vparams,false);
}

void VisualModelImpl::drawTransparent(const core::visual::VisualParams* vparams)
{
    if (hasTransparent())
        internalDraw(vparams,true);
}

void VisualModelImpl::drawShadow(const core::visual::VisualParams* vparams)
{
    if (hasOpaque() && getCastShadow())
        internalDraw(vparams, false);
}

void VisualModelImpl::setMesh(helper::io::Mesh &objLoader, bool tex)
{
    const vector< vector< vector<int> > > &facetsImport = objLoader.getFacets();
    const vector< Vector3 > &verticesImport = objLoader.getVertices();
    const vector< Vector3 > &normalsImport = objLoader.getNormals();
    const vector< Vector3 > &texCoordsImport = objLoader.getTexCoords();

    const Material &materialImport = objLoader.getMaterial();

    if (!material.isSet() && materialImport.activated)
    {
        Material M;
        M = materialImport;
        material.setValue(M);
    }

    if (!objLoader.getGroups().empty())
    {
        // Get informations about the multiple materials
        helper::WriteAccessor< Data< helper::vector<Material> > > materials = this->materials;
        helper::WriteAccessor< Data< helper::vector<FaceGroup> > > groups = this->groups;
        materials.resize(objLoader.getMaterials().size());
        for (unsigned i=0; i<materials.size(); ++i)
            materials[i] = objLoader.getMaterials()[i];

        // compute the edge / triangle / quad index corresponding to each facet
        // convert the groups info
        enum { NBE = 0, NBT = 1, NBQ = 2 };
        helper::fixed_array<int, 3> nbf(0,0,0);
        helper::vector< helper::fixed_array<int, 3> > facet2tq;
        facet2tq.resize(facetsImport.size()+1);
        for (unsigned int i = 0; i < facetsImport.size(); i++)
        {
            facet2tq[i] = nbf;
            const vector<vector <int> >& vertNormTexIndex = facetsImport[i];
            const vector<int>& verts = vertNormTexIndex[0];
            if (verts.size() < 2)
                ; // ignore points
            else if (verts.size() == 2)
                nbf[NBE] += 1;
            else if (verts.size() == 4)
                nbf[NBQ] += 1;
            else
                nbf[NBT] += verts.size()-2;
        }
        facet2tq[facetsImport.size()] = nbf;
        groups.resize(objLoader.getGroups().size());
        for (unsigned int ig = 0; ig < groups.size(); ig++)
        {
            const PrimitiveGroup& g0 = objLoader.getGroups()[ig];
            FaceGroup& g = groups[ig];
            if (g0.materialName.empty()) g.materialName = "defaultMaterial";
            else                         g.materialName = g0.materialName;
            if (g0.groupName.empty())    g.groupName = "defaultGroup";
            else                         g.groupName = g0.groupName;
            g.materialId = g0.materialId;
            g.edge0 = facet2tq[g0.p0][NBE];
            g.nbe = facet2tq[g0.p0+g0.nbp][NBE] - g.edge0;
            g.tri0 = facet2tq[g0.p0][NBT];
            g.nbt = facet2tq[g0.p0+g0.nbp][NBT] - g.tri0;
            g.quad0 = facet2tq[g0.p0][NBQ];
            g.nbq = facet2tq[g0.p0+g0.nbp][NBQ] - g.quad0;
            if (g.materialId == -1 && !g0.materialName.empty())
                msg_info() << "face group " << ig << " name " << g0.materialName << " uses missing material " << g0.materialName << "   ";
        }
    }

    int nbVIn = verticesImport.size();
    // First we compute for each point how many pair of normal/texcoord indices are used
    // The map store the final index of each combinaison
    vector< std::map< std::pair<int,int>, int > > vertTexNormMap;
    vertTexNormMap.resize(nbVIn);
    for (unsigned int i = 0; i < facetsImport.size(); i++)
    {
        const vector<vector <int> >& vertNormTexIndex = facetsImport[i];
        if (vertNormTexIndex[0].size() < 3 && !m_keepLines.getValue() ) continue; // ignore lines
        const vector<int>& verts = vertNormTexIndex[0];
        const vector<int>& texs = vertNormTexIndex[1];
        const vector<int>& norms = vertNormTexIndex[2];
        for (unsigned int j = 0; j < verts.size(); j++)
        {
            vertTexNormMap[verts[j]][std::make_pair((tex ? texs[j] : -1), (m_useNormals.getValue() ? norms[j] : 0))] = 0;
        }
    }

    // Then we can compute how many vertices are created
    int nbVOut = 0;
    bool vsplit = false;
    for (int i = 0; i < nbVIn; i++)
    {
        int s = vertTexNormMap[i].size();
        nbVOut += s;
    }

    msg_info() << nbVIn << " input positions, " << nbVOut << " final vertices.   ";

    if (nbVIn != nbVOut)
        vsplit = true;

    // Then we can create the final arrays
    VecCoord& restPositions = *(m_restPositions.beginEdit());
    VecCoord& positions = *(m_positions.beginEdit());
    VecCoord& vertices2 = *(m_vertices2.beginEdit());
    ResizableExtVector<Deriv>& vnormals = *(m_vnormals.beginEdit());
    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    ResizableExtVector<int>& vertPosIdx = (*m_vertPosIdx.beginEdit());
    ResizableExtVector<int>& vertNormIdx = (*m_vertNormIdx.beginEdit());;

    positions.resize(nbVIn);

    if (m_initRestPositions.getValue())
        restPositions.resize(nbVIn);

    if (vsplit)
    {
        vertices2.resize(nbVOut);
        if( m_useNormals.getValue() ) vnormals.resize(nbVOut);
        vtexcoords.resize(nbVOut);
        vertPosIdx.resize(nbVOut);
        vertNormIdx.resize(nbVOut);
    }
    else
    {
        //vertices2.resize(nbVIn);
        if( m_useNormals.getValue() ) vnormals.resize(nbVIn);
        vtexcoords.resize(nbVIn);
    }

    int nbNOut = 0; /// Number of different normals
    for (int i = 0, j = 0; i < nbVIn; i++)
    {
        positions[i] = verticesImport[i];

        if (m_initRestPositions.getValue())
            restPositions[i] = verticesImport[i];

        std::map<int, int> normMap;
        for (std::map<std::pair<int, int>, int>::iterator it = vertTexNormMap[i].begin();
             it != vertTexNormMap[i].end(); ++it)
        {
            int t = it->first.first;
            int n = it->first.second;
            if ( m_useNormals.getValue() && (unsigned)n < normalsImport.size())
                vnormals[j] = normalsImport[n];
            if ((unsigned)t < texCoordsImport.size())
                vtexcoords[j] = texCoordsImport[t];

            if (vsplit)
            {
                vertices2[j] = verticesImport[i];
                vertPosIdx[j] = i;
                if (normMap.count(n))
                    vertNormIdx[j] = normMap[n];
                else
                {
                    vertNormIdx[j] = nbNOut;
                    normMap[n] = nbNOut++;
                }
            }
            it->second = j++;
        }
    }

//    if (!vsplit)
//        nbNOut = nbVOut;
//    else if (nbNOut == nbVOut)
//        vertNormIdx.resize(0);
    if( vsplit && nbNOut == nbVOut )
        vertNormIdx.resize(0);


    m_vertices2.endEdit();
    m_vnormals.endEdit();
    m_vtexcoords.endEdit();
    m_positions.endEdit();
    m_restPositions.endEdit();
    m_vertPosIdx.endEdit();
    m_vertNormIdx.endEdit();

    // Then we create the triangles and quads
    ResizableExtVector< Edge >& edges = *(m_edges.beginEdit());
    ResizableExtVector< Triangle >& triangles = *(m_triangles.beginEdit());
    ResizableExtVector< Quad >& quads = *(m_quads.beginEdit());

    for (unsigned int i = 0; i < facetsImport.size(); i++)
    {
        const vector<vector <int> >& vertNormTexIndex = facetsImport[i];
        const vector<int>& verts = vertNormTexIndex[0];
        const vector<int>& texs = vertNormTexIndex[1];
        const vector<int>& norms = vertNormTexIndex[2];
        vector<int> idxs;
        idxs.resize(verts.size());
        for (unsigned int j = 0; j < verts.size(); j++)
        {
            idxs[j] = vertTexNormMap[verts[j]][std::make_pair((tex?texs[j]:-1), (m_useNormals.getValue() ? norms[j] : 0))];
            if ((unsigned)idxs[j] >= (unsigned)nbVOut)
            {
                msg_error() << this->getPathName()<<" index "<<idxs[j]<<" out of range";
                idxs[j] = 0;
            }
        }

        if (verts.size() == 2)
        {
            edges.push_back(Edge(idxs[0],idxs[1]));
        }
        else if (verts.size() == 4)
        {
            quads.push_back(Quad(idxs[0],idxs[1],idxs[2],idxs[3]));
        }
        else
        {
            for (unsigned int j = 2; j < verts.size(); j++)
            {
                triangles.push_back(Triangle(idxs[0],idxs[j-1],idxs[j]));
            }
        }
    }

    m_edges.endEdit();
    m_triangles.endEdit();
    m_quads.endEdit();

    computeNormals();
    computeTangents();

}

bool VisualModelImpl::load(const std::string& filename, const std::string& loader, const std::string& textureName)
{
    using sofa::helper::io::Mesh;

    //      bool tex = !textureName.empty() || putOnlyTexCoords.getValue();
    if (!textureName.empty())
    {
        std::string textureFilename(textureName);
        if (sofa::helper::system::DataRepository.findFile(textureFilename))
        {
            msg_info() << "loading file " << textureName;
            bool textureLoaded = loadTexture(textureName);
            if(!textureLoaded)
            {
                msg_error()<<"Texture "<<textureName<<" cannot be loaded";
            }
        }
        else
        {
            msg_error() << "Texture \"" << textureName << "\" not found";
        }
    }

    // Make sure all Data are up-to-date
    m_vertices2.updateIfDirty();
    m_vnormals.updateIfDirty();
    m_vtexcoords.updateIfDirty();
    m_vtangents.updateIfDirty();
    m_vbitangents.updateIfDirty();
    m_edges.updateIfDirty();
    m_triangles.updateIfDirty();
    m_quads.updateIfDirty();

    if (!filename.empty() && (m_positions.getValue()).size() == 0 && (m_vertices2.getValue()).size() == 0)
    {
        std::string meshFilename(filename);
        if (sofa::helper::system::DataRepository.findFile(meshFilename))
        {
            //name = filename;
            std::unique_ptr<Mesh> objLoader;
            if (loader.empty())
            {
                objLoader.reset(Mesh::Create(filename));
            }
            else
            {
                objLoader.reset(Mesh::Create(loader, filename));
            }

            if (objLoader.get() == 0)
            {
                return false;
            }
            else
            {
                //if( MeshSTL *Loader = dynamic_cast< MeshSTL *>(objLoader.get()) )
                if(objLoader.get()->loaderType == "stl" || objLoader.get()->loaderType == "vtu")
                {
                    setMesh(*objLoader, false);
                }
                else
                {
                    //Modified: previously, the texture coordinates were not loaded correctly if no texture name was specified.
                    //setMesh(*objLoader,tex);
                    setMesh(*objLoader, true);
                }
            }

            if(textureName.empty())
            {
                //we check how many textures are linked with a material (only if a texture name is not defined in the scn file)
                bool isATextureLinked = false;
                for (unsigned int i = 0 ; i < this->materials.getValue().size() ; i++)
                {
                    //we count only the texture with an activated material
                    if (this->materials.getValue()[i].useTexture && this->materials.getValue()[i].activated)
                    {
                        isATextureLinked=true;
                        break;
                    }
                }
                if (isATextureLinked)
                {
                    loadTextures();
                }
            }
        }
        else
        {
            msg_error() << "Mesh \"" << filename << "\" not found";
        }
    }
    else
    {
        if ((m_positions.getValue()).size() == 0 && (m_vertices2.getValue()).size() == 0)
        {
            msg_info() << "will use Topology.";
            useTopology = true;
        }

        modified = true;
    }

    if (!xformsModified)
    {
        // add one identity matrix
        xforms.resize(1);
    }
    applyUVTransformation();
    return true;
}

void VisualModelImpl::applyUVTransformation()
{
    applyUVScale(m_scaleTex.getValue()[0], m_scaleTex.getValue()[1]);
    applyUVTranslation(m_translationTex.getValue()[0], m_translationTex.getValue()[1]);
    m_scaleTex.setValue(TexCoord(1,1));
    m_translationTex.setValue(TexCoord(0,0));
}

void VisualModelImpl::applyTranslation(const SReal dx, const SReal dy, const SReal dz)
{
    Coord d((Real)dx,(Real)dy,(Real)dz);

    Data< VecCoord >* d_x = this->write(core::VecCoordId::position());
    VecCoord &x = *d_x->beginEdit();

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i] += d;
    }

    d_x->endEdit();

    if(m_initRestPositions.getValue())
    {
        VecCoord& restPositions = *(m_restPositions.beginEdit());

        for (unsigned int i = 0; i < restPositions.size(); i++)
        {
            restPositions[i] += d;
        }

        m_restPositions.endEdit();
    }


    updateVisual();
}

void VisualModelImpl::applyRotation(const SReal rx, const SReal ry, const SReal rz)
{
    Quaternion q = helper::Quater<SReal>::createQuaterFromEuler( Vec<3,SReal>(rx,ry,rz)*M_PI/180.0);
    applyRotation(q);
}

void VisualModelImpl::applyRotation(const Quat q)
{
    Data< VecCoord >* d_x = this->write(core::VecCoordId::position());
    VecCoord &x = *d_x->beginEdit();

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i] = q.rotate(x[i]);
    }

    d_x->endEdit();

    if(m_initRestPositions.getValue())
    {
        VecCoord& restPositions = *(m_restPositions.beginEdit());

        for (unsigned int i = 0; i < restPositions.size(); i++)
        {
            restPositions[i] = q.rotate(restPositions[i]);
        }

        m_restPositions.endEdit();
    }

    updateVisual();
}

void VisualModelImpl::applyScale(const SReal sx, const SReal sy, const SReal sz)
{
    Data< VecCoord >* d_x = this->write(core::VecCoordId::position());
    VecCoord &x = *d_x->beginEdit();

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i][0] *= (Real)sx;
        x[i][1] *= (Real)sy;
        x[i][2] *= (Real)sz;
    }

    d_x->endEdit();

    if(m_initRestPositions.getValue())
    {
        VecCoord& restPositions = *(m_restPositions.beginEdit());

        for (unsigned int i = 0; i < restPositions.size(); i++)
        {
            restPositions[i][0] *= (Real)sx;
            restPositions[i][1] *= (Real)sy;
            restPositions[i][2] *= (Real)sz;
        }

        m_restPositions.endEdit();
    }

    updateVisual();
}

void VisualModelImpl::applyUVTranslation(const Real dU, const Real dV)
{
    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    for (unsigned int i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] += dU;
        vtexcoords[i][1] += dV;
    }
    m_vtexcoords.endEdit();
}

void VisualModelImpl::applyUVScale(const Real scaleU, const Real scaleV)
{
    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    for (unsigned int i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] *= scaleU;
        vtexcoords[i][1] *= scaleV;
    }
    m_vtexcoords.endEdit();
}


template<class VecCoord>
class VisualModelPointHandler : public sofa::component::topology::TopologyDataHandler<sofa::core::topology::Point,VecCoord >
{
public:
    typedef typename VecCoord::value_type Coord;
    typedef typename Coord::value_type Real;
    VisualModelPointHandler(VisualModelImpl* obj, sofa::component::topology::PointData<VecCoord>* data, int algo)
        : sofa::component::topology::TopologyDataHandler<sofa::core::topology::Point, VecCoord >(data), obj(obj), algo(algo) {}

    void applyCreateFunction(unsigned int /*pointIndex*/, Coord& dest, const sofa::core::topology::Point &,
                             const sofa::helper::vector< unsigned int > &ancestors,
                             const sofa::helper::vector< double > &coefs)
    {
        const VecCoord& x = this->m_topologyData->getValue();
        if (!ancestors.empty())
        {
            if (algo == 1 && ancestors.size() > 1) //fixMergedUVSeams
            {
                Coord c0 = x[ancestors[0]];
                dest = c0*coefs[0];
                for (unsigned int i=1; i<ancestors.size(); ++i)
                {
                    Coord ci = x[ancestors[i]];
                    for (unsigned int j=0; j<ci.size(); ++j)
                        ci[j] += helper::rnear(c0[j]-ci[j]);
                    dest += ci*coefs[i];
                }
            }
            else
            {
                dest = x[ancestors[0]]*coefs[0];
                for (unsigned int i=1; i<ancestors.size(); ++i)
                    dest += x[ancestors[i]]*coefs[i];
            }
        }
        // BUGFIX: remove link to the Data as it is now specific to this instance
        this->m_topologyData->setParent(NULL);
    }

    void applyDestroyFunction(unsigned int, Coord& )
    {
    }

protected:
    VisualModelImpl* obj;
    int algo;
};

template<class VecType>
void VisualModelImpl::addTopoHandler(topology::PointData<VecType>* data, int algo)
{
    data->createTopologicalEngine(m_topology, new VisualModelPointHandler<VecType>(this, data, algo), true);
    data->registerTopologicalData();
}

void VisualModelImpl::init()
{
    load(fileMesh.getFullPath(), "", texturename.getFullPath());
    m_topology = getContext()->getMeshTopology();

    if (m_topology == 0 || (m_positions.getValue().size()!=0 && m_positions.getValue().size() != (unsigned int)m_topology->getNbPoints()))
    {
        // Fixes bug when neither an .obj file nor a topology is present in the VisualModel Node.
        // Thus nothing will be displayed.
        useTopology = false;
    }
    else
    {
        msg_info() << "Use topology " << m_topology->getName();
        // add the functions to handle topology changes.
        if (m_handleDynamicTopology.getValue())
        {
            //addTopoHandler(&m_positions);
            //addTopoHandler(&m_restPositions);
            //addTopoHandler(&m_vnormals);
            addTopoHandler(&m_vtexcoords,(m_fixMergedUVSeams.getValue()?1:0));
            //addTopoHandler(&m_vtangents);
            //addTopoHandler(&m_vbitangents);
        }
    }

    m_vertices2.beginEdit();
    m_vnormals.beginEdit();
    m_vtexcoords.beginEdit();
    m_vtangents.beginEdit();
    m_vbitangents.beginEdit();
    m_triangles.beginEdit();
    m_quads.beginEdit();

    applyScale(m_scale.getValue()[0], m_scale.getValue()[1], m_scale.getValue()[2]);
    applyRotation(m_rotation.getValue()[0], m_rotation.getValue()[1], m_rotation.getValue()[2]);
    applyTranslation(m_translation.getValue()[0], m_translation.getValue()[1], m_translation.getValue()[2]);


    m_translation.setValue(Vec3Real());
    m_rotation.setValue(Vec3Real());
    m_scale.setValue(Vec3Real(1,1,1));

    VisualModel::init();
    updateVisual();
}

void VisualModelImpl::computeNormals()
{
    const VecCoord& vertices = getVertices();
    //const VecCoord& vertices = m_vertices2.getValue();
    if (vertices.empty() || (!m_updateNormals.getValue() && (m_vnormals.getValue()).size() == (vertices).size())) return;

    const ResizableExtVector<Triangle>& triangles = m_triangles.getValue();
    const ResizableExtVector<Quad>& quads = m_quads.getValue();
    const ResizableExtVector<int> &vertNormIdx = m_vertNormIdx.getValue();

    if (vertNormIdx.empty())
    {
        int nbn = (vertices).size();

        ResizableExtVector<Deriv>& normals = *(m_vnormals.beginEdit());

        normals.resize(nbn);
        for (int i = 0; i < nbn; i++)
            normals[i].clear();

        for (unsigned int i = 0; i < triangles.size(); i++)
        {
            const Coord& v1 = vertices[triangles[i][0]];
            const Coord& v2 = vertices[triangles[i][1]];
            const Coord& v3 = vertices[triangles[i][2]];
            Coord n = cross(v2-v1, v3-v1);

            normals[triangles[i][0]] += n;
            normals[triangles[i][1]] += n;
            normals[triangles[i][2]] += n;
        }

        for (unsigned int i = 0; i < quads.size(); i++)
        {
            const Coord & v1 = vertices[quads[i][0]];
            const Coord & v2 = vertices[quads[i][1]];
            const Coord & v3 = vertices[quads[i][2]];
            const Coord & v4 = vertices[quads[i][3]];
            Coord n1 = cross(v2-v1, v4-v1);
            Coord n2 = cross(v3-v2, v1-v2);
            Coord n3 = cross(v4-v3, v2-v3);
            Coord n4 = cross(v1-v4, v3-v4);

            normals[quads[i][0]] += n1;
            normals[quads[i][1]] += n2;
            normals[quads[i][2]] += n3;
            normals[quads[i][3]] += n4;
        }

        for (unsigned int i = 0; i < normals.size(); i++)
            normals[i].normalize();

        m_vnormals.endEdit();
    }
    else
    {
        vector<Coord> normals;
        int nbn = 0;
        for (unsigned int i = 0; i < vertNormIdx.size(); i++)
        {
            if (vertNormIdx[i] >= nbn)
                nbn = vertNormIdx[i]+1;
        }

        normals.resize(nbn);
        for (int i = 0; i < nbn; i++)
            normals[i].clear();

        for (unsigned int i = 0; i < triangles.size() ; i++)
        {
            const Coord & v1 = vertices[triangles[i][0]];
            const Coord & v2 = vertices[triangles[i][1]];
            const Coord & v3 = vertices[triangles[i][2]];
            Coord n = cross(v2-v1, v3-v1);

            normals[vertNormIdx[triangles[i][0]]] += n;
            normals[vertNormIdx[triangles[i][1]]] += n;
            normals[vertNormIdx[triangles[i][2]]] += n;
        }

        for (unsigned int i = 0; i < quads.size() ; i++)
        {
            const Coord & v1 = vertices[quads[i][0]];
            const Coord & v2 = vertices[quads[i][1]];
            const Coord & v3 = vertices[quads[i][2]];
            const Coord & v4 = vertices[quads[i][3]];
            Coord n1 = cross(v2-v1, v4-v1);
            Coord n2 = cross(v3-v2, v1-v2);
            Coord n3 = cross(v4-v3, v2-v3);
            Coord n4 = cross(v1-v4, v3-v4);

            normals[vertNormIdx[quads[i][0]]] += n1;
            normals[vertNormIdx[quads[i][1]]] += n2;
            normals[vertNormIdx[quads[i][2]]] += n3;
            normals[vertNormIdx[quads[i][3]]] += n4;
        }

        for (unsigned int i = 0; i < normals.size(); i++)
        {
            normals[i].normalize();
        }

        ResizableExtVector<Deriv>& vnormals = *(m_vnormals.beginEdit());
        vnormals.resize(vertices.size());
        for (unsigned int i = 0; i < vertices.size(); i++)
        {
            vnormals[i] = normals[vertNormIdx[i]];
        }
        m_vnormals.endEdit();
    }
}

VisualModelImpl::Coord VisualModelImpl::computeTangent(const Coord &v1, const Coord &v2, const Coord &v3,
                                                       const TexCoord &t1, const TexCoord &t2, const TexCoord &t3)
{
    Coord v = (v2 - v1) * (t3.y() - t1.y()) + (v3 - v1) * (t1.y() - t2.y());
    v.normalize();
    return v;
}

VisualModelImpl::Coord VisualModelImpl::computeBitangent(const Coord &v1, const Coord &v2, const Coord &v3,
                                                         const TexCoord &t1, const TexCoord &t2, const TexCoord &t3)
{
    Coord v = (v2 - v1) * (t3.x() - t1.x()) + (v3 - v1) * (t1.x() - t2.x());
    v.normalize();
    return v;
}

void VisualModelImpl::computeTangents()
{
    if (!m_computeTangents.getValue() || !m_vtexcoords.getValue().size()) return;

    const ResizableExtVector<Triangle>& triangles = m_triangles.getValue();
    const ResizableExtVector<Quad>& quads = m_quads.getValue();
    const VecCoord& vertices = getVertices();
    const VecTexCoord& texcoords = m_vtexcoords.getValue();
    VecCoord& normals = *(m_vnormals.beginEdit());
    VecCoord& tangents = *(m_vtangents.beginEdit());
    VecCoord& bitangents = *(m_vbitangents.beginEdit());

    tangents.resize(vertices.size());
    bitangents.resize(vertices.size());

    for (unsigned i = 0; i < vertices.size(); i++)
    {
        tangents[i].clear();
        bitangents[i].clear();
    }
    const bool fixMergedUVSeams = m_fixMergedUVSeams.getValue();
    for (unsigned int i = 0; i < triangles.size() ; i++)
    {
        const Coord v1 = vertices[triangles[i][0]];
        const Coord v2 = vertices[triangles[i][1]];
        const Coord v3 = vertices[triangles[i][2]];
        TexCoord t1 = texcoords[triangles[i][0]];
        TexCoord t2 = texcoords[triangles[i][1]];
        TexCoord t3 = texcoords[triangles[i][2]];
        if (fixMergedUVSeams)
        {
            for (unsigned int j=0; j<t1.size(); ++j)
            {
                t2[j] += helper::rnear(t1[j]-t2[j]);
                t3[j] += helper::rnear(t1[j]-t3[j]);
            }
        }
        Coord t = computeTangent(v1, v2, v3, t1, t2, t3);
        Coord b = computeBitangent(v1, v2, v3, t1, t2, t3);

        tangents[triangles[i][0]] += t;
        tangents[triangles[i][1]] += t;
        tangents[triangles[i][2]] += t;
        bitangents[triangles[i][0]] += b;
        bitangents[triangles[i][1]] += b;
        bitangents[triangles[i][2]] += b;
    }

    for (unsigned int i = 0; i < quads.size() ; i++)
    {
        const Coord & v1 = vertices[quads[i][0]];
        const Coord & v2 = vertices[quads[i][1]];
        const Coord & v3 = vertices[quads[i][2]];
        const Coord & v4 = vertices[quads[i][3]];
        const TexCoord t1 = texcoords[quads[i][0]];
        const TexCoord t2 = texcoords[quads[i][1]];
        const TexCoord t3 = texcoords[quads[i][2]];
        const TexCoord t4 = texcoords[quads[i][3]];

        // Too many options how to split a quad into two triangles...
        Coord t123 = computeTangent  (v1, v2, v3, t1, t2, t3);
        Coord b123 = computeBitangent(v1, v2, v2, t1, t2, t3);

        Coord t234 = computeTangent  (v2, v3, v4, t2, t3, t4);
        Coord b234 = computeBitangent(v2, v3, v4, t2, t3, t4);

        Coord t341 = computeTangent  (v3, v4, v1, t3, t4, t1);
        Coord b341 = computeBitangent(v3, v4, v1, t3, t4, t1);

        Coord t412 = computeTangent  (v4, v1, v2, t4, t1, t2);
        Coord b412 = computeBitangent(v4, v1, v2, t4, t1, t2);

        tangents  [quads[i][0]] += t123        + t341 + t412;
        bitangents[quads[i][0]] += b123        + b341 + b412;
        tangents  [quads[i][1]] += t123 + t234        + t412;
        bitangents[quads[i][1]] += b123 + b234        + b412;
        tangents  [quads[i][2]] += t123 + t234 + t341;
        bitangents[quads[i][2]] += b123 + b234 + b341;
        tangents  [quads[i][3]] +=        t234 + t341 + t412;
        bitangents[quads[i][3]] +=        b234 + b341 + b412;
    }
    for (unsigned int i = 0; i < vertices.size(); i++)
    {
        Coord n = normals[i];
        Coord& t = tangents[i];
        Coord& b = bitangents[i];

        b = sofa::defaulttype::cross(n, t.normalized());
        t = sofa::defaulttype::cross(b, n);
    }
    m_vtangents.endEdit();
    m_vbitangents.endEdit();
}

void VisualModelImpl::computeBBox(const core::ExecParams* params, bool)
{
    const VecCoord& x = getVertices(); //m_vertices.getValue(params);

    SReal minBBox[3] = {std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max()};
    SReal maxBBox[3] = {-std::numeric_limits<Real>::max(),-std::numeric_limits<Real>::max(),-std::numeric_limits<Real>::max()};
    for (unsigned int i = 0; i < x.size(); i++)
    {
        const Coord& p = x[i];
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<SReal>(minBBox,maxBBox));
}

void VisualModelImpl::flipFaces()
{
    ResizableExtVector<Deriv>& vnormals = *(m_vnormals.beginEdit());
    ResizableExtVector<Edge>& edges = *(m_edges.beginEdit());
    ResizableExtVector<Triangle>& triangles = *(m_triangles.beginEdit());
    ResizableExtVector<Quad>& quads = *(m_quads.beginEdit());

    for (unsigned int i = 0; i < edges.size() ; i++)
    {
        int temp = edges[i][1];
        edges[i][1] = edges[i][0];
        edges[i][0] = temp;
    }

    for (unsigned int i = 0; i < triangles.size() ; i++)
    {
        int temp = triangles[i][1];
        triangles[i][1] = triangles[i][2];
        triangles[i][2] = temp;
    }

    for (unsigned int i = 0; i < quads.size() ; i++)
    {
        int temp = quads[i][1];
        quads[i][1] = quads[i][3];
        quads[i][3] = temp;
    }

    for (unsigned int i = 0; i < vnormals.size(); i++)
    {
        vnormals[i] = -vnormals[i];
    }

    m_vnormals.endEdit();
    m_edges.endEdit();
    m_triangles.endEdit();
    m_quads.endEdit();
}

void VisualModelImpl::setColor(float r, float g, float b, float a)
{
    Material M = material.getValue();
    M.setColor(r,g,b,a);
    material.setValue(M);
}

void VisualModelImpl::setColor(std::string color)
{
    if (color.empty())
        return;

    RGBAColor theColor;
    if( !RGBAColor::read(color, theColor) )
    {
        msg_info(this) << "Unable to decode color '"<< color <<"'." ;
    }
    setColor(theColor.r(),theColor.g(),theColor.b(),theColor.a());
}


void VisualModelImpl::updateVisual()
{
    /*
        static unsigned int last = 0;
        if (m_vtexcoords.getValue().size() != last)
        {
            std::cout << m_vtexcoords.getValue().size() << std::endl;
            last = m_vtexcoords.getValue().size();
        }
    */
    if (modified && (!getVertices().empty() || useTopology))
    {
        if (useTopology)
        {
            /** HD : build also a Ogl description from main Topology. But it needs to be build only once since the topology update
            is taken care of by the handleTopologyChange() routine */

            sofa::core::topology::TopologyModifier* topoMod;
            this->getContext()->get(topoMod);

            if (topoMod)
            {
                useTopology = false; // dynamic topology
                computeMesh();
            }
            else if (topoMod == NULL && (m_topology->getRevision() != lastMeshRev))  // static topology
            {
                computeMesh();
            }
        }
        computePositions();
        updateBuffers();

        computeNormals();
        if (m_updateTangents.getValue())
            computeTangents();
        modified = false;
    }

    m_positions.updateIfDirty();
    m_vertices2.updateIfDirty();
    m_vnormals.updateIfDirty();
    //m_vtexcoords.updateIfDirty();
    m_vtangents.updateIfDirty();
    m_vbitangents.updateIfDirty();
    m_edges.updateIfDirty();
    m_triangles.updateIfDirty();
    m_quads.updateIfDirty();

}


void VisualModelImpl::computePositions()
{
    const ResizableExtVector<int> &vertPosIdx = m_vertPosIdx.getValue();

    if (!vertPosIdx.empty())
    {
        // Need to transfer positions
        VecCoord& vertices = *(m_vertices2.beginEdit());
        const VecCoord& positions = this->m_positions.getValue();

        for (unsigned int i=0 ; i < vertices.size(); ++i)
            vertices[i] = positions[vertPosIdx[i]];

        m_vertices2.endEdit();
    }
}

void VisualModelImpl::computeMesh()
{
    using sofa::component::topology::SparseGridTopology;
    using sofa::core::behavior::BaseMechanicalState;

//	sofa::helper::vector<Coord> bezierControlPointsArray;

    if ((m_positions.getValue()).empty() && (m_vertices2.getValue()).empty())
    {
        VecCoord& vertices = *(m_positions.beginEdit());

        if (m_topology->hasPos())
        {
            if (SparseGridTopology *spTopo = dynamic_cast< SparseGridTopology *>(m_topology))
            {
                sofa::helper::io::Mesh m;
                spTopo->getMesh(m);
                setMesh(m, !texturename.getValue().empty());
                dmsg_info() << " getting marching cube mesh from topology, "
                            << m.getVertices().size() << " points, "
                            << m.getFacets().size()  << " triangles." ;

                useTopology = false; //visual model needs to be created only once at initial time
                return;
            }

            dmsg_info() << " copying " << m_topology->getNbPoints() << " points from topology." ;

            vertices.resize(m_topology->getNbPoints());

            for (unsigned int i=0; i<vertices.size(); i++)
            {
                vertices[i][0] = (Real)m_topology->getPX(i);
                vertices[i][1] = (Real)m_topology->getPY(i);
                vertices[i][2] = (Real)m_topology->getPZ(i);
            }

        }
        else
        {
            BaseMechanicalState* mstate = m_topology->getContext()->getMechanicalState();

            if (mstate)
            {
                dmsg_info() << " copying " << mstate->getSize() << " points from mechanical state" ;

                vertices.resize(mstate->getSize());

                for (unsigned int i=0; i<vertices.size(); i++)
                {
                    vertices[i][0] = (Real)mstate->getPX(i);
                    vertices[i][1] = (Real)mstate->getPY(i);
                    vertices[i][2] = (Real)mstate->getPZ(i);
                }

            }
        }
        m_positions.endEdit();
    }

    lastMeshRev = m_topology->getRevision();

    const vector< Triangle >& inputTriangles = m_topology->getTriangles();


    dmsg_info() << " copying " << inputTriangles.size() << " triangles from topology" ;

    ResizableExtVector< Triangle >& triangles = *(m_triangles.beginEdit());
    triangles.resize(inputTriangles.size());

    for (unsigned int i=0; i<triangles.size(); ++i)
    {
        triangles[i] = inputTriangles[i];
    }
    m_triangles.endEdit();


    const vector< BaseMeshTopology::Quad >& inputQuads = m_topology->getQuads();

    dmsg_info() << " copying " << inputQuads.size()<< " quads from topology." ;

    ResizableExtVector< Quad >& quads = *(m_quads.beginEdit());
    quads.resize(inputQuads.size());

    for (unsigned int i=0; i<quads.size(); ++i)
    {
        quads[i] = inputQuads[i];
    }
    m_quads.endEdit();
}

void VisualModelImpl::handleTopologyChange()
{
    if (!m_topology) return;

    bool debug_mode = false;

    ResizableExtVector<Triangle>& triangles = *(m_triangles.beginEdit());
    ResizableExtVector<Quad>& quads = *(m_quads.beginEdit());
    m_positions.beginEdit();

    std::list<const TopologyChange *>::const_iterator itBegin=m_topology->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=m_topology->endChange();

    while( itBegin != itEnd )
    {
        core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

        switch( changeType )
        {
        case core::topology::ENDING_EVENT:
        {
            updateVisual();
            break;
        }

        case core::topology::TRIANGLESADDED:
        {
            if (!groups.getValue().empty())
            {
                groups.beginEdit()->clear();
                groups.endEdit();
            }

            const sofa::core::topology::TrianglesAdded *ta = static_cast< const sofa::core::topology::TrianglesAdded * >( *itBegin );
            Triangle t;
            const unsigned int nbAddedTriangles = ta->getNbAddedTriangles();
            const unsigned int nbTririangles = triangles.size();
            triangles.resize(nbTririangles + nbAddedTriangles);

            for (unsigned int i = 0; i < nbAddedTriangles; ++i)
            {
                t[0] = (int)(ta->triangleArray[i])[0];
                t[1] = (int)(ta->triangleArray[i])[1];
                t[2] = (int)(ta->triangleArray[i])[2];
                triangles[nbTririangles + i] = t;
            }

            break;
        }

        case core::topology::QUADSADDED:
        {
            if (!groups.getValue().empty())
            {
                groups.beginEdit()->clear();
                groups.endEdit();
            }

            const sofa::core::topology::QuadsAdded *qa = static_cast< const sofa::core::topology::QuadsAdded * >( *itBegin );
            Quad q;
            const unsigned int nbAddedQuads = qa->getNbAddedQuads();
            const unsigned int nbQuaduads = quads.size();
            quads.resize(nbQuaduads + nbAddedQuads);

            for (unsigned int i = 0; i < nbAddedQuads; ++i)
            {
                //q[0] = (int)(qa->getQuad(i))[0];
                //q[1] = (int)(qa->getQuad(i))[1];
                //q[2] = (int)(qa->getQuad(i))[2];
                //q[3] = (int)(qa->getQuad(i))[3];
                //quads[nbQuaduads + i] = q;
                quads[nbQuaduads + i] = (qa->getQuad(i));
            }

            break;
        }

        case core::topology::TRIANGLESREMOVED:
        {
            if (!groups.getValue().empty())
            {
                groups.beginEdit()->clear();
                groups.endEdit();
            }

            unsigned int last;

            last = m_topology->getNbTriangles() - 1;

            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const sofa::core::topology::TrianglesRemoved *>( *itBegin ) )->getArray();

            Triangle tmp;

            for (unsigned int i = 0; i <tab.size(); ++i)
            {
                unsigned int ind_k = tab[i];

                tmp = triangles[ind_k];
                triangles[ind_k] = triangles[last];
                triangles[last] = tmp;

                unsigned int ind_last = triangles.size() - 1;

                if(last != ind_last)
                {
                    tmp = triangles[last];
                    triangles[last] = triangles[ind_last];
                    triangles[ind_last] = tmp;
                }

                triangles.resize( triangles.size() - 1 );

                --last;
            }

            break;
        }

        case core::topology::QUADSREMOVED:
        {
            if (!groups.getValue().empty())
            {
                groups.beginEdit()->clear();
                groups.endEdit();
            }

            unsigned int last;

            last = m_topology->getNbQuads() - 1;

            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const sofa::core::topology::QuadsRemoved *>( *itBegin ) )->getArray();

            Quad tmp;

            for (unsigned int i = 0; i <tab.size(); ++i)
            {
                unsigned int ind_k = tab[i];

                tmp = quads[ind_k];
                quads[ind_k] = quads[last];
                quads[last] = tmp;

                unsigned int ind_last = quads.size() - 1;

                if(last != ind_last)
                {
                    tmp = quads[last];
                    quads[last] = quads[ind_last];
                    quads[ind_last] = tmp;
                }

                quads.resize( quads.size() - 1 );

                --last;
            }

            break;
        }

        case core::topology::POINTSREMOVED:
        {
            if (m_topology->getNbTriangles()>0)
            {
                unsigned int last = m_topology->getNbPoints() -1;

                unsigned int i,j;

                const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::core::topology::PointsRemoved * >( *itBegin ) )->getArray();

                sofa::helper::vector<unsigned int> lastIndexVec;

                for(unsigned int i_init = 0; i_init < tab.size(); ++i_init)
                {
                    lastIndexVec.push_back(last - i_init);
                }

                for ( i = 0; i < tab.size(); ++i)
                {
                    unsigned int i_next = i;
                    bool is_reached = false;

                    while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
                    {
                        i_next += 1 ;
                        is_reached = is_reached || (lastIndexVec[i_next] == tab[i]);
                    }

                    if(is_reached)
                    {
                        lastIndexVec[i_next] = lastIndexVec[i];
                    }

                    const sofa::helper::vector<unsigned int> &shell= m_topology->getTrianglesAroundVertex(lastIndexVec[i]);
                    for (j=0; j<shell.size(); ++j)
                    {
                        unsigned int ind_j = shell[j];

                        if ((unsigned)triangles[ind_j][0]==last)
                            triangles[ind_j][0]=tab[i];
                        else if ((unsigned)triangles[ind_j][1]==last)
                            triangles[ind_j][1]=tab[i];
                        else if ((unsigned)triangles[ind_j][2]==last)
                            triangles[ind_j][2]=tab[i];
                    }

                    if (debug_mode)
                    {
                        for (unsigned int j_loc=0; j_loc<triangles.size(); ++j_loc)
                        {
                            bool is_forgotten = false;
                            if ((unsigned)triangles[j_loc][0]==last)
                            {
                                triangles[j_loc][0]=tab[i];
                                is_forgotten=true;
                            }
                            else
                            {
                                if ((unsigned)triangles[j_loc][1]==last)
                                {
                                    triangles[j_loc][1]=tab[i];
                                    is_forgotten=true;
                                }
                                else
                                {
                                    if ((unsigned)triangles[j_loc][2]==last)
                                    {
                                        triangles[j_loc][2]=tab[i];
                                        is_forgotten=true;
                                    }
                                }
                            }

                            if(is_forgotten)
                            {
                                int ind_forgotten = j_loc;

                                bool is_in_shell = false;
                                for (unsigned int j_glob=0; j_glob<shell.size(); ++j_glob)
                                {
                                    is_in_shell = is_in_shell || ((int)shell[j_glob] == ind_forgotten);
                                }

                                if(!is_in_shell)
                                {
                                    msg_info() << "INFO_print : Vis - triangle is forgotten in SHELL !!! global indices (point, triangle) = ( "  << last << " , " << ind_forgotten  << " )";

                                    if(ind_forgotten<m_topology->getNbTriangles())
                                    {
                                        const core::topology::BaseMeshTopology::Triangle t_forgotten = m_topology->getTriangle(ind_forgotten);
                                        msg_info() << "Vis - last = " << last << msgendl
                                                   << "Vis - lastIndexVec[i] = " << lastIndexVec[i] << msgendl
                                                   << "Vis - tab.size() = " << tab.size() << " , tab = " << tab << msgendl
                                                   << "Vis - t_local rectified = " << triangles[j_loc] << msgendl
                                                   << "Vis - t_global = " << t_forgotten;
                                    }
                                }
                            }
                        }
                    }

                    --last;
                }
            }
            else if (m_topology->getNbQuads()>0)
            {
                unsigned int last = m_topology->getNbPoints() -1;

                unsigned int i,j;

                const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::core::topology::PointsRemoved * >( *itBegin ) )->getArray();

                sofa::helper::vector<unsigned int> lastIndexVec;
                for(unsigned int i_init = 0; i_init < tab.size(); ++i_init)
                {
                    lastIndexVec.push_back(last - i_init);
                }

                for ( i = 0; i < tab.size(); ++i)
                {
                    unsigned int i_next = i;
                    bool is_reached = false;
                    while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
                    {
                        i_next += 1 ;
                        is_reached = is_reached || (lastIndexVec[i_next] == tab[i]);
                    }

                    if(is_reached)
                    {
                        lastIndexVec[i_next] = lastIndexVec[i];
                    }

                    const sofa::helper::vector<unsigned int> &shell= m_topology->getQuadsAroundVertex(lastIndexVec[i]);
                    for (j=0; j<shell.size(); ++j)
                    {
                        unsigned int ind_j = shell[j];

                        if ((unsigned)quads[ind_j][0]==last)
                            quads[ind_j][0]=tab[i];
                        else if ((unsigned)quads[ind_j][1]==last)
                            quads[ind_j][1]=tab[i];
                        else if ((unsigned)quads[ind_j][2]==last)
                            quads[ind_j][2]=tab[i];
                        else if ((unsigned)quads[ind_j][3]==last)
                            quads[ind_j][3]=tab[i];
                    }

                    --last;
                }
            }

            break;
        }

        case core::topology::POINTSRENUMBERING:
        {
            if (m_topology->getNbTriangles()>0)
            {
                unsigned int i;

                const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::core::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                for ( i = 0; i < triangles.size(); ++i)
                {
                    triangles[i][0]  = tab[triangles[i][0]];
                    triangles[i][1]  = tab[triangles[i][1]];
                    triangles[i][2]  = tab[triangles[i][2]];
                }

            }
            else if (m_topology->getNbQuads()>0)
            {
                unsigned int i;

                const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::core::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                for ( i = 0; i < quads.size(); ++i)
                {
                    quads[i][0]  = tab[quads[i][0]];
                    quads[i][1]  = tab[quads[i][1]];
                    quads[i][2]  = tab[quads[i][2]];
                    quads[i][3]  = tab[quads[i][3]];
                }
            }

            break;
        }

        case core::topology::POINTSMOVED:
        {
            updateVisual();
            break;
        }

        case core::topology::POINTSADDED:
        {
#if 0
            using sofa::core::behavior::BaseMechanicalState;
            BaseMechanicalState* mstate;
            //const unsigned int nbPoints = ( static_cast< const sofa::component::topology::PointsAdded * >( *itBegin ) )->getNbAddedVertices();
            m_topology->getContext()->get(mstate);
            /* fjourdes:
            ! THIS IS OBVIOUSLY NOT THE APPROPRIATE WAY TO DO IT !
            However : VisualModelImpl stores in two separates data the vertices
              - Data position in inherited ExtVec3State
              - Data vertices
            I don t know what is the purpose of the Data vertices (except at the init maybe ? )
            When doing topological operations on a graph like
            (removal points triangles / add of points triangles for instance)
            + Hexas
            ...
            + Triangles
            + - MechObj Triangles
            + - TriangleSetTopologyContainer Container
            + - Hexa2TriangleTopologycalMapping
            + + VisualModel
            + + - OglModel visual
            + + - IdentityMapping

            The IdentityMapping reflects the changes in topology by updating the Data position of the OglModel
            knowing the Data position of the MechObj named Triangles.
            However the Data vertices which is used to compute the normals is not updated, and the next computeNormals will
            fail. BTW this is odd that normals are computed using Data vertices since Data normals it belongs to ExtVec3State
            (like Data position) ...
            So my question is how the changes in the Data position of and OglModel are reflected to its Data vertices?
            It must be done somewhere since ultimately visual models are drawn correctly by OglModel::internalDraw !
            */

            if (mstate)
            {

                dmsg_info() << " changing size.  " << msgendl
                            << " oldsize    " << this->getSize() << msgendl
                            << " copying " << mstate->getSize() << " points from mechanical state.";

                vertices.resize(mstate->getSize());

                for (unsigned int i=0; i<vertices.size(); i++)
                {
                    vertices[i][0] = (Real)mstate->getPX(i);
                    vertices[i][1] = (Real)mstate->getPY(i);
                    vertices[i][2] = (Real)mstate->getPZ(i);
                }
            }
            updateVisual();
#endif
            break;
        }

        default:
            // Ignore events that are not Triangle  related.
            break;
        }; // switch( changeType )

        ++itBegin;
    } // while( changeIt != last; )

    m_triangles.endEdit();
    m_quads.endEdit();
    m_positions.endEdit();
}

void VisualModelImpl::initVisual()
{
}

void VisualModelImpl::exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, int& vindex, int& nindex, int& tindex, int& count)
{
    *out << "g "<<name<<"\n";

    if (mtl != NULL) // && !material.name.empty())
    {
        std::string name; // = material.name;
        if (name.empty())
        {
            std::ostringstream o; o << "mat" << count;
            name = o.str();
        }
        *mtl << "newmtl "<<name<<"\n";
        *mtl << "illum 4\n";
        if (material.getValue().useAmbient)
            *mtl << "Ka "<<material.getValue().ambient[0]<<' '<<material.getValue().ambient[1]<<' '<<material.getValue().ambient[2]<<"\n";
        if (material.getValue().useDiffuse)
            *mtl << "Kd "<<material.getValue().diffuse[0]<<' '<<material.getValue().diffuse[1]<<' '<<material.getValue().diffuse[2]<<"\n";
        *mtl << "Tf 1.00 1.00 1.00\n";
        *mtl << "Ni 1.00\n";
        if (material.getValue().useSpecular)
            *mtl << "Ks "<<material.getValue().specular[0]<<' '<<material.getValue().specular[1]<<' '<<material.getValue().specular[2]<<"\n";
        if (material.getValue().useShininess)
            *mtl << "Ns "<<material.getValue().shininess<<"\n";
        if (material.getValue().useDiffuse && material.getValue().diffuse[3]<1.0)
            *mtl << "Tf "<<material.getValue().diffuse[3]<<' '<<material.getValue().diffuse[3]<<' '<<material.getValue().diffuse[3]<<"\n";

        *out << "usemtl "<<name<<'\n';
    }

    const VecCoord& x = m_positions.getValue();
    const ResizableExtVector<Deriv>& vnormals = m_vnormals.getValue();
    const VecTexCoord& vtexcoords = m_vtexcoords.getValue();
    const ResizableExtVector<Edge>& edges = m_edges.getValue();
    const ResizableExtVector<Triangle>& triangles = m_triangles.getValue();
    const ResizableExtVector<Quad>& quads = m_quads.getValue();

    const ResizableExtVector<int> &vertPosIdx = m_vertPosIdx.getValue();
    const ResizableExtVector<int> &vertNormIdx = m_vertNormIdx.getValue();

    int nbv = x.size();

    for (int i=0; i<nbv; i++)
    {
        *out << "v "<< std::fixed << x[i][0]<<' '<< std::fixed <<x[i][1]<<' '<< std::fixed <<x[i][2]<<'\n';
    }

    int nbn = 0;

    if (vertNormIdx.empty())
    {
        nbn = vnormals.size();
        for (int i=0; i<nbn; i++)
        {
            *out << "vn "<< std::fixed << vnormals[i][0]<<' '<< std::fixed <<vnormals[i][1]<<' '<< std::fixed <<vnormals[i][2]<<'\n';
        }
    }
    else
    {
        for (unsigned int i = 0; i < vertNormIdx.size(); i++)
        {
            if (vertNormIdx[i] >= nbn)
                nbn = vertNormIdx[i]+1;
        }
        vector<int> normVertIdx(nbn);
        for (unsigned int i = 0; i < vertNormIdx.size(); i++)
        {
            normVertIdx[vertNormIdx[i]]=i;
        }
        for (int i = 0; i < nbn; i++)
        {
            int j = normVertIdx[i];
            *out << "vn "<< std::fixed << vnormals[j][0]<<' '<< std::fixed <<vnormals[j][1]<<' '<< std::fixed <<vnormals[j][2]<<'\n';
        }
    }

    int nbt = 0;
    if (!vtexcoords.empty())
    {
        nbt = vtexcoords.size();
        for (int i=0; i<nbt; i++)
        {
            *out << "vt "<< std::fixed << vtexcoords[i][0]<<' '<< std::fixed <<vtexcoords[i][1]<<'\n';
        }
    }

    for (unsigned int i = 0; i < edges.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<2; j++)
        {
            int i0 = edges[i][j];
            int i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            int i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    for (unsigned int i = 0; i < triangles.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<3; j++)
        {
            int i0 = triangles[i][j];
            int i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            int i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    for (unsigned int i = 0; i < quads.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<4; j++)
        {
            int i0 = quads[i][j];
            int i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            int i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    *out << sendl;
    vindex+=nbv;
    nindex+=nbn;
    tindex+=nbt;
}

//template class SOFA_BASE_VISUAL_API VisualModelPointHandler< ResizableExtVector<ExtVec3fTypes::Coord> >;
template class SOFA_BASE_VISUAL_API VisualModelPointHandler< ResizableExtVector<VisualModelImpl::Coord> >;
template class SOFA_BASE_VISUAL_API VisualModelPointHandler< ResizableExtVector<VisualModelImpl::TexCoord> >;

} // namespace visualmodel

namespace topology
{
template class PointData< sofa::defaulttype::ResizableExtVector<sofa::defaulttype::ExtVec3fTypes::Coord> >;
template class PointData< sofa::defaulttype::ResizableExtVector<sofa::defaulttype::ExtVec2fTypes::Coord> >;
}

} // namespace component

} // namespace sofa

