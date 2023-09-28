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
#include <sofa/component/visual/VisualModelImpl.h>

#include <sofa/type/Quat.h>
#include <sofa/type/vector.h>
#include <sofa/type/Material.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/grid/SparseGridTopology.h>

#include <sstream>
#include <map>
#include <memory>

namespace sofa::component::visual
{
using sofa::type::RGBAColor;
using sofa::type::Material;
using sofa::type::PrimitiveGroup;
using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;
using type::vector;

void VisualModelImpl::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::visual::VisualModel::parse(arg);

    VisualModelImpl* obj = this;

    if (arg->getAttribute("normals")!=nullptr)
        obj->setUseNormals(arg->getAttributeAsInt("normals", 1)!=0);

    if (arg->getAttribute("castshadow")!=nullptr)
        obj->setCastShadow(arg->getAttributeAsInt("castshadow", 1)!=0);

    if (arg->getAttribute("flip")!=nullptr)
        obj->flipFaces();

    if (arg->getAttribute("color"))
        obj->setColor(arg->getAttribute("color"));

    if (arg->getAttribute("su")!=nullptr || arg->getAttribute("sv")!=nullptr)
        m_scaleTex = TexCoord(arg->getAttributeAsFloat("su",1.0),
                              arg->getAttributeAsFloat("sv",1.0));

    if (arg->getAttribute("du")!=nullptr || arg->getAttribute("dv")!=nullptr)
        m_translationTex = TexCoord(arg->getAttributeAsFloat("du",0.0),
                                    arg->getAttributeAsFloat("dv",0.0));

    if (arg->getAttribute("rx")!=nullptr || arg->getAttribute("ry")!=nullptr || arg->getAttribute("rz")!=nullptr)
        m_rotation.setValue(Vec3Real((Real)arg->getAttributeAsFloat("rx",0.0),
                                     (Real)arg->getAttributeAsFloat("ry",0.0),
                                     (Real)arg->getAttributeAsFloat("rz",0.0)));

    if (arg->getAttribute("dx")!=nullptr || arg->getAttribute("dy")!=nullptr || arg->getAttribute("dz")!=nullptr)
        m_translation.setValue(Vec3Real((Real)arg->getAttributeAsFloat("dx",0.0),
                                        (Real)arg->getAttributeAsFloat("dy",0.0),
                                        (Real)arg->getAttributeAsFloat("dz",0.0)));

    if (arg->getAttribute("scale")!=nullptr)
    {
        m_scale.setValue(Vec3Real((Real)arg->getAttributeAsFloat("scale",1.0),
                                  (Real)arg->getAttributeAsFloat("scale",1.0),
                                  (Real)arg->getAttributeAsFloat("scale",1.0)));
    }
    else if (arg->getAttribute("sx")!=nullptr || arg->getAttribute("sy")!=nullptr || arg->getAttribute("sz")!=nullptr)
    {
        m_scale.setValue(Vec3Real((Real)arg->getAttributeAsFloat("sx",1.0),
                                  (Real)arg->getAttributeAsFloat("sy",1.0),
                                  (Real)arg->getAttributeAsFloat("sz",1.0)));
    }
}

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
    , fileMesh          (initData   (&fileMesh, "filename"," Path to an ogl model"))
    , texturename       (initData   (&texturename, "texturename", "Name of the Texture"))
    , m_translation     (initData   (&m_translation, Vec3Real(), "translation", "Initial Translation of the object"))
    , m_rotation        (initData   (&m_rotation, Vec3Real(), "rotation", "Initial Rotation of the object"))
    , m_scale           (initData   (&m_scale, Vec3Real(1.0,1.0,1.0), "scale3d", "Initial Scale of the object"))
    , m_scaleTex        (initData   (&m_scaleTex, TexCoord(1.f,1.f), "scaleTex", "Scale of the texture"))
    , m_translationTex  (initData   (&m_translationTex, TexCoord(0.f,0.f), "translationTex", "Translation of the texture"))
    , material			(initData	(&material, "material", "Material")) // tex(nullptr)
    , putOnlyTexCoords	(initData	(&putOnlyTexCoords, (bool) false, "putOnlyTexCoords", "Give Texture Coordinates without the texture binding"))
    , srgbTexturing		(initData	(&srgbTexturing, (bool) false, "srgbTexturing", "When sRGB rendering is enabled, is the texture in sRGB colorspace?"))
    , materials			(initData	(&materials, "materials", "List of materials"))
    , groups			(initData	(&groups, "groups", "Groups of triangles and quads using a given material"))
    , l_topology        (initLink   ("topology", "link to the topology container"))
    , xformsModified(false)
{
    m_topology = nullptr;

    addAlias(&fileMesh, "fileMesh");

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

    addUpdateCallback("updateTextures", { &texturename },
        [&](const core::DataTracker& tracker) -> sofa::core::objectmodel::ComponentState
    {
        SOFA_UNUSED(tracker);
        m_textureChanged = true;
        return sofa::core::objectmodel::ComponentState::Loading;
    }, { &d_componentState });
}

VisualModelImpl::~VisualModelImpl()
{
}

bool VisualModelImpl::hasTransparent()
{
    const Material& material = this->material.getValue();
    const helper::ReadAccessor< Data< type::vector<FaceGroup> > > groups = this->groups;
    const helper::ReadAccessor< Data< type::vector<Material> > > materials = this->materials;
    if (groups.empty())
        return (material.useDiffuse && material.diffuse[3] < 1.0);
    else
    {
        for (std::size_t i = 0; i < groups.size(); ++i)
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
    const helper::ReadAccessor< Data< type::vector<FaceGroup> > > groups = this->groups;
    const helper::ReadAccessor< Data< type::vector<Material> > > materials = this->materials;
    if (groups.empty())
        return !(material.useDiffuse && material.diffuse[3] < 1.0);
    else
    {
        for (std::size_t i = 0; i < groups.size(); ++i)
        {
            const Material& m = (groups[i].materialId == -1) ? material : materials[groups[i].materialId];
            if (!(m.useDiffuse && m.diffuse[3] < 1.0))
                return true;
        }
    }
    return false;
}

void VisualModelImpl::doDrawVisual(const core::visual::VisualParams* vparams)
{
    if (d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Loading)
    {
        if (m_textureChanged)
        {
            deleteTextures();
            loadTexture(texturename.getFullPath());
            m_textureChanged = false;
        }
        initVisual();
        updateBuffers();
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }

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
    const auto &facetsImport = objLoader.getFacets();
    const auto &verticesImport = objLoader.getVertices();
    const auto&normalsImport = objLoader.getNormals();
    const auto&texCoordsImport = objLoader.getTexCoords();

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
        helper::WriteAccessor< Data< type::vector<Material> > > materials = this->materials;
        helper::WriteAccessor< Data< type::vector<FaceGroup> > > groups = this->groups;
        materials.resize(objLoader.getMaterials().size());
        for (std::size_t i=0; i<materials.size(); ++i)
            materials[i] = objLoader.getMaterials()[i];

        // compute the edge / triangle / quad index corresponding to each facet
        // convert the groups info
        enum { NBE = 0, NBT = 1, NBQ = 2 };
        type::fixed_array<visual_index_type, 3> nbf{ 0,0,0 };
        type::vector< type::fixed_array<visual_index_type, 3> > facet2tq;
        facet2tq.resize(facetsImport.size()+1);
        for (std::size_t i = 0; i < facetsImport.size(); i++)
        {
            facet2tq[i] = nbf;
            const auto& vertNormTexIndex = facetsImport[i];
            const auto& verts = vertNormTexIndex[0];
            if (verts.size() < 2)
                ; // ignore points
            else if (verts.size() == 2)
                nbf[NBE] += 1;
            else if (verts.size() == 4)
                nbf[NBQ] += 1;
            else
                nbf[NBT] += visual_index_type(verts.size()-2);
        }
        facet2tq[facetsImport.size()] = nbf;
        groups.resize(objLoader.getGroups().size());
        for (std::size_t ig = 0; ig < groups.size(); ig++)
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

    std::size_t nbVIn = verticesImport.size();
    // First we compute for each point how many pair of normal/texcoord indices are used
    // The map store the final index of each combinaison
    vector< std::map< std::pair<sofa::Index,sofa::Index>, sofa::Index > > vertTexNormMap;
    vertTexNormMap.resize(nbVIn);
    for (std::size_t i = 0; i < facetsImport.size(); i++)
    {
        const auto& vertNormTexIndex = facetsImport[i];
        if (vertNormTexIndex[0].size() < 3 && !m_keepLines.getValue() ) continue; // ignore lines
        const auto& verts = vertNormTexIndex[0];
        const auto& texs = vertNormTexIndex[1];
        const auto& norms = vertNormTexIndex[2];
        for (std::size_t j = 0; j < verts.size(); j++)
        {
            vertTexNormMap[verts[j]][std::make_pair((tex ? texs[j] : sofa::InvalidID), (m_useNormals.getValue() ? norms[j] : 0))] = 0;
        }
    }

    // Then we can compute how many vertices are created
    std::size_t nbVOut = 0;
    bool vsplit = false;
    for (std::size_t i = 0; i < nbVIn; i++)
    {
        nbVOut += vertTexNormMap[i].size();
    }

    msg_info() << nbVIn << " input positions, " << nbVOut << " final vertices.   ";

    if (nbVIn != nbVOut)
        vsplit = true;

    // Then we can create the final arrays
    VecCoord& restPositions = *(m_restPositions.beginEdit());
    VecCoord& positions = *(m_positions.beginEdit());
    VecCoord& vertices2 = *(m_vertices2.beginEdit());
    VecDeriv& vnormals = *(m_vnormals.beginEdit());
    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    auto& vertPosIdx = (*m_vertPosIdx.beginEdit());
    auto& vertNormIdx = (*m_vertNormIdx.beginEdit());

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

    sofa::Size nbNOut = 0; /// Number of different normals
    for (sofa::Index i = 0, j = 0; i < nbVIn; i++)
    {
        positions[i] = verticesImport[i];

        if (m_initRestPositions.getValue())
            restPositions[i] = verticesImport[i];

        std::map<sofa::Index, sofa::Index> normMap;
        for (auto it = vertTexNormMap[i].begin();
             it != vertTexNormMap[i].end(); ++it)
        {
            sofa::Index t = it->first.first;
            sofa::Index n = it->first.second;
            if ( m_useNormals.getValue() && n < normalsImport.size())
                vnormals[j] = normalsImport[n];
            if (t < texCoordsImport.size())
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
    VecVisualEdge& edges = *(m_edges.beginEdit());
    VecVisualTriangle& triangles = *(m_triangles.beginEdit());
    VecVisualQuad& quads = *(m_quads.beginEdit());

    for (std::size_t i = 0; i < facetsImport.size(); i++)
    {
        const auto& vertNormTexIndex = facetsImport[i];
        const auto& verts = vertNormTexIndex[0];
        const auto& texs = vertNormTexIndex[1];
        const auto& norms = vertNormTexIndex[2];
        vector<visual_index_type> idxs;
        idxs.resize(verts.size());
        for (std::size_t j = 0; j < verts.size(); j++)
        {
            idxs[j] = vertTexNormMap[verts[j]][std::make_pair((tex?texs[j]:-1), (m_useNormals.getValue() ? norms[j] : 0))];
            if (idxs[j] >= nbVOut)
            {
                msg_error() << this->getPathName()<<" index "<<idxs[j]<<" out of range";
                idxs[j] = 0;
            }
        }

        if (verts.size() == 2)
        {
            edges.push_back({idxs[0], idxs[1]});
        }
        else if (verts.size() == 4)
        {
            quads.push_back({ idxs[0],idxs[1],idxs[2],idxs[3] });
        }
        else
        {
            for (std::size_t j = 2; j < verts.size(); j++)
            {
                triangles.push_back({ idxs[0],idxs[j - 1],idxs[j] });
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
            const bool textureLoaded = loadTexture(textureName);
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
                msg_error() << "Mesh creation failed. Loading mesh file directly inside the VisualModel is not maintained anymore. Use a MeshLoader and link the Data to the VisualModel. E.g:" << msgendl
                    << "<MeshOBJLoader name='myLoader' filename='myFilePath.obj'/>" << msgendl
                    << "<OglModel src='@myLoader'/>";
                return false;
            }
            else
            {
                if(objLoader.get()->loaderType == "obj")
                {
                    //Modified: previously, the texture coordinates were not loaded correctly if no texture name was specified.
                    //setMesh(*objLoader,tex);
                    msg_warning() << "Loading obj mesh file directly inside the VisualModel will be deprecated soon. Use a MeshOBJLoader and link the Data to the VisualModel. E.g:" << msgendl
                        << "<MeshOBJLoader name='myLoader' filename='myFilePath.obj'/>" << msgendl
                        << "<OglModel src='@myLoader'/>";

                    setMesh(*objLoader, true);
                }
                else
                {
                    msg_error() << "Loading mesh file directly inside the VisualModel is not anymore supported since release 18.06. Use a MeshLoader and link the Data to the VisualModel. E.g:" << msgendl
                        << "<MeshOBJLoader name='myLoader' filename='myFilePath.obj'/>" << msgendl
                        << "<OglModel src='@myLoader'/>";
                    return false;
                }
            }

            if(textureName.empty())
            {
                //we check how many textures are linked with a material (only if a texture name is not defined in the scn file)
                bool isATextureLinked = false;
                for (std::size_t i = 0 ; i < this->materials.getValue().size() ; i++)
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
    m_scaleTex.setValue(TexCoord(1.f,1.f));
    m_translationTex.setValue(TexCoord(0.f,0.f));
}

void VisualModelImpl::applyTranslation(const SReal dx, const SReal dy, const SReal dz)
{
    const Coord d((Real)dx,(Real)dy,(Real)dz);

    Data< VecCoord >* d_x = this->write(core::VecCoordId::position());
    VecCoord &x = *d_x->beginEdit();

    for (std::size_t i = 0; i < x.size(); i++)
    {
        x[i] += d;
    }

    d_x->endEdit();

    if(m_initRestPositions.getValue())
    {
        VecCoord& restPositions = *(m_restPositions.beginEdit());

        for (std::size_t i = 0; i < restPositions.size(); i++)
        {
            restPositions[i] += d;
        }

        m_restPositions.endEdit();
    }


    updateVisual();
}

void VisualModelImpl::applyRotation(const SReal rx, const SReal ry, const SReal rz)
{
    const auto q = type::Quat<SReal>::createQuaterFromEuler( Vec3(rx,ry,rz)*M_PI/180.0);
    applyRotation(q);
}

void VisualModelImpl::applyRotation(const Quat<SReal> q)
{
    Data< VecCoord >* d_x = this->write(core::VecCoordId::position());
    VecCoord &x = *d_x->beginEdit();

    for (std::size_t i = 0; i < x.size(); i++)
    {
        x[i] = q.rotate(x[i]);
    }

    d_x->endEdit();

    if(m_initRestPositions.getValue())
    {
        VecCoord& restPositions = *(m_restPositions.beginEdit());

        for (std::size_t i = 0; i < restPositions.size(); i++)
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

    for (std::size_t i = 0; i < x.size(); i++)
    {
        x[i][0] *= (Real)sx;
        x[i][1] *= (Real)sy;
        x[i][2] *= (Real)sz;
    }

    d_x->endEdit();

    if(m_initRestPositions.getValue())
    {
        VecCoord& restPositions = *(m_restPositions.beginEdit());

        for (std::size_t i = 0; i < restPositions.size(); i++)
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
    const float dUf = float(dU);
    const float dVf = float(dV);
    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    for (std::size_t i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] += dUf;
        vtexcoords[i][1] += dVf;
    }
    m_vtexcoords.endEdit();
}

void VisualModelImpl::applyUVScale(const Real scaleU, const Real scaleV)
{
    const float scaleUf = float(scaleU);
    const float scaleVf = float(scaleV);
    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    for (std::size_t i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] *= scaleUf;
        vtexcoords[i][1] *= scaleVf;
    }
    m_vtexcoords.endEdit();
}


void VisualModelImpl::init()
{
    VisualModel::init();

    if (fileMesh.isSet()) // check if using internal mesh
    {
        initFromFileMesh();
    }
    else
    {
        if (m_vertPosIdx.getValue().size() > 0 && m_vertices2.getValue().empty())
        {
            // handle case where vertPosIdx was initialized through a loader
            initPositionFromVertices();
        }

        // check if not init by Data
        if (m_positions.getValue().empty() || (!m_triangles.isSet() && !m_quads.isSet()) )
        {
            initFromTopology(); // if not init from Data nor from filemesh, will init from topology
        }

        // load texture
        if (texturename.isSet())
        {
            std::string textureFilename = texturename.getFullPath();
            if (sofa::helper::system::DataRepository.findFile(textureFilename))
            {
                msg_info() << "loading file " << textureFilename;
                const bool textureLoaded = loadTexture(textureFilename);
                if (!textureLoaded)
                {
                    msg_error() << "Texture " << textureFilename << " cannot be loaded";
                }
            }
            else
            {
                msg_error() << "Texture \"" << textureFilename << "\" not found";
            }

            applyUVTransformation();
        }
    }

    if (m_topology == nullptr && (m_positions.getValue().size() == 0))
    {
        msg_warning() << "Neither an .obj file nor a topology is present for this VisualModel.";
        useTopology = false;
        return;
    }

    applyScale(m_scale.getValue()[0], m_scale.getValue()[1], m_scale.getValue()[2]);
    applyRotation(m_rotation.getValue()[0], m_rotation.getValue()[1], m_rotation.getValue()[2]);
    applyTranslation(m_translation.getValue()[0], m_translation.getValue()[1], m_translation.getValue()[2]);

    m_translation.setValue(Vec3Real());
    m_rotation.setValue(Vec3Real());
    m_scale.setValue(Vec3Real(1,1,1));

    updateVisual();
}


void VisualModelImpl::initPositionFromVertices()
{
    m_vertices2.setValue(m_positions.getValue());
    if (m_positions.getParent())
    {
        m_positions.delInput(m_positions.getParent()); // remove any link to positions, as we need to recompute it
    }
    helper::WriteAccessor<Data<VecCoord>> vIn = m_positions;
    const helper::ReadAccessor<Data<VecCoord>> vOut = m_vertices2;
    const helper::ReadAccessor<Data<type::vector<visual_index_type>>> vertPosIdx = m_vertPosIdx;
    std::size_t nbVIn = 0;
    for (std::size_t i = 0; i < vertPosIdx.size(); ++i)
    {
        if (vertPosIdx[i] >= nbVIn)
        {
            nbVIn = vertPosIdx[i] + 1;
        }
    }
    vIn.resize(nbVIn);
    for (std::size_t i = 0; i < vertPosIdx.size(); ++i)
    {
        vIn[vertPosIdx[i]] = vOut[i];
    }
    m_topology = nullptr; // make sure we don't use the topology
}


void VisualModelImpl::initFromFileMesh()
{
    load(fileMesh.getFullPath(), "", texturename.getFullPath());
}


void VisualModelImpl::initFromTopology()
{
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
        return;

    msg_info() << "Use topology " << m_topology->getName();
    useTopology = true;

    // compute mesh from topology
    computeMesh();

    sofa::core::topology::TopologyModifier* topoMod;
    m_topology->getContext()->get(topoMod);

    if (topoMod == nullptr)
    {
        m_handleDynamicTopology.setValue(false);
    }
    // add the functions to handle topology changes.
    if (m_handleDynamicTopology.getValue())
    {
        if (m_topology->getTopologyType() == sofa::geometry::ElementType::QUAD || m_topology->getTopologyType() == sofa::geometry::ElementType::HEXAHEDRON)
        {
            m_quads.createTopologyHandler(m_topology);
            m_quads.setCreationCallback([](sofa::Index elemID, VisualQuad& visuQuad,
                const core::topology::BaseMeshTopology::Quad& topoQuad,
                const sofa::type::vector< sofa::Index >& ancestors,
                const sofa::type::vector< SReal >& coefs)
            {
                SOFA_UNUSED(elemID);
                SOFA_UNUSED(ancestors);
                SOFA_UNUSED(coefs);
                visuQuad = topoQuad; // simple copy from topology Data
            });
        }


        if (m_topology->getTopologyType() == sofa::geometry::ElementType::TRIANGLE || m_topology->getTopologyType() == sofa::geometry::ElementType::TETRAHEDRON)
        {
            m_triangles.createTopologyHandler(m_topology);
            m_triangles.setCreationCallback([](sofa::Index elemID, VisualTriangle& visuTri,
                const core::topology::BaseMeshTopology::Triangle& topoTri,
                const sofa::type::vector< sofa::Index >& ancestors,
                const sofa::type::vector< SReal >& coefs)
            {
                SOFA_UNUSED(elemID);
                SOFA_UNUSED(ancestors);
                SOFA_UNUSED(coefs);
                visuTri = topoTri; // simple copy from topology Data
            });
        }

        if (m_topology->getTopologyType() == sofa::geometry::ElementType::EDGE)
        {
            m_edges.createTopologyHandler(m_topology);
            m_edges.setCreationCallback([](sofa::Index elemID, VisualEdge& visuEdge,
                const core::topology::BaseMeshTopology::Edge& topoEdge,
                const sofa::type::vector< sofa::Index >& ancestors,
                const sofa::type::vector< SReal >& coefs)
            {
                SOFA_UNUSED(elemID);
                SOFA_UNUSED(ancestors);
                SOFA_UNUSED(coefs);
                visuEdge = topoEdge; // simple copy from topology Data
            });
        }

        m_positions.createTopologyHandler(m_topology);
        m_positions.setDestructionCallback([this](sofa::Index pointIndex, Coord& coord)
        {
            SOFA_UNUSED(pointIndex);
            SOFA_UNUSED(coord);

            const auto last = m_positions.getLastElementIndex();

            if (m_topology->getNbTriangles() > 0)
            {
                const auto& shell = m_topology->getTrianglesAroundVertex(last);
                for (sofa::Index j = 0; j < shell.size(); ++j)
                {
                    m_dirtyTriangles.insert(shell[j]);
                }
            }
            else if (m_topology->getNbQuads() > 0)
            {
                const auto& shell = m_topology->getQuadsAroundVertex(last);
                for (sofa::Index j = 0; j < shell.size(); ++j)
                {
                    m_dirtyQuads.insert(shell[j]);
                }
            }
        });

        if (m_vtexcoords.isSet()) // Data set from loader as not part of the topology
        {
            m_vtexcoords.updateIfDirty();
            m_vtexcoords.setParent(nullptr); // manually break the data link to follow topological changes
            m_vtexcoords.createTopologyHandler(m_topology);
            m_vtexcoords.setCreationCallback([this](sofa::Index pointIndex, TexCoord& tCoord,
                const core::topology::BaseMeshTopology::Point& point,
                const sofa::type::vector< sofa::Index >& ancestors,
                const sofa::type::vector< SReal >& coefs)
            {
                SOFA_UNUSED(pointIndex);
                SOFA_UNUSED(point);

                const VecTexCoord& texcoords = m_vtexcoords.getValue();
                tCoord = TexCoord(0, 0);
                for (sofa::Index i = 0; i < ancestors.size(); i++)
                {
                    const TexCoord& tAnces = texcoords[ancestors[i]];
                    tCoord += tAnces * coefs[i];
                }
            });
        }
    }

}



void VisualModelImpl::computeNormals()
{
    const VecCoord& vertices = getVertices();

    if (vertices.empty() || (!m_updateNormals.getValue() && (m_vnormals.getValue()).size() == (vertices).size())) return;

    const VecVisualTriangle& triangles = m_triangles.getValue();
    const VecVisualQuad& quads = m_quads.getValue();
    const type::vector<visual_index_type> &vertNormIdx = m_vertNormIdx.getValue();

    if (vertNormIdx.empty())
    {
        const std::size_t nbn = vertices.size();
        auto normals = sofa::helper::getWriteOnlyAccessor(m_vnormals);

        normals.resize(nbn);
        std::memset(&normals[0], 0, sizeof(normals[0]) * nbn); // bulk reset with zeros

        for (const auto& triangle : triangles)
        {
            const Coord& v1 = vertices[ triangle[0] ];
            const Coord& v2 = vertices[ triangle[1] ];
            const Coord& v3 = vertices[ triangle[2] ];
            const Coord n = cross(v2-v1, v3-v1);

            normals[ triangle[0] ] += n;
            normals[ triangle[1] ] += n;
            normals[ triangle[2] ] += n;
        }

        for (const auto& quad : quads)
        {
            const Coord & v1 = vertices[ quad[0] ];
            const Coord & v2 = vertices[ quad[1] ];
            const Coord & v3 = vertices[ quad[2] ];
            const Coord & v4 = vertices[ quad[3] ];
            const Coord n1 = cross(v2-v1, v4-v1);
            const Coord n2 = cross(v3-v2, v1-v2);
            const Coord n3 = cross(v4-v3, v2-v3);
            const Coord n4 = cross(v1-v4, v3-v4);

            normals[ quad[0] ] += n1;
            normals[ quad[1] ] += n2;
            normals[ quad[2] ] += n3;
            normals[ quad[3] ] += n4;
        }

        for (auto& normal : normals)
        {
            normal.normalize();
        }
    }
    else
    {
        const std::size_t nbn = static_cast<std::size_t>(*std::max_element(vertNormIdx.begin(), vertNormIdx.end())) + 1;
        sofa::type::vector<Coord> normals(nbn); // will call the default ctor, which initializes with zeros

        for (const auto& triangle : triangles)
        {
            const Coord & v1 = vertices[ triangle[0] ];
            const Coord & v2 = vertices[ triangle[1] ];
            const Coord & v3 = vertices[ triangle[2] ];
            const Coord n = cross(v2-v1, v3-v1);

            normals[vertNormIdx[ triangle[0] ]] += n;
            normals[vertNormIdx[ triangle[1] ]] += n;
            normals[vertNormIdx[ triangle[2] ]] += n;
        }

        for (const auto& quad : quads)
        {
            const Coord & v1 = vertices[ quad[0] ];
            const Coord & v2 = vertices[ quad[1] ];
            const Coord & v3 = vertices[ quad[2] ];
            const Coord & v4 = vertices[ quad[3] ];
            const Coord n1 = cross(v2-v1, v4-v1);
            const Coord n2 = cross(v3-v2, v1-v2);
            const Coord n3 = cross(v4-v3, v2-v3);
            const Coord n4 = cross(v1-v4, v3-v4);

            normals[vertNormIdx[ quad[0] ]] += n1;
            normals[vertNormIdx[ quad[1] ]] += n2;
            normals[vertNormIdx[ quad[2] ]] += n3;
            normals[vertNormIdx[ quad[3] ]] += n4;
        }

        for (auto& normal : normals)
        {
            normal.normalize();
        }

        auto vnormals = sofa::helper::getWriteOnlyAccessor(m_vnormals);
        vnormals.resize(vertices.size());
        for (std::size_t i = 0; i < vertices.size(); i++)
        {
            vnormals[i] = normals[vertNormIdx[i]];
        }
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

    const VecVisualTriangle& triangles = m_triangles.getValue();
    const VecVisualQuad& quads = m_quads.getValue();
    const VecCoord& vertices = getVertices();
    const VecTexCoord& texcoords = m_vtexcoords.getValue();
    const auto& normals = m_vnormals.getValue();

    auto tangents = sofa::helper::getWriteOnlyAccessor(m_vtangents);
    auto bitangents = sofa::helper::getWriteOnlyAccessor(m_vbitangents);

    tangents.resize(vertices.size());
    bitangents.resize(vertices.size());

    for (unsigned i = 0; i < vertices.size(); i++)
    {
        tangents[i].clear();
        bitangents[i].clear();
    }
    const bool fixMergedUVSeams = m_fixMergedUVSeams.getValue();
    for (std::size_t i = 0; i < triangles.size() ; i++)
    {
        const Coord& v1 = vertices[triangles[i][0]];
        const Coord& v2 = vertices[triangles[i][1]];
        const Coord& v3 = vertices[triangles[i][2]];
        const TexCoord& t1 = texcoords[triangles[i][0]];
        TexCoord t2 = texcoords[triangles[i][1]];
        TexCoord t3 = texcoords[triangles[i][2]];
        if (fixMergedUVSeams)
        {
            for (Size j=0; j<TexCoord::size(); ++j)
            {
                t2[j] += helper::rnear(t1[j]-t2[j]);
                t3[j] += helper::rnear(t1[j]-t3[j]);
            }
        }
        const Coord t = computeTangent(v1, v2, v3, t1, t2, t3);
        const Coord b = computeBitangent(v1, v2, v3, t1, t2, t3);

        tangents[triangles[i][0]] += t;
        tangents[triangles[i][1]] += t;
        tangents[triangles[i][2]] += t;
        bitangents[triangles[i][0]] += b;
        bitangents[triangles[i][1]] += b;
        bitangents[triangles[i][2]] += b;
    }

    for (std::size_t i = 0; i < quads.size() ; i++)
    {
        const Coord& v1 = vertices[quads[i][0]];
        const Coord& v2 = vertices[quads[i][1]];
        const Coord& v3 = vertices[quads[i][2]];
        const Coord& v4 = vertices[quads[i][3]];
        const TexCoord& t1 = texcoords[quads[i][0]];
        const TexCoord& t2 = texcoords[quads[i][1]];
        const TexCoord& t3 = texcoords[quads[i][2]];
        const TexCoord& t4 = texcoords[quads[i][3]];

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
    for (std::size_t i = 0; i < vertices.size(); i++)
    {
        const Coord& n = normals[i];
        Coord& t = tangents[i];
        Coord& b = bitangents[i];

        b = sofa::type::cross(n, t.normalized());
        t = sofa::type::cross(b, n);
    }

}

void VisualModelImpl::computeBBox(const core::ExecParams*, bool)
{
    const VecCoord& x = getVertices(); //m_vertices.getValue();

    SReal minBBox[3] = {std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max()};
    SReal maxBBox[3] = {-std::numeric_limits<Real>::max(),-std::numeric_limits<Real>::max(),-std::numeric_limits<Real>::max()};
    for (std::size_t i = 0; i < x.size(); i++)
    {
        const Coord& p = x[i];
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    this->f_bbox.setValue(sofa::type::TBoundingBox<SReal>(minBBox,maxBBox));
}


void VisualModelImpl::computeUVSphereProjection()
{
    const sofa::core::visual::VisualParams* vparams = sofa::core::visual::VisualParams::defaultInstance();
    this->computeBBox(vparams);

    auto center = (this->f_bbox.getValue().minBBox() + this->f_bbox.getValue().maxBBox())*0.5f;

    // Map mesh vertices to sphere
    // transform cart to spherical coordinates (r, theta, phi) and sphere to cart back with radius = 1
    const VecCoord& coords = getVertices();

    const std::size_t nbrV = coords.size();
    VecCoord m_sphereV;
    m_sphereV.resize(nbrV);

    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    vtexcoords.resize(nbrV);

    for (std::size_t i = 0; i < nbrV; ++i)
    {
        Coord Vcentered = coords[i] - center;
        SReal r = sqrt(Vcentered[0] * Vcentered[0] + Vcentered[1] * Vcentered[1] + Vcentered[2] * Vcentered[2]);
        const SReal theta = acos(Vcentered[2] / r);
        const SReal phi = atan2(Vcentered[1], Vcentered[0]);

        r = 1.0;
        m_sphereV[i][0] = r * sin(theta)*cos(phi) + center[0];
        m_sphereV[i][1] = r * sin(theta)*sin(phi) + center[1];
        m_sphereV[i][2] = r * cos(theta) + center[2];

        Coord pos = m_sphereV[i] - center;
        pos.normalize();
        vtexcoords[i][0] = float(0.5 + atan2(pos[1], pos[0]) / (2 * R_PI));
        vtexcoords[i][1] = float(0.5 - asin(pos[2]) / R_PI);
    }

    m_vtexcoords.endEdit();
}

void VisualModelImpl::flipFaces()
{
    VecDeriv& vnormals = *(m_vnormals.beginEdit());
    VecVisualEdge& edges = *(m_edges.beginEdit());
    VecVisualTriangle& triangles = *(m_triangles.beginEdit());
    VecVisualQuad& quads = *(m_quads.beginEdit());

    for (std::size_t i = 0; i < edges.size() ; i++)
    {
        const sofa::Index temp = edges[i][1];
        edges[i][1] = visual_index_type(edges[i][0]);
        edges[i][0] = visual_index_type(temp);
    }

    for (std::size_t i = 0; i < triangles.size() ; i++)
    {
        const sofa::Index temp = triangles[i][1];
        triangles[i][1] = visual_index_type(triangles[i][2]);
        triangles[i][2] = visual_index_type(temp);
    }

    for (std::size_t i = 0; i < quads.size() ; i++)
    {
        const sofa::Index temp = quads[i][1];
        quads[i][1] = visual_index_type(quads[i][3]);
        quads[i][3] = visual_index_type(temp);
    }

    for (std::size_t i = 0; i < vnormals.size(); i++)
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
    if (modified && !getVertices().empty())
    {
        if (useTopology)
        {
            if ((m_topology->getRevision() != lastMeshRev) && !m_handleDynamicTopology.getValue())
            {
                // Follow change from static topology, this should not be used as the whole mesh is copied
                computeMesh();
            }

            if (!m_dirtyTriangles.empty())
            {
                helper::WriteOnlyAccessor< Data<VecVisualTriangle > > triangles = m_triangles;
                const vector< Triangle >& inputTriangles = m_topology->getTriangles();

                for (const auto idTri : m_dirtyTriangles)
                {
                    triangles[idTri] = inputTriangles[idTri];
                }
                m_dirtyTriangles.clear();
            }

            if (!m_dirtyQuads.empty())
            {
                helper::WriteOnlyAccessor< Data<VecVisualQuad > > quads = m_quads;
                const vector< Quad >& inputQuads = m_topology->getQuads();

                for (const auto idQuad : m_dirtyQuads)
                {
                    quads[idQuad] = inputQuads[idQuad];
                }
                m_dirtyQuads.clear();
            }
        }

        {
            SCOPED_TIMER_VARNAME(t, "VisualModelImpl::computePositions");
            computePositions();
        }
        {
            SCOPED_TIMER_VARNAME(t, "VisualModelImpl::computeNormals");
            computeNormals();
        }
        if (m_updateTangents.getValue())
        {
            SCOPED_TIMER_VARNAME(t, "VisualModelImpl::computeTangents");
            computeTangents();
        }
        if (m_vtexcoords.getValue().size() == 0)
        {
            SCOPED_TIMER_VARNAME(t, "VisualModelImpl::computeUVSphereProjection");
            computeUVSphereProjection();
        }
        {
            SCOPED_TIMER_VARNAME(t, "VisualModelImpl::updateBuffers");
            updateBuffers();
        }

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
    const type::vector<visual_index_type> &vertPosIdx = m_vertPosIdx.getValue();

    if (!vertPosIdx.empty())
    {
        // Need to transfer positions
        VecCoord& vertices = *(m_vertices2.beginEdit());
        const VecCoord& positions = this->m_positions.getValue();

        for (std::size_t i=0 ; i < vertices.size(); ++i)
            vertices[i] = positions[vertPosIdx[i]];

        m_vertices2.endEdit();
    }
}

void VisualModelImpl::computeMesh()
{
    using sofa::component::topology::container::grid::SparseGridTopology;
    using sofa::core::behavior::BaseMechanicalState;

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

            for (std::size_t i=0; i<vertices.size(); i++)
            {
                vertices[i][0] = SReal(m_topology->getPX(Size(i)));
                vertices[i][1] = SReal(m_topology->getPY(Size(i)));
                vertices[i][2] = SReal(m_topology->getPZ(Size(i)));
            }

        }
        else
        {
            const BaseMechanicalState* mstate = m_topology->getContext()->getMechanicalState();

            if (mstate)
            {
                dmsg_info() << " copying " << mstate->getSize() << " points from mechanical state" ;

                vertices.resize(mstate->getSize());

                for (std::size_t i=0; i<vertices.size(); i++)
                {
                    vertices[i][0] = (Real)mstate->getPX(Size(i));
                    vertices[i][1] = (Real)mstate->getPY(Size(i));
                    vertices[i][2] = (Real)mstate->getPZ(Size(i));
                }

            }
        }
        m_positions.endEdit();
    }

    lastMeshRev = m_topology->getRevision();

    const vector< Triangle >& inputTriangles = m_topology->getTriangles();


    dmsg_info() << " copying " << inputTriangles.size() << " triangles from topology" ;

    VecVisualTriangle& triangles = *(m_triangles.beginEdit());
    triangles.resize(inputTriangles.size());

    for (std::size_t i=0; i<triangles.size(); ++i)
    {
        triangles[i][0] = visual_index_type(inputTriangles[i][0]);
        triangles[i][1] = visual_index_type(inputTriangles[i][1]);
        triangles[i][2] = visual_index_type(inputTriangles[i][2]);
    }
    m_triangles.endEdit();


    const vector< BaseMeshTopology::Quad >& inputQuads = m_topology->getQuads();

    dmsg_info() << " copying " << inputQuads.size()<< " quads from topology." ;

    VecVisualQuad& quads = *(m_quads.beginEdit());
    quads.resize(inputQuads.size());

    for (std::size_t i=0; i<quads.size(); ++i)
    {
        quads[i][0] = visual_index_type(inputQuads[i][0]);
        quads[i][1] = visual_index_type(inputQuads[i][1]);
        quads[i][2] = visual_index_type(inputQuads[i][2]);
        quads[i][3] = visual_index_type(inputQuads[i][3]);
    }
    m_quads.endEdit();
}


void VisualModelImpl::initVisual()
{
}

void VisualModelImpl::exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, sofa::Index& vindex, sofa::Index& nindex, sofa::Index& tindex, int& count)
{
    *out << "g "<<name<<"\n";

    if (mtl != nullptr)
    {
        std::string name;
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
    const VecDeriv& vnormals = m_vnormals.getValue();
    const VecTexCoord& vtexcoords = m_vtexcoords.getValue();
    const VecVisualEdge& edges = m_edges.getValue();
    const VecVisualTriangle& triangles = m_triangles.getValue();
    const VecVisualQuad& quads = m_quads.getValue();

    const type::vector<visual_index_type> &vertPosIdx = m_vertPosIdx.getValue();
    const type::vector<visual_index_type> &vertNormIdx = m_vertNormIdx.getValue();

    auto nbv = Size(x.size());

    for (std::size_t i=0; i<nbv; i++)
    {
        *out << "v "<< std::fixed << x[i][0]<<' '<< std::fixed <<x[i][1]<<' '<< std::fixed <<x[i][2]<<'\n';
    }

    Size nbn = 0;

    if (vertNormIdx.empty())
    {
        nbn = sofa::Size(vnormals.size());
        for (sofa::Index i=0; i<nbn; i++)
        {
            *out << "vn "<< std::fixed << vnormals[i][0]<<' '<< std::fixed <<vnormals[i][1]<<' '<< std::fixed <<vnormals[i][2]<<'\n';
        }
    }
    else
    {
        for (sofa::Index i = 0; i < vertNormIdx.size(); i++)
        {
            if (vertNormIdx[i] >= nbn)
                nbn = vertNormIdx[i]+1;
        }
        vector<sofa::Index> normVertIdx(nbn);
        for (sofa::Index i = 0; i < vertNormIdx.size(); i++)
        {
            normVertIdx[vertNormIdx[i]]=i;
        }
        for (sofa::Index i = 0; i < nbn; i++)
        {
            sofa::Index j = normVertIdx[i];
            *out << "vn "<< std::fixed << vnormals[j][0]<<' '<< std::fixed <<vnormals[j][1]<<' '<< std::fixed <<vnormals[j][2]<<'\n';
        }
    }

    Size nbt = 0;
    if (!vtexcoords.empty())
    {
        nbt = sofa::Size(vtexcoords.size());
        for (std::size_t i=0; i<nbt; i++)
        {
            *out << "vt "<< std::fixed << vtexcoords[i][0]<<' '<< std::fixed <<vtexcoords[i][1]<<'\n';
        }
    }

    for (std::size_t i = 0; i < edges.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<2; j++)
        {
            sofa::Index i0 = edges[i][j];
            sofa::Index i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            sofa::Index i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    for (std::size_t i = 0; i < triangles.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<3; j++)
        {
            sofa::Index i0 = triangles[i][j];
            sofa::Index i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            sofa::Index i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    for (std::size_t i = 0; i < quads.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<4; j++)
        {
            sofa::Index i0 = quads[i][j];
            sofa::Index i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            sofa::Index i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    *out << '\n';
    vindex+=nbv;
    nindex+=nbn;
    tindex+=nbt;
}

} // namespace sofa::component::visual
