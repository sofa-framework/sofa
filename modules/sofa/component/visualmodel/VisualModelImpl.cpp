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
#include <sofa/component/visualmodel/VisualModelImpl.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/component/topology/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/QuadSetTopologyModifier.h>
#include <sofa/component/topology/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/HexahedronSetTopologyModifier.h>

#include <sofa/component/topology/PointSetTopologyChange.h>
#include <sofa/component/topology/TriangleSetTopologyChange.h>
#include <sofa/component/topology/QuadSetTopologyChange.h>
#include <sofa/component/topology/TetrahedronSetTopologyChange.h>
#include <sofa/component/topology/HexahedronSetTopologyChange.h>

#include <sofa/component/topology/SparseGridTopology.h>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/gl/RAII.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/io/MeshOBJ.h>
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

void VisualModelImpl::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::visual::VisualModel::parse(arg);
    VisualModelImpl* obj = this;

    if (arg->getAttribute("normals")!=NULL)
        obj->setUseNormals(atoi(arg->getAttribute("normals"))!=0);

    if (arg->getAttribute("castshadow")!=NULL)
        obj->setCastShadow(atoi(arg->getAttribute("castshadow"))!=0);

    if (arg->getAttribute("flip")!=NULL)
        obj->flipFaces();

    if (arg->getAttribute("color"))
        obj->setColor(arg->getAttribute("color"));

    if (arg->getAttribute("su")!=NULL || arg->getAttribute("sv")!=NULL)
        m_scaleTex = TexCoord((float)atof(arg->getAttribute("su","1.0")),(float)atof(arg->getAttribute("sv","1.0")));

    if (arg->getAttribute("du")!=NULL || arg->getAttribute("dv")!=NULL)
        m_translationTex = TexCoord((float)atof(arg->getAttribute("du","0.0")),(float)atof(arg->getAttribute("dv","0.0")));

    if (arg->getAttribute("rx")!=NULL || arg->getAttribute("ry")!=NULL || arg->getAttribute("rz")!=NULL)
        m_rotation.setValue(Vector3((SReal)(atof(arg->getAttribute("rx","0.0"))),(SReal)(atof(arg->getAttribute("ry","0.0"))),(SReal)(atof(arg->getAttribute("rz","0.0")))));

    if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
        m_translation.setValue(Vector3((SReal)atof(arg->getAttribute("dx","0.0")), (SReal)atof(arg->getAttribute("dy","0.0")), (SReal)atof(arg->getAttribute("dz","0.0"))));

    if (arg->getAttribute("scale")!=NULL)
    {
        m_scale.setValue(Vector3((SReal)atof(arg->getAttribute("scale","1.0")), (SReal)atof(arg->getAttribute("scale","1.0")), (SReal)atof(arg->getAttribute("scale","1.0"))));
    }
    else if (arg->getAttribute("sx")!=NULL || arg->getAttribute("sy")!=NULL || arg->getAttribute("sz")!=NULL)
    {
        m_scale.setValue(Vector3((SReal)atof(arg->getAttribute("sx","1.0")), (SReal)atof(arg->getAttribute("sy","1.0")), (SReal)atof(arg->getAttribute("sz","1.0"))));
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
    , m_useNormals		(initData	(&m_useNormals, true, "useNormals", "True if normal smoothing groups should be read from file"))
    , m_updateNormals   (initData   (&m_updateNormals, true, "updateNormals", "True if normals should be updated at each iteration"))
    , m_computeTangents (initData   (&m_computeTangents, false, "computeTangents", "True if tangents should be computed at startup"))
    , m_updateTangents  (initData   (&m_updateTangents, true, "updateTangents", "True if tangents should be updated at each iteration"))
    , m_vertices		(initData   (&m_vertices, "vertices", "vertices of the model"))
    , m_vtexcoords		(initData   (&m_vtexcoords, "texcoords", "coordinates of the texture"))
    , m_vtangents		(initData   (&m_vtangents, "tangents", "tangents for normal mapping"))
    , m_vbitangents		(initData   (&m_vbitangents, "bitangents", "tangents for normal mapping"))
    , m_triangles		(initData   (&m_triangles, "triangles", "triangles of the model"))
    , m_quads			(initData   (&m_quads, "quads", "quads of the model"))
    , fileMesh          (initData   (&fileMesh, "fileMesh"," Path to the model"))
    , texturename       (initData   (&texturename, "texturename", "Name of the Texture"))
    , m_translation     (initData   (&m_translation, Vector3(), "translation", "Initial Translation of the object"))
    , m_rotation        (initData   (&m_rotation, Vector3(), "rotation", "Initial Rotation of the object"))
    , m_scale           (initData   (&m_scale, Vector3(1.0,1.0,1.0), "scale3d", "Initial Scale of the object"))
    , m_scaleTex        (initData   (&m_scaleTex, TexCoord(1.0,1.0), "scaleTex", "Scale of the texture"))
    , m_translationTex  (initData   (&m_translationTex, TexCoord(1.0,1.0), "translationTex", "Translation of the texture"))
#ifdef SOFA_SMP
    , previousProcessorColor(false)
#endif
    , material			(initData	(&material, "material", "Material")) // tex(NULL)
    , putOnlyTexCoords	(initData	(&putOnlyTexCoords, (bool) false, "putOnlyTexCoords", "Give Texture Coordinates without the texture binding"))
    , srgbTexturing		(initData	(&srgbTexturing, (bool) false, "srgbTexturing", "When sRGB rendering is enabled, is the texture in sRGB colorspace?"))
    , materials			(initData	(&materials, "materials", "List of materials"))
    , groups			(initData	(&groups, "groups", "Groups of triangles and quads using a given material"))
{
#ifdef SOFA_SMP
    originalMaterial = material.getValue();
#endif

    m_topology = 0;

    material.setDisplayed(false);
    addAlias(&fileMesh, "filename");

    m_vertices		.setGroup("Vector");
    m_vnormals		.setGroup("Vector");
    m_vtexcoords	.setGroup("Vector");
    m_vtangents		.setGroup("Vector");
    m_vbitangents	.setGroup("Vector");
    m_triangles		.setGroup("Vector");
    m_quads			.setGroup("Vector");

    m_translation	.setGroup("Transformation");
    m_rotation		.setGroup("Transformation");
    m_scale			.setGroup("Transformation");

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
#ifdef SOFA_SMP
        originalMaterial=M;
#endif
    }

    if (!objLoader.getGroups().empty())
    {
        // Get informations about the multiple materials
        helper::WriteAccessor< Data< helper::vector<Material> > > materials = this->materials;
        helper::WriteAccessor< Data< helper::vector<FaceGroup> > > groups = this->groups;
        materials.resize(objLoader.getMaterials().size());
        for (unsigned i=0; i<materials.size(); ++i)
            materials[i] = objLoader.getMaterials()[i];

        // compute the triangle and quad index corresponding to each facet
        // convert the groups info
        int nbt = 0, nbq = 0;
        helper::vector< std::pair<int, int> > facet2tq;
        facet2tq.resize(facetsImport.size()+1);
        for (unsigned int i = 0; i < facetsImport.size(); i++)
        {
            facet2tq[i] = std::make_pair(nbt, nbq);
            const vector<vector <int> >& vertNormTexIndex = facetsImport[i];
            const vector<int>& verts = vertNormTexIndex[0];
            if (verts.size() < 3)
                ; // ignore lines
            else if (verts.size() == 4)
                nbq += 1;
            else
                nbt += verts.size()-2;
        }
        facet2tq[facetsImport.size()] = std::make_pair(nbt, nbq);
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
            g.tri0 = facet2tq[g0.p0].first;
            g.nbt = facet2tq[g0.p0+g0.nbp].first - g.tri0;
            g.quad0 = facet2tq[g0.p0].second;
            g.nbq = facet2tq[g0.p0+g0.nbp].second - g.quad0;
            if (g.materialId == -1 && !g0.materialName.empty())
                serr << "face group " << ig << " name " << g0.materialName << " uses missing material " << g0.materialName << sendl;
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
        if (vertNormTexIndex[0].size() < 3) continue; // ignore lines
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

    sout << nbVIn << " input positions, " << nbVOut << " final vertices." << sendl;

    if (nbVIn != nbVOut)
        vsplit = true;

    // Then we can create the final arrays
    ResizableExtVector<Coord>& positions = *(m_positions.beginEdit());
    ResizableExtVector<Coord>& vertices = *(m_vertices.beginEdit());
    ResizableExtVector<Deriv>& vnormals = *(m_vnormals.beginEdit());
    ResizableExtVector<TexCoord>& vtexcoords = *(m_vtexcoords.beginEdit());

    positions.resize(nbVIn);

    if (vsplit)
    {
        vertices.resize(nbVOut);
        vnormals.resize(nbVOut);
        vtexcoords.resize(nbVOut);
        vertPosIdx.resize(nbVOut);
        vertNormIdx.resize(nbVOut);
    }
    else
    {
        vertices.resize(nbVIn);
        vnormals.resize(nbVIn);
        vtexcoords.resize(nbVIn);
    }

    int nbNOut = 0; /// Number of different normals
    for (int i = 0, j = 0; i < nbVIn; i++)
    {
        positions[i] = verticesImport[i];

        std::map<int, int> normMap;
        for (std::map<std::pair<int, int>, int>::iterator it = vertTexNormMap[i].begin();
                it != vertTexNormMap[i].end(); ++it)
        {
            vertices[j] = verticesImport[i];
            int t = it->first.first;
            int n = it->first.second;
            if ((unsigned)n < normalsImport.size())
                vnormals[j] = normalsImport[n];
            if ((unsigned)t < texCoordsImport.size())
                vtexcoords[j] = texCoordsImport[t];

            if (vsplit)
            {
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

    if (!vsplit)
        nbNOut = nbVOut;
    else if (nbNOut == nbVOut)
        vertNormIdx.resize(0);


    m_vertices.endEdit();
    m_vnormals.endEdit();
    m_vtexcoords.endEdit();
    m_positions.endEdit();

    // Then we create the triangles and quads
    ResizableExtVector< Triangle >& triangles = *(m_triangles.beginEdit());
    ResizableExtVector< Quad >& quads = *(m_quads.beginEdit());

    for (unsigned int i = 0; i < facetsImport.size(); i++)
    {
        const vector<vector <int> >& vertNormTexIndex = facetsImport[i];
        if (vertNormTexIndex[0].size() < 3) continue; // ignore lines
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
                serr << "ERROR(VisualModelImpl): index "<<idxs[j]<<" out of range"<<sendl;
                idxs[j] = 0;
            }
        }

        if (verts.size() == 4)
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

    m_triangles.endEdit();
    m_quads.endEdit();

    computeNormals();
    computeTangents();
    computeBBox();
}

bool VisualModelImpl::load(const std::string& filename, const std::string& loader, const std::string& textureName)
{
    bool tex = !textureName.empty() || putOnlyTexCoords.getValue();
    if (!textureName.empty())
    {
        std::string textureFilename(textureName);
        if (sofa::helper::system::DataRepository.findFile(textureFilename))
        {
            std::cout << "loading file " << textureName << std::endl;
            tex = loadTexture(textureName);
        }
        else
            serr << "Texture \"" << textureName << "\" not found" << sendl;
    }

    // Make sure all Data are up-to-date
    m_vertices.updateIfDirty();
    m_vnormals.updateIfDirty();
    m_vtexcoords.updateIfDirty();
    m_vtangents.updateIfDirty();
    m_vbitangents.updateIfDirty();
    m_triangles.updateIfDirty();
    m_quads.updateIfDirty();

    if (!filename.empty() && (m_vertices.getValue()).size() == 0)
    {
        std::string meshFilename(filename);
        if (sofa::helper::system::DataRepository.findFile(meshFilename))
        {
            //name = filename;
            std::auto_ptr<helper::io::Mesh> objLoader;
            if (loader.empty())
            {
                objLoader.reset(helper::io::Mesh::Create(filename));
            }
            else
            {
                objLoader.reset(helper::io::Mesh::Create(loader, filename));
            }

            if (objLoader.get() == 0)
            {
                return false;
            }
            else
            {
                //Modified: previously, the texture coordinates were not loaded correctly if no texture name was specified.
                //setMesh(*objLoader,tex);
                setMesh(*objLoader, true);
                //sout << "VisualModel::load, vertices.size = "<< vertices.size() <<sendl;
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
            serr << "Mesh \"" << filename << "\" not found" << sendl;
        }
    }
    else
    {
        if ((m_vertices.getValue()).size() == 0)
        {
            sout << "VisualModel: will use Topology." << sendl;
            useTopology = true;
        }
        else
        {
            computeBBox();
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

void VisualModelImpl::applyTranslation(const double dx, const double dy, const double dz)
{
    Vector3 d((GLfloat)dx,(GLfloat)dy,(GLfloat)dz);

    Data< ResizableExtVector<Coord> >* d_x = this->write(core::VecCoordId::position());
    ResizableExtVector<Coord> &x = *d_x->beginEdit();

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i] += d;
    }

    d_x->endEdit();

    updateVisual();
}

void VisualModelImpl::applyRotation(const double rx, const double ry, const double rz)
{
    Quaternion q = helper::Quater<SReal>::createQuaterFromEuler( Vec<3,SReal>(rx,ry,rz)*M_PI/180.0);
    applyRotation(q);
}

void VisualModelImpl::applyRotation(const Quat q)
{
    Data< ResizableExtVector<Coord> >* d_x = this->write(core::VecCoordId::position());
    ResizableExtVector<Coord> &x = *d_x->beginEdit();

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i] = q.rotate(x[i]);
    }

    d_x->endEdit();

    updateVisual();
}

void VisualModelImpl::applyScale(const double sx, const double sy, const double sz)
{
    Data< ResizableExtVector<Coord> >* d_x = this->write(core::VecCoordId::position());
    ResizableExtVector<Coord> &x = *d_x->beginEdit();

    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i][0] *= (GLfloat) sx;
        x[i][1] *= (GLfloat) sy;
        x[i][2] *= (GLfloat) sz;
    }

    d_x->endEdit();

    updateVisual();
}

void VisualModelImpl::applyUVTranslation(const double dU, const double dV)
{
    ResizableExtVector<TexCoord>& vtexcoords = *(m_vtexcoords.beginEdit());
    for (unsigned int i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] += (GLfloat) dU;
        vtexcoords[i][1] += (GLfloat) dV;
    }
    m_vtexcoords.endEdit();
}

void VisualModelImpl::applyUVScale(const double scaleU, const double scaleV)
{
    ResizableExtVector<TexCoord>& vtexcoords = *(m_vtexcoords.beginEdit());
    for (unsigned int i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] *= (GLfloat) scaleU;
        vtexcoords[i][1] *= (GLfloat) scaleV;
    }
    m_vtexcoords.endEdit();
}

void VisualModelImpl::init()
{
    load(fileMesh.getFullPath(), "", texturename.getFullPath());
    m_topology = getContext()->getMeshTopology();

    if (m_topology == 0)
    {
        // Fixes bug when neither an .obj file nor a topology is present in the VisualModel Node.
        // Thus nothing will be displayed.
        useTopology = false;
    }

    m_vertices.beginEdit();
    m_vnormals.beginEdit();
    m_vtexcoords.beginEdit();
    m_vtangents.beginEdit();
    m_vbitangents.beginEdit();
    m_triangles.beginEdit();
    m_quads.beginEdit();

    applyScale(m_scale.getValue()[0], m_scale.getValue()[1], m_scale.getValue()[2]);
    applyRotation(m_rotation.getValue()[0], m_rotation.getValue()[1], m_rotation.getValue()[2]);
    applyTranslation(m_translation.getValue()[0], m_translation.getValue()[1], m_translation.getValue()[2]);


    m_translation.setValue(Vector3());
    m_rotation.setValue(Vector3());
    m_scale.setValue(Vector3(1,1,1));

    VisualModel::init();
    updateVisual();
}

void VisualModelImpl::computeNormals()
{
    const ResizableExtVector<Coord>& vertices = getVertices();
    //const ResizableExtVector<Coord>& vertices = m_vertices.getValue();
    if (vertices.empty() || (!m_updateNormals.getValue() && (m_vnormals.getValue()).size() != (vertices).size())) return;

    const ResizableExtVector<Triangle>& triangles = m_triangles.getValue();
    const ResizableExtVector<Quad>& quads = m_quads.getValue();

    if (vertNormIdx.empty())
    {
        int nbn = (vertices).size();
        //serr << "CN0("<<nbn<<")"<<sendl;

        ResizableExtVector<Deriv>& normals = *(m_vnormals.beginEdit());

        normals.resize(nbn);
        for (int i = 0; i < nbn; i++)
            normals[i].clear();

        for (unsigned int i = 0; i < triangles.size(); i++)
        {
            const Coord v1 = vertices[triangles[i][0]];
            const Coord v2 = vertices[triangles[i][1]];
            const Coord v3 = vertices[triangles[i][2]];
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
        //serr << "CN1("<<nbn<<")"<<sendl;

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
    const ResizableExtVector<Coord>& vertices = m_vertices.getValue();
    const ResizableExtVector<TexCoord>& texcoords = m_vtexcoords.getValue();
    ResizableExtVector<Coord>& tangents = *(m_vtangents.beginEdit());
    ResizableExtVector<Coord>& bitangents = *(m_vbitangents.beginEdit());

    tangents.resize(vertices.size());
    bitangents.resize(vertices.size());

    for (unsigned i = 0; i < vertices.size(); i++)
    {
        tangents[i].clear();
        bitangents[i].clear();
    }

    for (unsigned int i = 0; i < triangles.size() ; i++)
    {
        const Coord v1 = vertices[triangles[i][0]];
        const Coord v2 = vertices[triangles[i][1]];
        const Coord v3 = vertices[triangles[i][2]];
        const TexCoord t1 = texcoords[triangles[i][0]];
        const TexCoord t2 = texcoords[triangles[i][1]];
        const TexCoord t3 = texcoords[triangles[i][2]];
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
        tangents[i].normalize();
        bitangents[i].normalize();
    }
    m_vtangents.endEdit();
    m_vbitangents.endEdit();
}

void VisualModelImpl::computeBBox()
{
    const VecCoord& x = m_vertices.getValue();
    Vec3f minBBox(1e10,1e10,1e10);
    Vec3f maxBBox(-1e10,-1e10,-1e10);
    for (unsigned int i = 0; i < x.size(); i++)
    {
        const Coord& p = x[i];
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    bbox[0] = minBBox;
    bbox[1] = maxBBox;
}

bool VisualModelImpl::addBBox(double* minBBox, double* maxBBox)
{
    if (bbox[0][0] > bbox[1][0]) return false;
    for (unsigned int i=0; i<xforms.size(); i++)
    {
        const Vec3f& center = xforms[i].getCenter();
        const Quat& orientation = xforms[i].getOrientation();
        for (int j=0; j<8; j++)
        {
            Coord p ( bbox[(j>>0)&1][0], bbox[(j>>1)&1][1], bbox[(j>>2)&1][2]);
            p = orientation.rotate(p) + center;
            for (int c=0; c<3; c++)
            {
                if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
                if (p[c] < minBBox[c]) minBBox[c] = p[c];
            }
        }
    }
    return true;
}

void VisualModelImpl::flipFaces()
{
    ResizableExtVector<Deriv>& vnormals = *(m_vnormals.beginEdit());
    ResizableExtVector<Triangle>& triangles = *(m_triangles.beginEdit());
    ResizableExtVector<Quad>& quads = *(m_quads.beginEdit());

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
    m_triangles.endEdit();
    m_quads.endEdit();
}

void VisualModelImpl::setColor(float r, float g, float b, float a)
{
    Material M = material.getValue();
    M.setColor(r,g,b,a);
    material.setValue(M);
#ifdef SOFA_SMP
    originalMaterial=M;
#endif
}

static int hexval(char c)
{
    if (c>='0' && c<='9') return c-'0';
    else if (c>='a' && c<='f') return (c-'a')+10;
    else if (c>='A' && c<='F') return (c-'A')+10;
    else return 0;
}

void VisualModelImpl::setColor(std::string color)
{
    if (color.empty()) return;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
    if (color[0]>='0' && color[0]<='9')
    {
        sscanf(color.c_str(),"%f %f %f %f", &r, &g, &b, &a);
    }
    else if (color[0]=='#' && color.length()>=7)
    {
        r = (hexval(color[1])*16+hexval(color[2]))/255.0f;
        g = (hexval(color[3])*16+hexval(color[4]))/255.0f;
        b = (hexval(color[5])*16+hexval(color[6]))/255.0f;
        if (color.length()>=9)
            a = (hexval(color[7])*16+hexval(color[8]))/255.0f;
    }
    else if (color[0]=='#' && color.length()>=4)
    {
        r = (hexval(color[1])*17)/255.0f;
        g = (hexval(color[2])*17)/255.0f;
        b = (hexval(color[3])*17)/255.0f;
        if (color.length()>=5)
            a = (hexval(color[4])*17)/255.0f;
    }
    else if (color == "white")    { r = 1.0f; g = 1.0f; b = 1.0f; }
    else if (color == "black")    { r = 0.0f; g = 0.0f; b = 0.0f; }
    else if (color == "red")      { r = 1.0f; g = 0.0f; b = 0.0f; }
    else if (color == "green")    { r = 0.0f; g = 1.0f; b = 0.0f; }
    else if (color == "blue")     { r = 0.0f; g = 0.0f; b = 1.0f; }
    else if (color == "cyan")     { r = 0.0f; g = 1.0f; b = 1.0f; }
    else if (color == "magenta")  { r = 1.0f; g = 0.0f; b = 1.0f; }
    else if (color == "yellow")   { r = 1.0f; g = 1.0f; b = 0.0f; }
    else if (color == "gray")     { r = 0.5f; g = 0.5f; b = 0.5f; }
    else
    {
        serr << "Unknown color "<<color<<sendl;
        return;
    }
    setColor(r,g,b,a);
}

#ifdef SOFA_SMP
struct colors
{
    float r;
    float g;
    float b;
};
static colors colorTab[]=
{
    {1.0f,0.0f,0.0f},
    {1.0f,1.0f,0.0f},
    {0.0f,1.0f,0.0f},
    {0.0f,1.0f,1.0f},
    {0.0f,0.0f,1.0f},
    {0.5f,.5f,.5f},
    {0.5f,0.0f,0.0f},
    {.5f,.5f,0.0f},
    {0.0f,1.0f,0.0f},
    {0.0f,1.0f,1.0f},
    {0.0f,0.0f,1.0f},
    {0.5f,.5f,.5f}
};
#endif

void VisualModelImpl::updateVisual()
{
#ifdef SOFA_SMP
    modified = true;
#endif

    if (modified && (!(m_vertices.getValue()).empty() || useTopology))
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
        computeNormals();
        if (m_updateTangents.getValue())
            computeTangents();
        computeBBox();
        updateBuffers();
        modified = false;
    }

    m_vertices.updateIfDirty();
    m_vnormals.updateIfDirty();
    m_vtexcoords.updateIfDirty();
    m_vtangents.updateIfDirty();
    m_vbitangents.updateIfDirty();
    m_triangles.updateIfDirty();
    m_quads.updateIfDirty();

#ifdef SOFA_SMP

    if(vparams->displayFlags().getShowProcessorColor())
    {
        sofa::core::objectmodel::Context *context=dynamic_cast<sofa::core::objectmodel::Context *>(this->getContext());
        if(context&&context->getPartition())
        {

            if(context->getPartition()->getThread()&&context->getPartition()->getThread()->get_processor())
            {
                unsigned int proc =context->getPartition()->getThread()->get_processor()->get_pid();
                this->setColor(colorTab[proc].r,colorTab[proc].g,colorTab[proc].b,1.0f);
            }
            else if(context->getPartition()->getCPU()!=-1)
            {
                unsigned int proc=context->getPartition()->getCPU();
                this->setColor(colorTab[proc].r,colorTab[proc].g,colorTab[proc].b,1.0f);

            }

        }
    }

    if(previousProcessorColor&&!vparams->displayFlags().getShowProcessorColor())
    {
        material.setValue(originalMaterial);
    }
    previousProcessorColor=vparams->displayFlags().getShowProcessorColor();
#endif
}


void VisualModelImpl::computePositions()
{
    if (!vertPosIdx.empty())
    {
        // Need to transfer positions
        ResizableExtVector<Coord>& vertices = *(m_vertices.beginEdit());
        const ResizableExtVector<Coord>& positions = this->m_positions.getValue();

        for (unsigned int i=0 ; i < vertices.size(); ++i)
            vertices[i] = positions[vertPosIdx[i]];

        m_vertices.endEdit();
    }
}

void VisualModelImpl::computeMesh()
{
    using sofa::component::topology::SparseGridTopology;
    using sofa::core::behavior::BaseMechanicalState;

    if ((m_vertices.getValue()).empty())
    {
        ResizableExtVector<Coord>& vertices = *(m_vertices.beginEdit());

        if (m_topology->hasPos())
        {
            if (SparseGridTopology *spTopo = dynamic_cast< SparseGridTopology *>(m_topology))
            {
                sout << "VisualModel: getting marching cube mesh from topology : ";
                sofa::helper::io::Mesh m;
                spTopo->getMesh(m);
                setMesh(m, !texturename.getValue().empty());
                sout << m.getVertices().size() << " points, " << m.getFacets().size()  << " triangles." << sendl;
                useTopology = false; //visual model needs to be created only once at initial time
                return;
            }

            if (this->f_printLog.getValue())
                sout << "VisualModel: copying " << m_topology->getNbPoints() << " points from topology." << sendl;

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
            BaseMechanicalState* mstate = dynamic_cast< BaseMechanicalState* >(m_topology->getContext()->getMechanicalState());

            if (mstate)
            {
                if (this->f_printLog.getValue())
                    sout << "VisualModel: copying " << mstate->getSize() << " points from mechanical state." << sendl;

                vertices.resize(mstate->getSize());

                for (unsigned int i=0; i<vertices.size(); i++)
                {
                    vertices[i][0] = (Real)mstate->getPX(i);
                    vertices[i][1] = (Real)mstate->getPY(i);
                    vertices[i][2] = (Real)mstate->getPZ(i);
                }
            }
        }
        m_vertices.endEdit();
    }

    lastMeshRev = m_topology->getRevision();

    const vector< Triangle >& inputTriangles = m_topology->getTriangles();

    if (this->f_printLog.getValue())
        sout << "VisualModel: copying " << inputTriangles.size() << " triangles from topology." << sendl;

    ResizableExtVector< Triangle >& triangles = *(m_triangles.beginEdit());
    triangles.resize(inputTriangles.size());

    for (unsigned int i=0; i<triangles.size(); ++i)
    {
        triangles[i] = inputTriangles[i];
    }
    m_triangles.endEdit();

    const vector< BaseMeshTopology::Quad >& inputQuads = m_topology->getQuads();

    if (this->f_printLog.getValue())
        sout << "VisualModel: copying " << inputQuads.size()<< " quads from topology." << sendl;

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
    ResizableExtVector<Coord>& vertices = *(m_vertices.beginEdit());


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

            const sofa::component::topology::TrianglesAdded *ta = static_cast< const sofa::component::topology::TrianglesAdded * >( *itBegin );
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

            const sofa::component::topology::QuadsAdded *qa = static_cast< const sofa::component::topology::QuadsAdded * >( *itBegin );
            Quad q;
            const unsigned int nbAddedQuads = qa->getNbAddedQuads();
            const unsigned int nbQuaduads = triangles.size();
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
            unsigned int ind_last;

            last = m_topology->getNbTriangles() - 1;

            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const sofa::component::topology::TrianglesRemoved *>( *itBegin ) )->getArray();

            Triangle tmp;

            for (unsigned int i = 0; i <tab.size(); ++i)
            {
                unsigned int ind_k = tab[i];

                tmp = triangles[ind_k];
                triangles[ind_k] = triangles[last];
                triangles[last] = tmp;

                ind_last = triangles.size() - 1;

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
            unsigned int ind_last;

            last = m_topology->getNbQuads() - 1;

            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const sofa::component::topology::QuadsRemoved *>( *itBegin ) )->getArray();

            Quad tmp;

            for (unsigned int i = 0; i <tab.size(); ++i)
            {
                unsigned int ind_k = tab[i];

                tmp = quads[ind_k];
                quads[ind_k] = quads[last];
                quads[last] = tmp;

                ind_last = quads.size() - 1;

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

                const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

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
                                    sout << "INFO_print : Vis - triangle is forgotten in SHELL !!! global indices (point, triangle) = ( "  << last << " , " << ind_forgotten  << " )" << sendl;

                                    if(ind_forgotten<m_topology->getNbTriangles())
                                    {
                                        const sofa::component::topology::Triangle t_forgotten = m_topology->getTriangle(ind_forgotten);
                                        sout << "INFO_print : Vis - last = " << last << sendl;
                                        sout << "INFO_print : Vis - lastIndexVec[i] = " << lastIndexVec[i] << sendl;
                                        sout << "INFO_print : Vis - tab.size() = " << tab.size() << " , tab = " << tab << sendl;
                                        sout << "INFO_print : Vis - t_local rectified = " << triangles[j_loc] << sendl;
                                        sout << "INFO_print : Vis - t_global = " << t_forgotten << sendl;
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

                const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

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

                const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::component::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

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

                const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::component::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

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
                if (this->f_printLog.getValue())
                {
                    sout << "VisualModel: oldsize    " << this->getSize()  << sendl;
                    sout << "VisualModel: copying " << mstate->getSize() << " points from mechanical state." << sendl;
                }

                vertices.resize(mstate->getSize());

                for (unsigned int i=0; i<vertices.size(); i++)
                {
                    vertices[i][0] = (Real)mstate->getPX(i);
                    vertices[i][1] = (Real)mstate->getPY(i);
                    vertices[i][2] = (Real)mstate->getPZ(i);
                }
            }
            updateVisual();
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
    m_vertices.endEdit();
}

void VisualModelImpl::initVisual()
{
    //if (tex)
    //{
    //    tex->init();
    //}
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

    const ResizableExtVector<Coord>& x = m_positions.getValue();
    const ResizableExtVector<Deriv>& vnormals = m_vnormals.getValue();
    const ResizableExtVector<TexCoord>& vtexcoords = m_vtexcoords.getValue();
    const ResizableExtVector<Triangle>& triangles = m_triangles.getValue();
    const ResizableExtVector<Quad>& quads = m_quads.getValue();

    int nbv = x.size();

    for (unsigned int i=0; i<x.size(); i++)
    {
        *out << "v "<< std::fixed << x[i][0]<<' '<< std::fixed <<x[i][1]<<' '<< std::fixed <<x[i][2]<<'\n';
    }

    int nbn = 0;

    if (vertNormIdx.empty())
    {
        nbn = vnormals.size();
        for (unsigned int i=0; i<vnormals.size(); i++)
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
        for (unsigned int i=0; i<vtexcoords.size(); i++)
        {
            *out << "vt "<< std::fixed << vtexcoords[i][0]<<' '<< std::fixed <<vtexcoords[i][1]<<'\n';
        }
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

} // namespace visualmodel

} // namespace component

} // namespace sofa

