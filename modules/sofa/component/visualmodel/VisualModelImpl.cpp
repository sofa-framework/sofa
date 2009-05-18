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
#include <sofa/helper/rmath.h>
#include <sstream>

#include <map>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::topology;

void VisualModelImpl::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::VisualModel::parse(arg);
    VisualModelImpl* obj = this;

    if (arg->getAttribute("normals")!=NULL)
        obj->setUseNormals(atoi(arg->getAttribute("normals"))!=0);

    if (arg->getAttribute("castshadow")!=NULL)
        obj->setCastShadow(atoi(arg->getAttribute("castshadow"))!=0);
    /*
        std::string file;
         file=(arg->getAttribute("texturename",""));
         if (!file.empty())
         {
              texturename.setValue( sofa::helper::system::DataRepository.getFile ( file ));
         }
    */

    if (arg->getAttribute("flip")!=NULL)
    {
        obj->flipFaces();
    }
    if (arg->getAttribute("color"))
    {
        obj->setColor(arg->getAttribute("color"));
    }


    if (arg->getAttribute("su")!=NULL || arg->getAttribute("sv")!=NULL)
    {
        scaleTex = TexCoord((float)atof(arg->getAttribute("su","1.0")),(float)atof(arg->getAttribute("sv","1.0")));
    }
    if (arg->getAttribute("du")!=NULL || arg->getAttribute("dv")!=NULL)
    {
        translationTex = TexCoord((float)atof(arg->getAttribute("du","0.0")),(float)atof(arg->getAttribute("dv","0.0")));
    }

    if (arg->getAttribute("rx")!=NULL || arg->getAttribute("ry")!=NULL || arg->getAttribute("rz")!=NULL)
    {
        rotation.setValue(Vector3((SReal)(atof(arg->getAttribute("rx","0.0"))),(SReal)(atof(arg->getAttribute("ry","0.0"))),(SReal)(atof(arg->getAttribute("rz","0.0")))));
    }
    if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
    {
        translation.setValue(Vector3((SReal)atof(arg->getAttribute("dx","0.0")), (SReal)atof(arg->getAttribute("dy","0.0")), (SReal)atof(arg->getAttribute("dz","0.0"))));
    }
}

SOFA_DECL_CLASS(VisualModelImpl)

int VisualModelImplClass = core::RegisterObject("Generic visual model. If a viewer is active it will replace the VisualModel alias, otherwise nothing will be displayed.")
        .add< VisualModelImpl >()
        .addAlias("VisualModel")
        ;

VisualModelImpl::VisualModelImpl() //const std::string &name, std::string filename, std::string loader, std::string textureName)
    :  useTopology(false), lastMeshRev(-1), useNormals(true), castShadow(true),

       field_vertices    (initDataPtr(&field_vertices,&vertices,  "position",   "vertices of the model") ),
       field_vnormals    (initDataPtr(&field_vnormals, &vnormals, "normals",   "normals of the model") ),
       field_vtexcoords  (initDataPtr(&field_vtexcoords, &vtexcoords, "texcoords",  "coordinates of the texture") ),
       field_triangles   (initDataPtr(&field_triangles, &triangles,"triangles" ,  "triangles of the model") ),
       field_quads       (initDataPtr(&field_quads, &quads,   "quads",    "quads of the model") ),
       fileMesh          (initData   (&fileMesh,    "fileMesh","Path to the model")),
       texturename       (initData                            (&texturename, "texturename","Name of the Texture")),
       translation       (initData   (&translation, Vector3(), "translation", "Initial Translation of the object")),
       rotation          (initData   (&rotation, Vector3(), "rotation", "Initial Rotation of the object")),
       scale             (initData   (&scale, (SReal)(1.0), "scale", "Initial Scale of the object")),
       scaleTex          (initData   (&scaleTex, TexCoord(1.0,1.0), "scaleTex", "Scale of the texture")),
       translationTex    (initData   (&translationTex, TexCoord(1.0,1.0), "translationTex", "Translation of the texture")),
       material(initData(&material,"material","Material")), // tex(NULL)
       putOnlyTexCoords(initData(&putOnlyTexCoords, (bool) false, "putOnlyTexCoords", "Give Texture Coordinates without the texture binding"))
{
    inputVertices = &vertices;
    addAlias(&fileMesh, "filename");
}

VisualModelImpl::~VisualModelImpl()
{
    if (inputVertices != &vertices) delete inputVertices;
}

bool VisualModelImpl::isTransparent()
{
    return (material.getValue().useDiffuse && material.getValue().diffuse[3] < 1.0);
}

void VisualModelImpl::drawVisual()
{
    if (!isTransparent())
        internalDraw();
}

void VisualModelImpl::drawTransparent()
{
    if (isTransparent())
        internalDraw();
}

void VisualModelImpl::drawShadow()
{
    if (!isTransparent() && getCastShadow())
    {
        //sout << "drawShadow for "<<getName()<<sendl;
        internalDraw();
    }
}

void VisualModelImpl::setMesh(helper::io::Mesh &objLoader, bool tex)
{
    const vector< vector< vector<int> > > &facetsImport = objLoader.getFacets();
    const vector<Vector3> &verticesImport = objLoader.getVertices();
    const vector<Vector3> &normalsImport = objLoader.getNormals();
    const vector<Vector3> &texCoordsImport = objLoader.getTexCoords();

    const helper::io::Mesh::Material &materialImport = objLoader.getMaterial();

    if (!material.isSet() && materialImport.activated)
    {
        helper::io::Mesh::Material M;
        M = materialImport;
        material.setValue(M);
    }

//             sout << "Vertices Import size : " << verticesImport.size() << " (" << normalsImport.size() << " normals)." << sendl;

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
            vertTexNormMap[verts[j]][std::make_pair((tex?texs[j]:-1), (useNormals?norms[j]:0))] = 0;
        }
    }

    // Then we can compute how many vertices are created
    int nbVOut = 0;
    bool vsplit = false;
    for (int i = 0; i < nbVIn; i++)
    {
        int s = vertTexNormMap[i].size();
        nbVOut += s;
        if (s!=1)
            vsplit = true;
    }

    // Then we can create the final arrays

    vertices.resize(nbVOut);
    vnormals.resize(nbVOut);

    //if (tex)
    vtexcoords.resize(nbVOut);

    if (vsplit)
    {
        inputVertices = new ResizableExtVector<Coord>;
        inputVertices->resize(nbVIn);
        vertPosIdx.resize(nbVOut);
        vertNormIdx.resize(nbVOut);
    }
    else
        inputVertices = &vertices;

    int nbNOut = 0; /// Number of different normals
    for (int i = 0, j = 0; i < nbVIn; i++)
    {
        if (vsplit)
            (*inputVertices)[i] = verticesImport[i];
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
                    vertNormIdx[j] = normMap[n] = nbNOut++;
            }
            it->second = j++;
        }
    }
    if (!vsplit) nbNOut = nbVOut;
    else if (nbNOut == nbVOut) vertNormIdx.resize(0);

//             sout << "Vertices Export size : " << nbVOut << " (" << nbNOut << " normals)." << sendl;

//             sout << "Facets Import size : " << facetsImport.size() << sendl;

    // Then we create the triangles and quads

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
            idxs[j] = vertTexNormMap[verts[j]][std::make_pair((tex?texs[j]:-1), (useNormals?norms[j]:0))];
            if ((unsigned)idxs[j] >= (unsigned)nbVOut)
            {
                serr << "ERROR(VisualModelImpl): index "<<idxs[j]<<" out of range"<<sendl;
                idxs[j] = 0;
            }
        }

        if (verts.size() == 4)
        {
            // quad
            quads.push_back(helper::make_array(idxs[0],idxs[1],idxs[2],idxs[3]));
        }
        else
        {
            // triangle(s)
            for (unsigned int j = 2; j < verts.size(); j++)
            {
                triangles.push_back(helper::make_array(idxs[0],idxs[j-1],idxs[j]));
            }
        }
    }

//             sout << "Facets Export size : ";
//             if (!triangles.empty())
//                 sout << triangles.size() << " triangles";
//             if (!quads.empty())
//                 sout << quads.size() << " quads";
//             sout << "." << sendl;

    //for (unsigned int i = 0; i < triangles.size() ; i++)
    //    sout << "T"<<i<<": "<<triangles[i][0]<<" "<<triangles[i][1]<<" "<<triangles[i][2]<<sendl;

    computeNormals();
    computeBBox();
}

bool VisualModelImpl::load(const std::string& filename, const std::string& loader, const std::string& textureName)
{
    bool tex = !textureName.empty() || putOnlyTexCoords.getValue();
    if (!textureName.empty() )
    {
        std::string textureFilename(textureName);
        if (sofa::helper::system::DataRepository.findFile (textureFilename))
            tex = loadTexture(textureName);
        else
            serr <<"Texture \""<<textureName <<"\" not found" << sendl;
    }

    if (!filename.empty() && vertices.size() == 0)
    {
        std::string meshFilename(filename);
        if (sofa::helper::system::DataRepository.findFile (meshFilename))
        {
            //name = filename;
            helper::io::Mesh *objLoader;
            if (loader.empty())
                objLoader = helper::io::Mesh::Create(filename);
            else
                objLoader = helper::io::Mesh::Create(loader, filename);

            if (!objLoader)
            {
                return false;
            }
            else
            {
                setMesh(*objLoader,tex);
                //setMesh(*objLoader,true);
                //sout << "VisualModel::load, vertices.size = "<< vertices.size() <<sendl;
            }
        }
        else
            serr <<"Mesh \""<< filename <<"\" not found" << sendl;
    }
    else
    {
        if (vertices.size() == 0)
        {
            sout << "VisualModel: will use Topology."<<sendl;
            useTopology = true;
        }
        else  computeBBox();
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
    applyUVScale(scaleTex.getValue()[0], scaleTex.getValue()[1]);
    applyUVTranslation(translationTex.getValue()[0],translationTex.getValue()[1]);
    scaleTex.setValue(TexCoord(1,1));
    translationTex.setValue(TexCoord(0,0));
}

void VisualModelImpl::applyTranslation(const double dx, const double dy, const double dz)
{
    Vector3 d((GLfloat)dx,(GLfloat)dy,(GLfloat)dz);
    VecCoord& x = *getVecX();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i] += d;
    }
    updateVisual();
}

//Apply Rotation from Euler angles (in degree!)
void VisualModelImpl::applyRotation (const double rx, const double ry, const double rz)
{
    Quaternion q=helper::Quater<SReal>::createQuaterFromEuler( Vec<3,SReal>(rx,ry,rz)*M_PI/180.0);
    applyRotation(q);
}
void VisualModelImpl::applyRotation(const Quat q)
{
    VecCoord& x = *getVecX();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i] = q.rotate(x[i]);
    }
    updateVisual();
}

void VisualModelImpl::applyScale(const double scale)
{
    VecCoord& x = *getVecX();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i] *= (GLfloat) scale;
    }
    updateVisual();
}

void VisualModelImpl::applyUVTranslation(const double dU, const double dV)
{
    for (unsigned int i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] += (GLfloat) dU;
        vtexcoords[i][1] += (GLfloat) dV;
    }
}

void VisualModelImpl::applyUVScale(const double scaleU, const double scaleV)
{
    for (unsigned int i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] *= (GLfloat) scaleU;
        vtexcoords[i][1] *= (GLfloat) scaleV;
    }
}

void VisualModelImpl::init()
{

    load(fileMesh.getFullPath(), "", texturename.getFullPath());
    _topology = getContext()->getMeshTopology();

    field_vertices.beginEdit();
    field_vnormals.beginEdit();
    field_vtexcoords.beginEdit();
    field_triangles.beginEdit();
    field_quads.beginEdit();



    applyScale(scale.getValue());
    applyRotation(rotation.getValue()[0],rotation.getValue()[1],rotation.getValue()[2]);
    applyTranslation(translation.getValue()[0],translation.getValue()[1],translation.getValue()[2]);


    translation.setValue(Vector3());
    rotation.setValue(Vector3());
    scale.setValue((SReal)1.0);
    VisualModel::init();
    updateVisual();
}

void VisualModelImpl::computeNormals()
{
    if (vertNormIdx.empty())
    {
        int nbn = vertices.size();
// 		serr << "nb of visual vertices"<<nbn<<sendl;
// 		serr << "nb of visual triangles"<<triangles.size()<<sendl;

        ResizableExtVector<Coord>& normals = vnormals;

        normals.resize(nbn);
        for (int i = 0; i < nbn; i++)
            normals[i].clear();

        for (unsigned int i = 0; i < triangles.size() ; i++)
        {

            const Coord  v1 = vertices[triangles[i][0]];
            const Coord  v2 = vertices[triangles[i][1]];
            const Coord  v3 = vertices[triangles[i][2]];
            Coord n = cross(v2-v1, v3-v1);

            n.normalize();
            normals[triangles[i][0]] += n;
            normals[triangles[i][1]] += n;
            normals[triangles[i][2]] += n;

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
            n1.normalize(); n2.normalize(); n3.normalize(); n4.normalize();
            normals[quads[i][0]] += n1;
            normals[quads[i][1]] += n2;
            normals[quads[i][2]] += n3;
            normals[quads[i][3]] += n4;
        }
        for (unsigned int i = 0; i < normals.size(); i++)
        {
            normals[i].normalize();
        }
    }
    else
    {
        vector<Coord> normals;
        int nbn = 0;
        for (unsigned int i = 0; i < vertNormIdx.size(); i++)
            if (vertNormIdx[i] >= nbn)
                nbn = vertNormIdx[i]+1;

        normals.resize(nbn);
        for (int i = 0; i < nbn; i++)
            normals[i].clear();

        for (unsigned int i = 0; i < triangles.size() ; i++)
        {
            const Coord & v1 = vertices[triangles[i][0]];
            const Coord & v2 = vertices[triangles[i][1]];
            const Coord & v3 = vertices[triangles[i][2]];
            Coord n = cross(v2-v1, v3-v1);
            n.normalize();
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
            n1.normalize(); n2.normalize(); n3.normalize(); n4.normalize();
            normals[vertNormIdx[quads[i][0]]] += n1;
            normals[vertNormIdx[quads[i][1]]] += n2;
            normals[vertNormIdx[quads[i][2]]] += n3;
            normals[vertNormIdx[quads[i][3]]] += n4;
        }

        for (unsigned int i = 0; i < normals.size(); i++)
        {
            normals[i].normalize();
        }
        vnormals.resize(vertices.size());
        for (unsigned int i = 0; i < vertices.size(); i++)
        {
            vnormals[i] = normals[vertNormIdx[i]];
        }
    }
}

void VisualModelImpl::computeBBox()
{
    const VecCoord& x = vertices;
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
}

void VisualModelImpl::setColor(float r, float g, float b, float a)
{
    helper::io::Mesh::Material M = material.getValue();
    M.setColor(r,g,b,a);
    material.setValue(M);
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

void VisualModelImpl::updateVisual()
{
    //sout << "VisualModelImpl::updateVisual()"<<sendl;
    if (modified && (!vertices.empty() || useTopology))
    {
        if (useTopology)
        {
            /** HD : build also a Ogl description from main Topology. But it needs to be build only once since the topology update
            is taken care of by the handleTopologyChange() routine */

            sofa::core::componentmodel::topology::TopologyModifier* topoMod;
            this->getContext()->get(topoMod);

            if (topoMod)   // dynamic topology
            {
                useTopology=false;
                computeMesh();
            }
            else
            {

                if (topoMod == NULL && (_topology->getRevision() != lastMeshRev))  // static topology
                {
                    computeMesh();
                }
            }
        }
        computePositions();
        computeNormals();
        computeBBox();
        updateBuffers();
        modified = false;
    }
}


void VisualModelImpl::computePositions()
{
    if (!vertPosIdx.empty())
    {
        // Need to transfer positions
        for (unsigned int i=0 ; i < vertices.size(); ++i)
            vertices[i] = (*inputVertices)[vertPosIdx[i]];
    }
}

void VisualModelImpl::computeMesh()
{
    if (vertices.empty())
    {
        if (_topology->hasPos())
        {

            if (sofa::component::topology::SparseGridTopology * spTopo = dynamic_cast< sofa::component::topology::SparseGridTopology *>(_topology))
            {
                sout << "VisualModel: getting marching cube mesh from topology : ";
                sofa::helper::io::Mesh m;
                spTopo->getMesh(m);
                setMesh(m, !texturename.getValue().empty());
                sout
                        <<m.getVertices().size()<<" points, "
                                <<m.getFacets().size()  << " triangles."<<sendl;
                useTopology = false; //visual model needs to be created only once at initial time
                return;
            }
            if (this->f_printLog.getValue())
                sout << "VisualModel: copying "<<_topology->getNbPoints()<<" points from topology."<<sendl;
            vertices.resize(_topology->getNbPoints());

            for (unsigned int i=0; i<vertices.size(); i++)
            {
                vertices[i][0] = (Real)_topology->getPX(i);
                vertices[i][1] = (Real)_topology->getPY(i);
                vertices[i][2] = (Real)_topology->getPZ(i);
            }

        }
        else
        {
            core::componentmodel::behavior::BaseMechanicalState* mstate = dynamic_cast<core::componentmodel::behavior::BaseMechanicalState*>(_topology->getContext()->getMechanicalState());
            if (mstate)
            {
                if (this->f_printLog.getValue())
                    sout << "VisualModel: copying "<<mstate->getSize()<<" points from mechanical state."<<sendl;
                vertices.resize(mstate->getSize());

                for (unsigned int i=0; i<vertices.size(); i++)
                {
                    vertices[i][0] = (Real)mstate->getPX(i);
                    vertices[i][1] = (Real)mstate->getPY(i);
                    vertices[i][2] = (Real)mstate->getPZ(i);
                }
            }
        }
    }

    lastMeshRev = _topology->getRevision();
    const vector<sofa::core::componentmodel::topology::BaseMeshTopology::Triangle>& inputTriangles = _topology->getTriangles();
    if (this->f_printLog.getValue())
        sout << "VisualModel: copying "<<inputTriangles.size()<<" triangles from topology."<<sendl;

    triangles.resize(inputTriangles.size());

    for (unsigned int i=0; i<triangles.size(); ++i)
    {
        triangles[i] = inputTriangles[i];
    }

    const vector<sofa::core::componentmodel::topology::BaseMeshTopology::Quad>& inputQuads = _topology->getQuads();
    if (this->f_printLog.getValue())
        sout << "VisualModel: copying "<<inputQuads.size()<<" quads from topology."<<sendl;
    quads.resize(inputQuads.size());
    for (unsigned int i=0; i<quads.size(); ++i)
        quads[i] = inputQuads[i];
}

void VisualModelImpl::handleTopologyChange()
{

    bool debug_mode = false;

    std::list<const TopologyChange *>::const_iterator itBegin=_topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=_topology->lastChange();

    while( itBegin != itEnd )
    {
        core::componentmodel::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

        switch( changeType )
        {
        case core::componentmodel::topology::ENDING_EVENT:
        {
            //sout << "INFO_print : Vis - ENDING_EVENT" << sendl;
            updateVisual();
            break;
        }

        case core::componentmodel::topology::TRIANGLESADDED:
        {
            //sout << "INFO_print : Vis - TRIANGLESADDED" << sendl;

            const sofa::component::topology::TrianglesAdded *ta=static_cast< const sofa::component::topology::TrianglesAdded * >( *itBegin );
            Triangle t;

            for (unsigned int i=0; i<ta->getNbAddedTriangles(); ++i)
            {

                t[0]=(int)(ta->triangleArray[i])[0];
                t[1]=(int)(ta->triangleArray[i])[1];
                t[2]=(int)(ta->triangleArray[i])[2];
                triangles.push_back(t);
            }

            break;
        }

        case core::componentmodel::topology::QUADSADDED:
        {
            //sout << "INFO_print : Vis - QUADSADDED" << sendl;

            const sofa::component::topology::QuadsAdded *ta_const=static_cast< const sofa::component::topology::QuadsAdded * >( *itBegin );
            sofa::component::topology::QuadsAdded *ta = const_cast< sofa::component::topology::QuadsAdded * >(ta_const);
            Quad t;

            for (unsigned int i=0; i<ta->getNbAddedQuads(); ++i)
            {

                t[0]=(int)(ta->getQuad(i))[0];
                t[1]=(int)(ta->getQuad(i))[1];
                t[2]=(int)(ta->getQuad(i))[2];
                t[3]=(int)(ta->getQuad(i))[3];
                quads.push_back(t);
            }

            break;
        }

        case core::componentmodel::topology::TRIANGLESREMOVED:
        {
            //sout << "INFO_print : Vis - TRIANGLESREMOVED" << sendl;

            unsigned int last;
            unsigned int ind_last;

            last= _topology->getNbTriangles() - 1;

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

        case core::componentmodel::topology::QUADSREMOVED:
        {
            //sout << "INFO_print : Vis - QUADSREMOVED" << sendl;

            unsigned int last;
            unsigned int ind_last;

            last= _topology->getNbQuads() - 1;

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

        // Case "POINTSREMOVED" added to propagate the treatment to the Visual Model

        case core::componentmodel::topology::POINTSREMOVED:
        {
            //sout << "INFO_print : Vis - POINTSREMOVED" << sendl;

            if (_topology->getNbTriangles()>0)
            {

                unsigned int last = _topology->getNbPoints() -1;

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

                    const sofa::helper::vector<unsigned int> &shell= _topology->getTriangleVertexShell(lastIndexVec[i]);
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

                                    if(ind_forgotten<_topology->getNbTriangles())
                                    {
                                        const sofa::component::topology::Triangle t_forgotten = _topology->getTriangle(ind_forgotten);
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
            else
            {

                if (_topology->getNbQuads()>0)
                {

                    unsigned int last = _topology->getNbPoints() -1;

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

                        const sofa::helper::vector<unsigned int> &shell= _topology->getQuadVertexShell(lastIndexVec[i]);
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

                    ///

                }


            }

            //}

            break;

        }


        // Case "POINTSRENUMBERING" added to propagate the treatment to the Visual Model

        case core::componentmodel::topology::POINTSRENUMBERING:
        {
            //sout << "INFO_print : Vis - POINTSRENUMBERING" << sendl;

            if (_topology->getNbTriangles()>0)
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
            else
            {
                if (_topology->getNbQuads()>0)
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
            }

            //}

            break;

        }

        case core::componentmodel::topology::POINTSMOVED:
        {
            updateVisual();
            break;
        }

        default:
            // Ignore events that are not Triangle  related.
            break;
        }; // switch( changeType )

        ++itBegin;
    } // while( changeIt != last; )


}

void VisualModelImpl::initVisual()
{
    //if (tex)
    //{
    //    tex->init();
    //}
}

void VisualModelImpl::exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, int& vindex, int& nindex, int& tindex)
{
    *out << "g "<<name<<"\n";

    if (mtl != NULL) // && !material.name.empty())
    {
        std::string name; // = material.name;
        if (name.empty())
        {
            static int count = 0;
            std::ostringstream o; o << "mat" << ++count;
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
    const ResizableExtVector<Coord>& x = *inputVertices;

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

