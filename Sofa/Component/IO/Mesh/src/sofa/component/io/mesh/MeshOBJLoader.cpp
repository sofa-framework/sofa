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
#include <sofa/component/io/mesh/MeshOBJLoader.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/SetDirectory.h>
#include <fstream>
#include <sofa/helper/accessor.h>
#include <sofa/helper/system/Locale.h>

namespace sofa::component::io::mesh
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::core::loader;
using sofa::helper::getWriteOnlyAccessor;

void registerMeshOBJLoader(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Specific mesh loader for OBJ file format.")
        .add< MeshOBJLoader >());
}

MeshOBJLoader::MeshOBJLoader()
    : MeshLoader()
    , faceType(MeshOBJLoader::TRIANGLE)
    , d_handleSeams(initData(&d_handleSeams, (bool)false, "handleSeams", "Preserve UV and normal seams information (vertices with multiple UV and/or normals)"))
    , d_loadMaterial(initData(&d_loadMaterial, (bool) true, "loadMaterial", "Load the related MTL file or use a default one?"))
    , d_material(initData(&d_material,"defaultMaterial","Default material") )
    , d_materials(initData(&d_materials,"materials","List of materials") )
    , d_faceList(initData(&d_faceList,"faceList","List of face definitions.") )
    , d_texIndexList(initData(&d_texIndexList,"texcoordsIndex","Indices of textures coordinates used in faces definition."))
    , d_positionsList(initData(&d_positionsList,"positionsDefinition", "Vertex positions definition"))
    , d_texCoordsList(initData(&d_texCoordsList,"texcoordsDefinition", "Texture coordinates definition"))
    , d_normalsIndexList(initData(&d_normalsIndexList,"normalsIndex","List of normals of elements of the mesh loaded."))
    , d_normalsList(initData(&d_normalsList,"normalsDefinition","Normals definition"))
    , d_texCoords(initData(&d_texCoords,"texcoords","Texture coordinates of all faces, to be used as the parent data of a VisualModel texcoords data"))
    , d_computeMaterialFaces(initData(&d_computeMaterialFaces, false, "computeMaterialFaces", "True to activate export of Data instances containing list of face indices for each material"))
    , d_vertPosIdx      (initData   (&d_vertPosIdx, "vertPosIdx", "If vertices have multiple normals/texcoords stores vertices position indices"))
    , d_vertNormIdx     (initData   (&d_vertNormIdx, "vertNormIdx", "If vertices have multiple normals/texcoords stores vertices normal indices"))
{
    addAlias(&d_material, "material");

    d_material.setGroup("Shading");
    d_materials.setGroup("Shading");

    d_texIndexList.setGroup("Texturing");
    d_texCoordsList.setGroup("Texturing");
    d_texCoords.setGroup("Texturing");

    d_faceList.setGroup("Geometry");
    d_normalsIndexList.setGroup("Geometry");
    d_normalsList.setGroup("Geometry");
    d_positionsList.setGroup("Geometry");
    d_vertPosIdx.setGroup("Geometry");
    d_vertNormIdx.setGroup("Geometry");

    addOutputsToCallback("filename", {&d_texCoordsList, &d_normalsList,
        &d_material, &d_materials, &d_faceList, &d_normalsIndexList,
        &d_texIndexList});
}

MeshOBJLoader::~MeshOBJLoader()
{

}

bool MeshOBJLoader::doLoad()
{
    dmsg_info() << "Loading OBJ file: " << d_filename;

    bool fileRead = false;

    // -- Loading file
    const char* filename = d_filename.getFullPath().c_str();
    std::ifstream file(filename);

    if (!file.good())
    {
        msg_error() << "Cannot read file '" << d_filename << "'.";
        return false;
    }

    // -- Reading file
    fileRead = readOBJ (file,filename);
    file.close();

    return fileRead;
}

///
/// \brief MeshOBJLoader::clearBuffers
/// Clear all the buffer containing the data loaded from the file.
///
void MeshOBJLoader::doClearBuffers()
{
    getWriteOnlyAccessor(d_texCoordsList).clear();
    getWriteOnlyAccessor(d_normalsList).clear();

    getWriteOnlyAccessor(d_material)->activated = false;
    getWriteOnlyAccessor(d_materials).clear();
    getWriteOnlyAccessor(d_faceList)->clear();
    getWriteOnlyAccessor(d_normalsIndexList)->clear();
    getWriteOnlyAccessor(d_texIndexList)->clear();
}

void MeshOBJLoader::addGroup (const PrimitiveGroup& g)
{
    /// Get the accessors to the data vectors.
    /// The accessors are using the RAII design pattern to handle automatically
    /// the beginEdit/endEdit pairs.
    auto my_edgesGroups = getWriteOnlyAccessor(d_edgesGroups);
    auto my_trianglesGroups = getWriteOnlyAccessor(d_trianglesGroups);
    auto my_quadsGroups = getWriteOnlyAccessor(d_quadsGroups);

    switch (faceType)
    {
    case MeshOBJLoader::EDGE:
        my_edgesGroups.push_back(g);
        break;
    case MeshOBJLoader::TRIANGLE:
        my_trianglesGroups.push_back(g);
        break;
    case MeshOBJLoader::QUAD:
        my_quadsGroups.push_back(g);
        break;
    default: break;
    }
}

bool MeshOBJLoader::readOBJ (std::ifstream &file, const char* filename)
{
    // Make sure that fscanf() uses a dot '.' as the decimal separator.
    sofa::helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    const bool handleSeams = d_handleSeams.getValue();
    auto my_positions = getWriteOnlyAccessor(d_positions);
    auto my_texCoords = getWriteOnlyAccessor(d_texCoordsList);
    auto my_normals   = getWriteOnlyAccessor(d_normalsList);

    auto material = getWriteOnlyAccessor(d_material);
    auto my_materials = getWriteOnlyAccessor(d_materials);
    auto my_faceList = getWriteOnlyAccessor(d_faceList);
    auto my_normalsList = getWriteOnlyAccessor(d_normalsIndexList);
    auto my_texturesList  = getWriteOnlyAccessor(d_texIndexList);
    type::vector<int> nodes, nIndices, tIndices;

    auto my_edges = getWriteOnlyAccessor(d_edges);
    auto my_triangles = getWriteOnlyAccessor(d_triangles);
    auto my_quads = getWriteOnlyAccessor(d_quads);

    //BUGFIX: clear pre-existing data before loading the file
    my_positions.clear();
    material->activated = false;
    my_texCoords.clear();
    my_normals.clear();
    my_materials.clear();
    my_faceList->clear();
    my_normalsList->clear();
    my_texturesList->clear();
    my_edges.clear();
    my_triangles.clear();
    my_quads.clear();

    getWriteOnlyAccessor(d_edgesGroups).clear();
    getWriteOnlyAccessor(d_trianglesGroups).clear();
    getWriteOnlyAccessor(d_quadsGroups).clear();

    int vtn[3];
    Vec3 result;
    helper::WriteOnlyAccessor<Data<type::vector< PrimitiveGroup> > > my_faceGroups[NBFACETYPE] =
    {
        d_edgesGroups,
        d_trianglesGroups,
        d_quadsGroups
    };
    std::string curGroupName = "Default_Group";
    std::string curMaterialName;
    int curMaterialId = -1;
    int nbFaces[NBFACETYPE] = {0}; // number of edges, triangles, quads
    int groupF0[NBFACETYPE] = {0}; // first primitives indices in current group for edges, triangles, quads
    std::string line;
    while( std::getline(file,line) )
    {
        if (line.empty()) continue;
        std::istringstream values(line);
        std::string token;
        values >> token;

        if (token == "#")
        {
            // comment
        }
        else if (token == "v")
        {
            // vertex
            values >> result[0] >> result[1] >> result[2];
            my_positions.push_back(Vec3(result[0],result[1], result[2]));
        }
        else if (token == "vn")
        {
            // normal
            values >> result[0] >> result[1] >> result[2];
            my_normals.push_back(Vec3(result[0],result[1], result[2]));
        }
        else if (token == "vt")
        {
            // texcoord
            values >> result[0] >> result[1];
            my_texCoords.push_back(Vec2(result[0],result[1]));
        }
        else if ((token == "mtllib") && d_loadMaterial.getValue())
        {
            while (!values.eof())
            {
                std::string materialLibaryName;
                values >> materialLibaryName;
                std::string mtlfile = sofa::helper::system::SetDirectory::GetRelativeFromFile(materialLibaryName.c_str(), filename);
                this->readMTL(mtlfile.c_str(), my_materials.wref());
            }
        }
        else if (token == "usemtl" || token == "g")
        {
            // end of current group
            //curGroup.nbp = nbf - curGroup.p0;
            for (int ft = 0; ft < NBFACETYPE; ++ft)
                if (nbFaces[ft] > groupF0[ft])
                {
                    my_faceGroups[ft].push_back(PrimitiveGroup(groupF0[ft], nbFaces[ft]-groupF0[ft], curMaterialName, curGroupName, curMaterialId));
                    groupF0[ft] = nbFaces[ft];
                }
            if (token == "usemtl")
            {
                values >> curMaterialName;
                curMaterialId = -1;
                type::vector<Material>::iterator it = my_materials.begin();
                type::vector<Material>::iterator itEnd = my_materials.end();
                for (; it != itEnd; ++it)
                {
                    if (it->name == curMaterialName)
                    {
                        (*it).activated = true;
                        if (!material->activated)
                            material.wref() = *it;
                        curMaterialId = int(it - my_materials.begin());
                        break;
                    }
                }
            }
            else if (token == "g")
            {
                curGroupName.clear();
                while (!values.eof())
                {
                    std::string g;
                    values >> g;
                    if (!curGroupName.empty())
                        curGroupName += " ";
                    curGroupName += g;
                }
            }
        }
        else if (token == "l" || token == "f")
        {
            // face
            nodes.clear();
            nIndices.clear();
            tIndices.clear();

            while (!values.eof())
            {
                std::string face;
                values >> face;
                if (face.empty()) continue;
                for (int j = 0; j < 3; j++)
                {
                    vtn[j] = -1;
                    std::string::size_type pos = face.find('/');
                    std::string tmp = face.substr(0, pos);
                    if (pos == std::string::npos)
                        face = "";
                    else
                    {
                        face = face.substr(pos + 1);
                    }

                    if (!tmp.empty())
                    {
                        vtn[j] = atoi(tmp.c_str());
                        if (vtn[j] >= 1)
                            vtn[j] -=1; // -1 because the numerotation begins at 1 and a vector begins at 0
                        else if (vtn[j] < 0)
                            vtn[j] += (j==0) ? sofa::Size(my_positions.size()) : (j==1) ? sofa::Size(my_texCoords.size()) : sofa::Size(my_normals.size());
                        else
                        {
                            msg_error() << "Invalid index " << tmp;
                            vtn[j] = -1;
                        }
                    }
                }

                nodes.push_back(vtn[0]);
                tIndices.push_back(vtn[1]);
                nIndices.push_back(vtn[2]);
            }

            my_faceList->push_back(nodes);
            my_normalsList->push_back(nIndices);
            my_texturesList->push_back(tIndices);

            if (nodes.size() == 2) // Edge
            {
                if (!handleSeams) // we have to wait for renumbering vertices if we handle seams
                {
                    if (nodes[0]<nodes[1])
                        addEdge(my_edges.wref(), Edge(nodes[0], nodes[1]));
                    else
                        addEdge(my_edges.wref(), Edge(nodes[1], nodes[0]));
                }
                ++nbFaces[MeshOBJLoader::EDGE];
                faceType = MeshOBJLoader::EDGE;
            }
            else if (nodes.size()==4 && !this->d_triangulate.getValue()) // Quad
            {
                if (!handleSeams) // we have to wait for renumbering vertices if we handle seams
                {
                    addQuad(my_quads.wref(), Quad(nodes[0], nodes[1], nodes[2], nodes[3]));
                }
                ++nbFaces[MeshOBJLoader::QUAD];
                faceType = MeshOBJLoader::QUAD;
            }
            else // Triangulate
            {
                if (!handleSeams) // we have to wait for renumbering vertices if we handle seams
                {
                    for (size_t j=2; j<nodes.size(); j++)
                        addTriangle(my_triangles.wref(), Triangle(nodes[0], nodes[j-1], nodes[j]));
                }
                ++nbFaces[MeshOBJLoader::TRIANGLE];
                faceType = MeshOBJLoader::TRIANGLE;
            }

        }
        else
        {

        }
    }

    // end of current group
    for (size_t ft = 0; ft < NBFACETYPE; ++ft)
        if (nbFaces[ft] > groupF0[ft])
        {
            my_faceGroups[ft].push_back(PrimitiveGroup(groupF0[ft], nbFaces[ft]-groupF0[ft], curMaterialName, curGroupName, curMaterialId));
            groupF0[ft] = nbFaces[ft];
        }

    if (!d_handleSeams.getValue())
    {
        /// default mode, vertices are never duplicated, only one texcoord and normal is used per vertex
        auto vTexCoords = getWriteOnlyAccessor(d_texCoords);
        auto vNormals   = getWriteOnlyAccessor(d_normals);
        auto vVertices  = getWriteOnlyAccessor(d_positions);

        /// Copy the complete array.
        vVertices.wref() = my_positions.ref();

        size_t vertexCount = my_positions.size();
        if( my_texCoords.size() > 0 )
        {
            vTexCoords.resize(vertexCount);
        }
        else
        {
            vTexCoords.resize(0);
        }
        if( my_normals.size() > 0 )
        {
            vNormals.resize(vertexCount);
        }
        else
        {
            vNormals.resize(0);
        }
        for (size_t fi=0; fi<my_faceList->size(); ++fi)
        {
            const type::SVector<int>& nodes = (*my_faceList)[fi];
            const type::SVector<int>& nIndices = (*my_normalsList)[fi];
            const type::SVector<int>& tIndices = (*my_texturesList)[fi];
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                unsigned int pi = nodes[i];
                unsigned int ni = nIndices[i];
                unsigned int ti = tIndices[i];
                if (pi >= vertexCount) continue;
                if (ti < my_texCoords.size() && (vTexCoords[pi] == sofa::type::Vec2() ||
                                                 (my_texCoords[ti]-vTexCoords[pi])*sofa::type::Vec2(-1,1) > 0))
                    vTexCoords[pi] = my_texCoords[ti];
                if (ni < my_normals.size())
                    vNormals[pi] += my_normals[ni];
            }
        }
        for (auto& vNormal : vNormals)
        {
            vNormal.normalize();
        }
    }
    else
    { // handleSeam mode : vertices are duplicated in case they have different texcoords and/or normals
        // This code was initially in VisualModelImpl::setMesh()

        auto nbVIn = my_positions.size();
        // First we compute for each point how many pair of normal/texcoord indices are used
        // The map store the final index of each combinaison
        std::vector< std::map< std::pair<int,int>, int > > vertTexNormMap;
        vertTexNormMap.resize(nbVIn);
        for (size_t fi=0; fi<my_faceList->size(); ++fi)
        {
            const type::SVector<int>& nodes = (*my_faceList)[fi];
            const type::SVector<int>& nIndices = (*my_normalsList)[fi];
            const type::SVector<int>& tIndices = (*my_texturesList)[fi];
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                unsigned int pi = nodes[i];
                unsigned int ni = nIndices[i];
                unsigned int ti = tIndices[i];
                vertTexNormMap[pi][std::make_pair(ti, ni)] = 0;
            }
        }

        // Then we can compute how many vertices are created
        sofa::Size nbVOut = 0;
        bool vsplit = false;
        for (sofa::Index i = 0; i < nbVIn; i++)
        {
            auto s = sofa::Size(vertTexNormMap[i].size());
            nbVOut += s;
        }

        dmsg_info() << nbVIn << " input positions, " << nbVOut << " final vertices.";

        if (nbVIn != nbVOut)
            vsplit = true;

        // Then we can create the final arrays

        type::vector<sofa::type::Vec3> vertices2;
        auto vnormals = getWriteOnlyAccessor(d_normals);
        auto vtexcoords = getWriteOnlyAccessor(d_texCoords);
        auto vertPosIdx = getWriteOnlyAccessor(d_vertPosIdx);
        auto vertNormIdx = getWriteOnlyAccessor(d_vertNormIdx);

        vertices2.resize(nbVOut);
        vnormals.resize(nbVOut);
        vtexcoords.resize(nbVOut);
        if (vsplit)
        {
            vertPosIdx.resize(nbVOut);
            vertNormIdx.resize(nbVOut);
        }

        sofa::Size nbNOut = 0; /// Number of different normals
        for (sofa::Index i = 0, j = 0; i < nbVIn; i++)
        {
            std::map<int, int> normMap;
            for (auto it = vertTexNormMap[i].begin();
                 it != vertTexNormMap[i].end(); ++it)
            {
                int t = it->first.first;
                int n = it->first.second;
                if ( (unsigned)n < my_normals.size())
                    vnormals[j] = my_normals[n];
                if ((unsigned)t < my_texCoords.size())
                    vtexcoords[j] = my_texCoords[t];
                
                vertices2[j] = my_positions[i];
                if (vsplit)
                {
                    vertPosIdx[j] = i;
                    if (normMap.contains(n))
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

        // replace the original (non duplicated) vector with the new one
        my_positions.clear();
        for(const sofa::type::Vec3& c : vertices2)
            my_positions.push_back(c);

        if( vsplit && nbNOut == nbVOut )
            vertNormIdx.resize(0);

        // Then we create the triangles and quads
        
        for (size_t fi=0; fi<my_faceList->size(); ++fi)
        {
            const type::SVector<int>& verts = (*my_faceList)[fi];
            const type::SVector<int>& nIndices = (*my_normalsList)[fi];
            const type::SVector<int>& tIndices = (*my_texturesList)[fi];
            std::vector<int> nodes;
            nodes.resize(verts.size());
            for (size_t i = 0; i < verts.size(); ++i)
            {
                unsigned int pi = verts[i];
                unsigned int ni = nIndices[i];
                unsigned int ti = tIndices[i];
                nodes[i] = vertTexNormMap[pi][std::make_pair(ti, ni)];
                if ((unsigned)nodes[i] >= (unsigned)nbVOut)
                {
                    msg_error() << this->getPathName()<<" index "<<nodes[i]<<" out of range";
                    nodes[i] = 0;
                }
            }

            if (nodes.size() == 2) // Edge
            {
                if (nodes[0]<nodes[1])
                    addEdge(my_edges.wref(), Edge(nodes[0], nodes[1]));
                else
                    addEdge(my_edges.wref(), Edge(nodes[1], nodes[0]));
            }
            else if (nodes.size()==4 && !this->d_triangulate.getValue()) // Quad
            {
                addQuad(my_quads.wref(), Quad(nodes[0], nodes[1], nodes[2], nodes[3]));
            }
            else // Triangulate
            {
                for (size_t j=2; j<nodes.size(); j++)
                    addTriangle(my_triangles.wref(), Triangle(nodes[0], nodes[j-1], nodes[j]));
            }
        }
        for (auto& vnormal : vnormals)
        {
            vnormal.normalize();
        }
    }


    if (d_computeMaterialFaces.getValue())
    {
        // create subset lists
        std::map< std::string, type::vector<unsigned int> > materialFaces[NBFACETYPE];
        for (int ft = 0; ft < NBFACETYPE; ++ft)
        {
            for (size_t gi=0; gi<my_faceGroups[ft].size(); ++gi)
            {
                PrimitiveGroup g = my_faceGroups[ft][gi];
                type::vector<unsigned int>& out = materialFaces[ft][g.materialName];
                for (int f=g.p0; f<g.p0+g.nbp; ++f)
                    out.push_back(f);
            }
        }
        for (const auto& materialFace : materialFaces)
        {
            std::string fname;
            switch (faceType)
            {
            case MeshOBJLoader::EDGE:     fname = "edge"; break;
            case MeshOBJLoader::TRIANGLE: fname = "triangle"; break;
            case MeshOBJLoader::QUAD:     fname = "quad"; break;
            default: break;
            }
            for (const auto& [materialName, faces] : materialFace)
            {
                if (faces.empty()) continue;
                std::ostringstream oname;
                oname << "material_" << materialName << "_" << fname << "Indices";
                Data< type::vector<unsigned int> >* dOut = new Data< type::vector<unsigned int> >("list of face indices corresponding to a given material");
                dOut->setName(oname.str());

                this->addData(dOut);
                dOut->setGroup("Materials");
                dOut->setValue(faces);
                d_subsets_indices.push_back(dOut);
            }
        }
    }

    return true;
}


// -----------------------------------------------------
// readMTL: read a wavefront material library file
//
//    model - properly initialized GLMmodel structure
//    name  - name of the material library
// -----------------------------------------------------
bool MeshOBJLoader::readMTL(const char* filename, type::vector<Material>& materials)
{
    FILE* file;
    char buf[128]; // Note: in the strings below, 127 is sizeof(buf)-1
    const char *single_string_format = "%127s"; // Better than "%s" for scanf
    const char *double_string_format = "%127s %127s"; // Better than "%s %s"

    file = fopen(filename, "r");
    if (!file) {
        msg_info() << "readMTL(): can't open material file " << filename;
        return false;
    }

    Material *mat = nullptr;
    /* now, read in the data */
    while (fscanf(file, single_string_format, buf) != EOF)
    {

        switch (buf[0])
        {
        case '#':
            /* comment */
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == nullptr)
            {
                if (feof(file))
                    msg_error() << "Error: MeshOBJLoader: fgets function has encounter end of file. case #.";
                else
                    msg_error() << "Error: MeshOBJLoader: fgets function has encounter an error. case #.";
            }
            break;
        case 'n':
            /* newmtl */
            if (mat != nullptr)
            {
                materials.push_back(*mat);
                delete mat;
                mat = nullptr;
            }
            mat = new Material();
            if ( fgets(buf, sizeof(buf), file) == nullptr)
            {
                if (feof (file) )
                    msg_error() << "Problem while reading file, fgets function has encounter end of file. case n.";
                else
                    msg_error() << "Problem while reading file, fgets function has encounter an error. case n.";
            }
            sscanf(buf, double_string_format, buf, buf);
            mat->name = buf;
            break;
        case 'N':
            switch (buf[1])
            {
            case 'i':
            {
                float optical_density;
                if ( fscanf(file, "%f", &optical_density) == EOF )
                    msg_error() << "Problem while reading file, fscanf function has encounter an error. case N i.";
                break;
            }
            case 's':
                if (fscanf(file, "%f", &mat->shininess) == EOF )
                    msg_error() << "Problem while reading file, fscanf function has encounter an error. case N s.";

                mat->useShininess = true;
                break;
            default:
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == nullptr)
                {
                    if (feof (file) )
                        msg_error() << "Problem while reading file, fgets function has encounter end of file. case N.";
                    else
                        msg_error() << "Problem while reading file, fgets function has encounter an error. case N.";
                }
                break;
            }
            break;
        case 'K':
            switch (buf[1])
            {
            case 'd':
                if ( fscanf(file, "%f %f %f", &mat->diffuse[0], &mat->diffuse[1], &mat->diffuse[2]) == EOF)
                    msg_error() << "Problem while reading file, fscanf function has encounter an error. case K d.";
                mat->useDiffuse = true;
                break;
            case 's':
                if ( fscanf(file, "%f %f %f", &mat->specular[0], &mat->specular[1], &mat->specular[2]) == EOF)
                    msg_error() << "Problem while reading file, fscanf function has encounter an error. case K s.";
                mat->useSpecular = true;
                break;
            case 'a':
                if ( fscanf(file, "%f %f %f", &mat->ambient[0], &mat->ambient[1], &mat->ambient[2]) == EOF)
                    msg_error() << "Problem while reading file, fscanf function has encounter an error. case K a.";
                mat->useAmbient = true;
                break;
            default:
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == nullptr)
                {
                    if (feof (file) )
                        msg_error() << "Problem while reading file, fgets function has encounter end of file. case K.";
                    else
                        msg_error() << "Problem while reading file, fgets function has encounter an error. case K.";
                }
                break;
            }
            break;
        case 'd':
        case 'T':
            if (!mat)
            {
                msg_error() << "Problem while reading file, readMTL 'T' no material";
                break;
            }
            // transparency value
            if ( fscanf(file, "%f", &mat->diffuse[3]) == EOF)
                msg_error() << "Problem while reading file, fscanf function has encounter an error. case T i.";
            break;

        case 'm':
        {
            if (!mat)
            {
                msg_error() << "Problem while reading file, readMTL 'm' no material";
                break;
            }
            //texture map
            char charFilename[128] = { 0 };
            if (fgets(charFilename, sizeof(charFilename), file) == nullptr)
            {
                msg_error() << "Problem while reading file, fgets has encountered an error";
            }
            else
            {
                mat->useTexture = true;

                //store the filename of the texture map in the material
                std::string stringFilename(charFilename);

                //delete carriage return from the string assuming the next property of the .mtl file is at the next line
                stringFilename.erase(stringFilename.end() - 1, stringFilename.end());
                stringFilename.erase(stringFilename.begin(), stringFilename.begin() + 1);
                mat->textureFilename = stringFilename;
            }

            break;
        }
        case 'b':
        {
            if (!mat)
            {
                msg_error() << "Problem while reading file, readMTL 'b' no material";
                break;
            }
            //bump mapping texture map
            char charFilename[128] = { 0 };
            if (fgets(charFilename, sizeof(charFilename), file) == nullptr)
            {
                msg_error() << "Problem while reading file, fgets has encountered an error";
            }
            else
            {
                mat->useBumpMapping = true;

                //store the filename of the texture map in the material
                std::string stringFilename(charFilename);

                //delete carriage return from the string assuming the next property of the .mtl file is at the next line
                stringFilename.erase(stringFilename.end() - 1, stringFilename.end());
                stringFilename.erase(stringFilename.begin(), stringFilename.begin() + 1);
                mat->bumpTextureFilename = stringFilename;
            }

            break;
        }
        default:
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == nullptr)
            {
                if (feof (file) )
                    msg_error() << "Problem while reading file, fgets function has encounter end of file. case default.";
                else
                    msg_error() << "Problem while reading file, fgets function has encounter an error. case default.";
            }
            break;
        }

    }
    fclose(file);

    if (mat != nullptr)
    {
        materials.push_back(*mat);
        delete mat;
        mat = nullptr;
    }

    return true;
}


} // namespace sofa::component::io::mesh
