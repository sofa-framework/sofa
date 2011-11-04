/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/loader/MeshObjLoader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/SetDirectory.h>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;

SOFA_DECL_CLASS(MeshObjLoader)

int MeshObjLoaderClass = core::RegisterObject("Specific mesh loader for Obj file format.")
        .add< MeshObjLoader >()
        ;



MeshObjLoader::MeshObjLoader(): MeshLoader()
    , faceType(MeshObjLoader::TRIANGLE)
    , faceList(initData(&faceList,"faceList","List of face definitions.") )
    , texIndexList(initData(&texIndexList,"texIndex","Indices of textures coordinates used in faces definition."))
    , texCoordsList(initData(&texCoordsList,"texCoordsDefinition", "Texture coordinates definition"))
    , texCoords(initData(&texCoords,"texcoords","Texture coordinates of all faces, to be used as the parent data of a VisualModel texcoords data"))
    , normalsList(initData(&normalsList,"normalsList","List of normals of elements of the mesh loaded."))
    , vertices(initData(&vertices,"vertices","List of vertices. Different from position when more than \
                                               one texcoord normal pair is attached to a vertex." ) )
{
    texIndexList.setPersistent(false);
    texCoords.setPersistent(false);
    normalsList.setPersistent(false);
}




bool MeshObjLoader::load()
{
    sout << "Loading OBJ file: " << m_filename << sendl;

    bool fileRead = false;

    // -- Loading file
    const char* filename = m_filename.getFullPath().c_str();
    std::ifstream file(filename);

    if (!file.good())
    {
        serr << "Error: MeshObjLoader: Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }

    // -- Reading file
    fileRead = this->readOBJ (file,filename);
    file.close();

    return fileRead;
}


void MeshObjLoader::addGroup (const PrimitiveGroup& g)
{

    helper::vector< PrimitiveGroup>& my_edgesGroups = *(edgesGroups.beginEdit());
    helper::vector< PrimitiveGroup>& my_trianglesGroups = *(trianglesGroups.beginEdit());
    helper::vector< PrimitiveGroup>& my_quadsGroups = *(quadsGroups.beginEdit());

    switch (faceType)
    {
    case MeshObjLoader::EDGE:
        my_edgesGroups.push_back(g);
        break;
    case MeshObjLoader::TRIANGLE:
        my_trianglesGroups.push_back(g);
        break;
    case MeshObjLoader::QUAD:
        my_quadsGroups.push_back(g);
        break;
    default: break;
    }

    edgesGroups.endEdit();
    trianglesGroups.endEdit();
    quadsGroups.endEdit();
}

bool MeshObjLoader::readOBJ (std::ifstream &file, const char* filename)
{
    sout << "MeshObjLoader::readOBJ" << sendl;

    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());
    helper::vector<sofa::defaulttype::Vector2>& my_texCoords = *(texCoordsList.beginEdit());
    helper::vector<sofa::defaulttype::Vector3>& my_normals   = *(normalsList.beginEdit());

    helper::vector<Material>& my_materials = *(materials.beginEdit());
    helper::SVector< helper::SVector <int> >& my_faceList = *(faceList.beginEdit() );
    helper::SVector< helper::SVector <int> >& my_normalsList = *(normalsIndexList.beginEdit());
    helper::SVector< helper::SVector <int> >& my_texturesList   = *(texIndexList.beginEdit());
    helper::vector<int> nodes, nIndices, tIndices;

    helper::vector<Edge >& my_edges = *(edges.beginEdit());
    helper::vector<Triangle >& my_triangles = *(triangles.beginEdit());
    helper::vector<Quad >& my_quads = *(quads.beginEdit());

    int vtn[3];
    Vec3d result;
    int nbf = 0;
    PrimitiveGroup curGroup;
    std::string line;
    std::string face, tmp;
    while( std::getline(file,line) )
    {
        if (line.empty()) continue;
        std::istringstream values(line);
        std::string token;

        values >> token;
        if (token == "#")
        {
            /* comment */
        }
        else if (token == "v")
        {
            /* vertex */
            values >> result[0] >> result[1] >> result[2];
            my_positions.push_back(Vector3(result[0],result[1], result[2]));
        }
        else if (token == "vn")
        {
            /* normal */
            values >> result[0] >> result[1] >> result[2];
            my_normals.push_back(Vector3(result[0],result[1], result[2]));
        }
        else if (token == "vt")
        {
            /* texcoord */
            values >> result[0] >> result[1];
            my_texCoords.push_back(Vector2(result[0],result[1]));
        }
        else if (token == "mtllib")
        {
            while (!values.eof())
            {
                std::string materialLibaryName;
                values >> materialLibaryName;
                std::string mtlfile = sofa::helper::system::SetDirectory::GetRelativeFromFile(materialLibaryName.c_str(), filename);
                this->readMTL(mtlfile.c_str(), my_materials);
            }
        }
        else if (token == "usemtl" || token == "g")
        {
            // end of current group
            curGroup.nbp = nbf - curGroup.p0;
            if (curGroup.nbp > 0)
            {
                //warning : a group is supposed to be composed with the same type of face ...
                addGroup(curGroup);
            }
            curGroup.p0 = nbf;
            curGroup.nbp = 0;
            if (token == "usemtl")
            {
                curGroup.materialId = -1;
                std::string materialName;
                values >> materialName;
                helper::vector<Material>::iterator it = my_materials.begin();
                helper::vector<Material>::iterator itEnd = my_materials.end();

                for (; it != itEnd; it++)
                {
                    if (it->name == curGroup.materialName)
                    {
                        // std::cout << "Using material "<<it->name<<std::endl;
                        (*it).activated = true;
                        if (!material.activated)
                            material = *it;
                        curGroup.materialId = it - my_materials.begin();
                        break;
                    }
                }
            }
            else if (token == "g")
            {
                curGroup.groupName.clear();
                while (!values.eof())
                {
                    std::string g;
                    values >> g;
                    if (!curGroup.groupName.empty())
                        curGroup.groupName += " ";
                    curGroup.groupName += g;
                }
            }
        }
        else if (token == "l" || token == "f")
        {
            /* face */
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
                            vtn[j] += (j==0) ? my_positions.size() : (j==1) ? my_texCoords.size() : my_normals.size();
                        else
                        {
                            serr << "Invalid index " << tmp << sendl;
                            vtn[j] = -1;
                        }
                    }
                }

                nodes.push_back(vtn[0]);
                tIndices.push_back(vtn[1]);
                nIndices.push_back(vtn[2]);
            }

            my_faceList.push_back(nodes);
            my_normalsList.push_back(nIndices);
            my_texturesList.push_back(tIndices);


            for( unsigned int i =0 ; i < nodes.size(); ++i)
            {
                vertexIdx2textureNormalIdx[nodes[i]].insert( std::make_pair(tIndices[i],nIndices[i]) );
            }



            if (nodes.size() == 2) // Edge
            {
                if (nodes[0]<nodes[1])
                    addEdge(&my_edges, Edge(nodes[0], nodes[1]));
                else
                    addEdge(&my_edges, Edge(nodes[1], nodes[0]));

                faceType = MeshObjLoader::EDGE;
            }
            else if (nodes.size()==4) // Quad
            {
                addQuad(&my_quads, Quad(nodes[0], nodes[1], nodes[2], nodes[3]));

                faceType = MeshObjLoader::QUAD;
            }
            else // Triangularize
            {
                for (unsigned int j=2; j<nodes.size(); j++)
                    addTriangle(&my_triangles, Triangle(nodes[0], nodes[j-1], nodes[j]));

                faceType = MeshObjLoader::TRIANGLE;
            }

            ++nbf;

        }
        else
        {
            // std::cerr << "readObj : Unknown token for line " << line << std::endl;
        }
    }

    // end of current group
    if (curGroup.groupName.empty())
        curGroup.groupName = "Default_Group";

    curGroup.nbp = nbf - curGroup.p0;
    if (curGroup.nbp > 0) addGroup(curGroup);


    unsigned int vertexCount = 0;

    VertexIdx2TextureNormalIdxPairs::const_iterator iterVertexIndex;
    for( iterVertexIndex = vertexIdx2textureNormalIdx.begin();
            iterVertexIndex != vertexIdx2textureNormalIdx.end();
            ++iterVertexIndex )
    {
        const std::set< std::pair< int, int > >& setTextureNormalIndex = iterVertexIndex->second;
        vertexCount += setTextureNormalIndex.size();
    }


    helper::vector<sofa::defaulttype::Vector2>& vTexCoords = *texCoords.beginEdit();
    //helper::vector<sofa::defaulttype::Vector3>& vNormals   = *normals.beginEdit();
    helper::vector<sofa::defaulttype::Vector3>& vVertices  = *vertices.beginEdit();

    vVertices.resize(vertexCount);
    if( my_texCoords.size() > 0 )
    {
        vTexCoords.resize(vertexCount);
    }
    else
    {
        vTexCoords.resize(0);
    }
    /* if( my_normals.size() > 0 ){
       vNormals.resize(vertexCount);
     }
     else{
       vNormals.resize(0);
     }*/

    for ( iterVertexIndex = vertexIdx2textureNormalIdx.begin(); iterVertexIndex != vertexIdx2textureNormalIdx.end(); ++iterVertexIndex)
    {
        const int vertexIndex = iterVertexIndex->first;
        const std::set< std::pair< int, int > >& setTextureNormalIndex = iterVertexIndex->second;
        std::set< std::pair< int, int >  > ::const_iterator itTextureNormalIndex;
        for( itTextureNormalIndex = setTextureNormalIndex.begin();
                itTextureNormalIndex != setTextureNormalIndex.end();
                ++itTextureNormalIndex)
        {
            if( vertexIndex >= (int)my_positions.size() )
            {
                this->serr << "Invalid index for vertex: " << vertexIndex << " >= " <<
                        my_positions.size() << this->sendl;
                vVertices[vertexIndex] = sofa::defaulttype::Vector3();
            }
            else
            {
                vVertices[vertexIndex] = my_positions[vertexIndex];

            }


            const std::pair < int, int >& textureNormalIndex = *itTextureNormalIndex;
            if( vTexCoords.size() > 0 )
            {
                if( textureNormalIndex.first >= (int)my_texCoords.size() )
                {
                    this->serr << "Invalid index for texture: " << textureNormalIndex.first << " >= "
                            << my_texCoords.size() << this->sendl;
                    vTexCoords[vertexIndex] = sofa::defaulttype::Vector2();
                }
                else
                {
                    vTexCoords[vertexIndex] = my_texCoords[textureNormalIndex.first < 0 ? 0 : textureNormalIndex.first];
                }
            }
            /*        if( vNormals.size() > 0 ){
                      if( (int)my_normals.size() < textureNormalIndex.second ){
                        this->serr << "Invalid index for normal: " << textureNormalIndex.second << " > "
                          << my_normals.size() << this->sendl;
                        vNormals[vertexIndex] = sofa::defaulttype::Vector3();
                      }
                      else{
                       vNormals[vertexIndex] = my_normals[textureNormalIndex.second == -1 ? 0 : textureNormalIndex.second];
                      }
                    }    */
        }

    }

    positions.endEdit();
    edges.endEdit();
    triangles.endEdit();
    quads.endEdit();
    normalsList.endEdit();
    normalsIndexList.endEdit();
    materials.endEdit();
    texIndexList.endEdit();
    texCoordsList.endEdit();
    texCoords.endEdit();
    faceList.endEdit();
    vertices.endEdit();
    //normals.endEdit();
    return true;
}



// -----------------------------------------------------
// readMTL: read a wavefront material library file
//
//    model - properly initialized GLMmodel structure
//    name  - name of the material library
// -----------------------------------------------------
bool MeshObjLoader::readMTL(const char* filename, helper::vector <Material>& materials)
{
    sout << "MeshObjLoader::readMTL" << sendl;

    FILE* file;
    char buf[128];
    file = fopen(filename, "r");
    Material *mat = NULL;
    if (!file);//serr << "readMTL() failed: can't open material file " << filename << sendl;
    else
    {
        /* now, read in the data */
        while (fscanf(file, "%s", buf) != EOF)
        {

            switch (buf[0])
            {
            case '#':
                /* comment */
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        serr << "Error: MeshObjLoader: fgets function has encounter end of file. case #." << sendl;
                    else
                        serr << "Error: MeshObjLoader: fgets function has encounter an error. case #." << sendl;
                }
                break;
            case 'n':
                /* newmtl */
                if (mat != NULL)
                {
                    materials.push_back(*mat);
                    delete mat;
                    mat = NULL;
                }
                mat = new Material();
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        serr << "Error: MeshObjLoader: fgets function has encounter end of file. case n." << sendl;
                    else
                        serr << "Error: MeshObjLoader: fgets function has encounter an error. case n." << sendl;
                }

                sscanf(buf, "%s %s", buf, buf);
                mat->name = buf;
                break;
            case 'N':
                switch (buf[1])
                {
                case 'i':
                {
                    float optical_density;
                    if ( fscanf(file, "%f", &optical_density) == EOF )
                        serr << "Error: MeshObjLoader: fscanf function has encounter an error. case N i." << sendl;
                    break;
                }
                case 's':
                    if (fscanf(file, "%f", &mat->shininess) == EOF )
                        serr << "Error: MeshObjLoader: fscanf function has encounter an error. case N s." << sendl;
                    // wavefront shininess is from [0, 1000], so scale for OpenGL
                    //mat->shininess /= 1000.0;
                    //mat->shininess *= 128.0;
                    mat->useShininess = true;
                    break;
                default:
                    /* eat up rest of line */
                    if ( fgets(buf, sizeof(buf), file) == NULL)
                    {
                        if (feof (file) )
                            serr << "Error: MeshObjLoader: fgets function has encounter end of file. case N." << sendl;
                        else
                            serr << "Error: MeshObjLoader: fgets function has encounter an error. case N." << sendl;
                    }
                    break;
                }
                break;
            case 'K':
                switch (buf[1])
                {
                case 'd':
                    if ( fscanf(file, "%f %f %f", &mat->diffuse[0], &mat->diffuse[1], &mat->diffuse[2]) == EOF)
                        serr << "Error: MeshObjLoader: fscanf function has encounter an error. case K d." << sendl;
                    mat->useDiffuse = true;
                    /*sout << mat->name << " diffuse = "<<mat->diffuse[0]<<' '<<mat->diffuse[1]<<'*/ /*'<<mat->diffuse[2]<<sendl;*/
                    break;
                case 's':
                    if ( fscanf(file, "%f %f %f", &mat->specular[0], &mat->specular[1], &mat->specular[2]) == EOF)
                        serr << "Error: MeshObjLoader: fscanf function has encounter an error. case K s." << sendl;
                    mat->useSpecular = true;
                    /*sout << mat->name << " specular = "<<mat->specular[0]<<' '<<mat->specular[1]<<'*/ /*'<<mat->specular[2]<<sendl;*/
                    break;
                case 'a':
                    if ( fscanf(file, "%f %f %f", &mat->ambient[0], &mat->ambient[1], &mat->ambient[2]) == EOF)
                        serr << "Error: MeshObjLoader: fscanf function has encounter an error. case K a." << sendl;
                    mat->useAmbient = true;
                    /*sout << mat->name << " ambient = "<<mat->ambient[0]<<' '<<mat->ambient[1]<<'*/ /*'<<mat->ambient[2]<<sendl;*/
                    break;
                default:
                    /* eat up rest of line */
                    if ( fgets(buf, sizeof(buf), file) == NULL)
                    {
                        if (feof (file) )
                            serr << "Error: MeshObjLoader: fgets function has encounter end of file. case K." << sendl;
                        else
                            serr << "Error: MeshObjLoader: fgets function has encounter an error. case K." << sendl;
                    }
                    break;
                }
                break;
            case 'd':
            case 'T':
                // transparency value
                if ( fscanf(file, "%f", &mat->diffuse[3]) == EOF)
                    serr << "Error: MeshObjLoader: fscanf function has encounter an error. case T i." << sendl;
                break;
            default:
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        serr << "Error: MeshObjLoader: fgets function has encounter end of file. case default." << sendl;
                    else
                        serr << "Error: MeshObjLoader: fgets function has encounter an error. case default." << sendl;
                }
                break;
            }

        }
        fclose(file);
    }
    if (mat != NULL)
    {
        materials.push_back(*mat);
        delete mat;
        mat = NULL;
    }

    return true;
}



} // namespace loader

} // namespace component

} // namespace sofa

