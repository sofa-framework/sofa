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
#include <sofa/core/ObjectFactory.h>
#include <SofaAssimp/AssimpLoader.h>


#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/cimport.h>        // Plain-C interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags


namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(AssimpLoader)

int AssimpLoaderClass = core::RegisterObject("Specific mesh loader for STL file format.")
        .add< AssimpLoader >()
        ;


AssimpLoader::AssimpLoader()
    : MeshLoader()
    , m_assimpScene(NULL)
{

    pp_Loadsteps =
        aiProcess_JoinIdenticalVertices;//  */  //| //join identical vertices/ optimize indexing
                                        // pp_Loadsteps =
                                        //         aiProcess_GenSmoothNormals       | // generate smooth normal vectors if not existing
                                        //         aiProcess_JoinIdenticalVertices  | //join identical vertices/ optimize indexing
                                        //         aiProcess_ValidateDataStructure  | // perform a full validation of the loader's output
                                        //         aiProcess_ImproveCacheLocality   | // improve the cache locality of the output vertices
                                        //         aiProcess_FindDegenerates        | // remove degenerated polygons from the import
                                        //         aiProcess_FindInvalidData        | // detect invalid model data, such as invalid normal vectors
                                        //         aiProcess_OptimizeMeshes   ;
}

AssimpLoader::~AssimpLoader()
{
    if (m_assimpScene != NULL)
    {
        aiReleaseImport(m_assimpScene);
        //delete m_assimpScene;
        m_assimpScene = NULL;
    }
}


bool AssimpLoader::load()
{
    // -- Loading file
    if (!canLoad())
        return false;

    const char* filename = m_filename.getFullPath().c_str();
    
    // Create an instance of the Importer class
    Assimp::Importer importer;
    bool res = importer.IsExtensionSupported(m_filename.getExtension());
    if (!res)
    {
        msg_error() << "Extension not handled: " << m_filename.getExtension() << " . Assimp scene not created.";
        return false;
    }
    std::cout << m_filename.getExtension() << std::endl;
    
    // And have it read the given file with some example postprocessing
    // Usually - if speed is not the most important aspect for you - you'll 
    // propably to request more postprocessing than we do in this example.
    m_assimpScene = (aiScene*)aiImportFile(filename, pp_Loadsteps);
       /* aiProcess_CalcTangentSpace |
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType);*/


    // If the import failed, report it
    if (!m_assimpScene)
    {
        msg_error() << "Assimp scene from file: '" << m_filename << "' creation failed with error: " << importer.GetErrorString();
        return false;
    }
    // Now we can access the file's contents. 
    //DoTheSceneProcessing(scene);
    // We're done. Everything will be cleaned up by the importer destructor
    return convertAssimpScene();
}


bool AssimpLoader::convertAssimpScene()
{
    if (!m_assimpScene)
        return false;

    msg_info() << "m_assimpScene->mNumMeshes: " << m_assimpScene->mNumMeshes;
    msg_info() << "m_assimpScene->mNumMaterials: " << m_assimpScene->mNumMaterials;

    WriteAccessor<Data<helper::vector<sofa::defaulttype::Vec<3, SReal> > > > waPositions = d_positions;
    WriteAccessor<Data<helper::vector<sofa::defaulttype::Vec<3, SReal> > > > waNormals = d_normals;

    WriteAccessor<Data<helper::vector< Edge > > > waEdges = d_edges; 
    WriteAccessor<Data<helper::vector< Quad > > > waQuads = d_quads;
    WriteAccessor<Data<helper::vector< Triangle > > > waTriangles = d_triangles;
    
    // Clear potential buffer previous init.
    waPositions.clear();
    waNormals.clear();

    waEdges.clear();
    waTriangles.clear();
    waQuads.clear();

    for (unsigned int i = 0; i<m_assimpScene->mNumMeshes; ++i)
    {
        aiMesh* currentMesh = m_assimpScene->mMeshes[i]; //The ith mesh of the array of meshes.
        unsigned int nbr_pos = currentMesh->mNumVertices;
        
        unsigned int cpt_pos = waPositions.size();
        unsigned int cpt_norm = waNormals.size();
        //unsigned int cpt_uv = m_texCoords.size();

        if (cpt_pos != cpt_norm /*|| cpt_pos != cpt_uv*/)
            msg_warning() << "No conscistent number of element in mesh: pos: " << cpt_pos
            << " normals: " << cpt_norm/* << " texCoords: " << cpt_uv*/;

        waPositions.resize(waPositions.size() + nbr_pos);
        waNormals.resize(waNormals.size() + nbr_pos);
        //m_texCoords.resize(m_texCoords.size() + nbr_pos);

        for (unsigned int j = 0; j<nbr_pos; ++j)
        {
            // create position array
            sofa::defaulttype::Vec<3, SReal>& pos = waPositions[j + cpt_pos];
            const aiVector3D& aiPos = currentMesh->mVertices[j];
            pos[0] = aiPos.x;
            pos[1] = aiPos.y;
            pos[2] = aiPos.z;

            // create normal array
            sofa::defaulttype::Vec<3, SReal>& normal = waNormals[j + cpt_norm];
            const aiVector3D& aiNorm = currentMesh->mNormals[j];
            normal[0] = aiNorm.x;
            normal[1] = aiNorm.y;
            normal[2] = aiNorm.z;
        }

        //// create Texcoords array
        //aiVector3D* aiUVs = currentMesh->mTextureCoords[0];
        //if (aiUVs)
        //{
        //    for (unsigned int j = 0; j<nbr_pos; ++j)
        //    {
        //        const aiVector3D& aiUV = aiUVs[j];
        //        TexCoord& uv = m_texCoords[j + cpt_uv];
        //        uv[0] = aiUV.x;
        //        uv[1] = aiUV.y;
        //    }
        //}


        // create faces
        unsigned int nbr_faces = currentMesh->mNumFaces;
        for (unsigned int j = 0; j<nbr_faces; ++j)
        {
            const aiFace& my_face = currentMesh->mFaces[j];
            unsigned int nbr_id = my_face.mNumIndices;

            if (nbr_id == 2)
                waEdges.push_back(Edge(my_face.mIndices[0] + cpt_pos, my_face.mIndices[1] + cpt_pos));
            else if (nbr_id == 3)
                waTriangles.push_back(Triangle(my_face.mIndices[0] + cpt_pos, my_face.mIndices[1] + cpt_pos, my_face.mIndices[2] + cpt_pos));
            else if (nbr_id == 4)
                waQuads.push_back(Quad(my_face.mIndices[0] + cpt_pos, my_face.mIndices[1] + cpt_pos, my_face.mIndices[2] + cpt_pos, my_face.mIndices[3] + cpt_pos));
        }
    }

    return true;
}

} // namespace loader

} // namespace component

} // namespace sofa

