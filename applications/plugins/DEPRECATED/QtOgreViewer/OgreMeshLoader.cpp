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
#include "OgreMeshLoader.h"
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{
namespace component
{
namespace loader
{

SOFA_DECL_CLASS(OgreMeshLoader)

int OgreMeshLoaderClass = core::RegisterObject("Specific mesh loader for Ogre compiled mesh files.")
        .add< OgreMeshLoader >()
        ;

OgreMeshLoader::OgreMeshLoader():
    core::loader::MeshLoader(),
    texCoords(initData(&texCoords,"texcoords","Texture coordinates of the mesh"))
{
}

bool OgreMeshLoader::canLoad()
{
    bool canload = false;
    if( Ogre::MeshManager::getSingletonPtr() == NULL )
    {
        // it is OK to step here from the Modeler...
        // we do not have a clear policy to init and tidy external resources from a plugin.
        // It is the case for Ogre (see Ogre::Root::initialise() and Ogre::Root::shutdown()
        this->sout << "Ogre::MeshManager NULL." << sendl;
        canload = false;
    }
    else
    {
        canload = core::loader::MeshLoader::canLoad();
    }
    return canload;
}

bool OgreMeshLoader::load()
{
    sout << "Loading MESH file: " << m_filename << sendl;
    bool fileRead = false;
    try
    {
        Ogre::MeshPtr mesh = Ogre::MeshManager::getSingleton().load(m_filename.getFullPath(),"General");
        fileRead = this->readMesh(mesh.getPointer());
    }
    catch (Ogre::Exception& e)
    {
        serr << e.getFullDescription() << sendl;
        fileRead = false;
    }
    return fileRead;
}

bool OgreMeshLoader::readMesh(Ogre::Mesh* mesh)
{
    using namespace Ogre;
    helper::vector<sofa::defaulttype::Vector3>& vertices = *(positions.beginEdit());
    helper::vector< helper::fixed_array <unsigned int,3> >& indices = *(triangles.beginEdit());
    helper::vector<sofa::defaulttype::Vector2>& coords = *(texCoords.beginEdit());
    helper::vector<sofa::defaulttype::Vector3>& normal = *(normals.beginEdit());

    bool added_shared = false;
    size_t vertex_count   = 0;
    size_t index_count    = 0;
    size_t current_offset = 0;
    size_t shared_offset  = 0;
    size_t next_offset    = 0;
    size_t index_offset   = 0;

    // Calculate how many vertices and indices we're going to need
    for(int i = 0; i < mesh->getNumSubMeshes(); i++)
    {
        SubMesh* submesh = mesh->getSubMesh(i);

        // We only need to add the shared vertices once
        if(submesh->useSharedVertices)
        {
            if(!added_shared)
            {
                VertexData* vertex_data = mesh->sharedVertexData;
                vertex_count += vertex_data->vertexCount;
                added_shared = true;
            }
        }
        else
        {
            VertexData* vertex_data = submesh->vertexData;
            vertex_count += vertex_data->vertexCount;
        }

        // Add the indices
        Ogre::IndexData* index_data = submesh->indexData;
        index_count += index_data->indexCount;
    }

    // Allocate space for the vertices and indices
    vertices.reserve(vertex_count);
    coords.reserve(vertex_count);
    indices.reserve(index_count / 3);

    added_shared = false;

    // Run through the submeshes again, adding the data into the arrays
    for(int i = 0; i < mesh->getNumSubMeshes(); i++)
    {
        SubMesh* submesh = mesh->getSubMesh(i);

        Ogre::VertexData* vertex_data = submesh->useSharedVertices ? mesh->sharedVertexData : submesh->vertexData;
        if((!submesh->useSharedVertices)||(submesh->useSharedVertices && !added_shared))
        {
            if(submesh->useSharedVertices)
            {
                added_shared = true;
                shared_offset = current_offset;
            }
            readMeshVertices(vertex_data,vertices);
            readMeshTexCoords(vertex_data,coords);
            readMeshNormals(vertex_data,normal);
            next_offset += vertex_data->vertexCount;
        }

        Ogre::IndexData* index_data = submesh->indexData;

        size_t numTris = index_data->indexCount / 3;
        unsigned short* pShort;
        unsigned int* pInt;
        Ogre::HardwareIndexBufferSharedPtr ibuf = index_data->indexBuffer;
        bool use32bitindexes = (ibuf->getType() == Ogre::HardwareIndexBuffer::IT_32BIT);
        if (use32bitindexes) pInt = static_cast<unsigned int*>(ibuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));
        else pShort = static_cast<unsigned short*>(ibuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));

        for(size_t k = 0; k < numTris; ++k)
        {
            size_t offset = (submesh->useSharedVertices)?shared_offset:current_offset;
            helper::fixed_array <unsigned int,3> tri;
            unsigned int vindex = use32bitindexes? *pInt++ : *pShort++;
            tri[0] = vindex + offset;
            vindex = use32bitindexes? *pInt++ : *pShort++;
            tri[1] = vindex + offset;
            vindex = use32bitindexes? *pInt++ : *pShort++;
            tri[2] = vindex + offset;
            indices.push_back(tri);
            index_offset += 3;
        }
        ibuf->unlock();
        current_offset = next_offset;
    }
    positions.endEdit();
    triangles.endEdit();
    texCoords.endEdit();
    normals.endEdit();

    return true;
}

void OgreMeshLoader::readMeshVertices(Ogre::VertexData* vertexData,
        helper::vector< sofa::defaulttype::Vector3 >& vertices)
{
    const Ogre::VertexElement* posElem =
        vertexData->vertexDeclaration->findElementBySemantic(Ogre::VES_POSITION);

    if( !posElem ) return;

    Ogre::HardwareVertexBufferSharedPtr vbuf =
        vertexData->vertexBufferBinding->getBuffer(posElem->getSource());
    unsigned char* vertex =
        static_cast<unsigned char*>(vbuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));
    Ogre::Real* pReal;
    for(size_t j = 0; j < vertexData->vertexCount; ++j, vertex += vbuf->getVertexSize())
    {
        posElem->baseVertexPointerToElement(vertex, &pReal);
        sofa::defaulttype::Vector3 pt;
        pt[0] = (*pReal++);
        pt[1] = (*pReal++);
        pt[2] = (*pReal++);
        vertices.push_back(pt);
    }
    vbuf->unlock();
}

/*   void OgreMeshLoader::readMeshIndices(Ogre::VertexData* vertexData,
     helper::vector< helper::fixed_array <unsigned int,3> >& indices)
   {

   }*/

void OgreMeshLoader::readMeshNormals(Ogre::VertexData* vertexData,
        helper::vector< sofa::defaulttype::Vector3 >& normal)
{
    const Ogre::VertexElement* normalElem =
        vertexData->vertexDeclaration->findElementBySemantic(Ogre::VES_NORMAL);

    if( ! normalElem ) return;

    Ogre::HardwareVertexBufferSharedPtr vbuf =
        vertexData->vertexBufferBinding->getBuffer(normalElem->getSource());
    unsigned char* vertex =
        static_cast<unsigned char*>(vbuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));
    Ogre::Real* pReal;
    for(size_t j = 0; j < vertexData->vertexCount; ++j, vertex += vbuf->getVertexSize())
    {
        normalElem->baseVertexPointerToElement(vertex, &pReal);
        sofa::defaulttype::Vector3 n;
        n[0] = (*pReal++);
        n[1] = (*pReal++);
        n[2] = (*pReal++);
        normal.push_back(n);
    }
    vbuf->unlock();
}

void OgreMeshLoader::readMeshTexCoords(Ogre::VertexData* vertexData,
        helper::vector< sofa::defaulttype::Vector2>& coords)
{
    const Ogre::VertexElement* coordElem =
        vertexData->vertexDeclaration->findElementBySemantic(Ogre::VES_TEXTURE_COORDINATES);

    if( !coordElem ) return;

    Ogre::HardwareVertexBufferSharedPtr vbuf =
        vertexData->vertexBufferBinding->getBuffer(coordElem->getSource());
    unsigned char* vertex =
        static_cast<unsigned char*>(vbuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));
    Ogre::Real* pReal;
    for(size_t j = 0; j < vertexData->vertexCount; ++j, vertex += vbuf->getVertexSize())
    {
        coordElem->baseVertexPointerToElement(vertex, &pReal);
        sofa::defaulttype::Vector2 coord;
        coord[0] = (*pReal++);
        coord[1] = (*pReal++);
        coords.push_back(coord);
    }
    vbuf->unlock();

}

} //namespace loader
} //namespace component
} // namespace sofa
