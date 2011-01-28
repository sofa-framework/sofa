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

OgreMeshLoader::OgreMeshLoader()
{

}

bool OgreMeshLoader::load()
{
    sout << "Loading MESH file: " << m_filename << sendl;
    bool fileRead = false;

    /*   std::string parentDir = sofa::helper::system::SetDirectory::GetParentDir(filename);
       Ogre::ResourceGroupManager* rgm = Ogre::ResourceGroupManager::getSingletonPtr();

       if( !rgm->resourceLocationExists(parentDir,"General") || !rgm->resourceLocationExists(parentDir,"External") )
       {
         rgm->addResourceLocation(parentDir,"FileSystem","External");
         rgm->initialiseResourceGroup("External");
       }

       std::string meshName = sofa::helper::system::SetDirectory::GetFileName(filename);
       */
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
    typedef helper::fixed_array <unsigned int,3> Vec3ui;
    using namespace Ogre;
    helper::vector<sofa::defaulttype::Vector3>& vertices = *(positions.beginEdit());
    helper::vector< Vec3ui >& indices = *(triangles.beginEdit());


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

            const Ogre::VertexElement* posElem = vertex_data->vertexDeclaration->findElementBySemantic(Ogre::VES_POSITION);
            Ogre::HardwareVertexBufferSharedPtr vbuf = vertex_data->vertexBufferBinding->getBuffer(posElem->getSource());
            unsigned char* vertex = static_cast<unsigned char*>(vbuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));
            Ogre::Real* pReal;

            for(size_t j = 0; j < vertex_data->vertexCount; ++j, vertex += vbuf->getVertexSize())
            {
                posElem->baseVertexPointerToElement(vertex, &pReal);

                sofa::defaulttype::Vector3 pt;

                pt[0] = (*pReal++);
                pt[1] = (*pReal++);
                pt[2] = (*pReal++);

                //pt = (orient * (pt * scale)) + position;
                vertices.push_back(pt);
                //vertices[current_offset + j] = pt;
            }
            vbuf->unlock();
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
            Vec3ui tri;
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



    return true;
}

} //namespace loader
} //namespace component
} // namespace sofa
