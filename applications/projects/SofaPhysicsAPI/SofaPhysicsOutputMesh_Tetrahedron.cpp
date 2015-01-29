#include "SofaPhysicsAPI.h"
#include "SofaPhysicsOutputMesh_Tetrahedron_impl.h"

SofaPhysicsOutputMeshTetrahedron::SofaPhysicsOutputMeshTetrahedron()
    : impl(new Impl)
{
}

SofaPhysicsOutputMeshTetrahedron::~SofaPhysicsOutputMeshTetrahedron()
{
    delete impl;
}

const char* SofaPhysicsOutputMeshTetrahedron::getName() ///< (non-unique) name of this object
{
    return impl->getName();
}

ID SofaPhysicsOutputMeshTetrahedron::getID() ///< unique ID of this object
{
    return impl->getID();
}

unsigned int SofaPhysicsOutputMeshTetrahedron::getNbVertices() ///< number of vertices
{
    return impl->getNbVertices();
}
const Real* SofaPhysicsOutputMeshTetrahedron::getVPositions()  ///< vertices positions (Vec3)
{
    return impl->getVPositions();
}
const Real* SofaPhysicsOutputMeshTetrahedron::getVNormals()    ///< vertices normals   (Vec3)
{
    return impl->getVNormals();
}
const Real* SofaPhysicsOutputMeshTetrahedron::getVTexCoords()  ///< vertices UVs       (Vec2)
{
    return impl->getVTexCoords();
}
int SofaPhysicsOutputMeshTetrahedron::getTexCoordRevision()    ///< changes each time tex coord data are updated
{
    return impl->getTexCoordRevision();
}
int SofaPhysicsOutputMeshTetrahedron::getVerticesRevision()    ///< changes each time vertices data are updated
{
    return impl->getVerticesRevision();
}


unsigned int SofaPhysicsOutputMeshTetrahedron::getNbVAttributes()                    ///< number of vertices attributes
{
    return impl->getNbVAttributes();
}
unsigned int SofaPhysicsOutputMeshTetrahedron::getNbAttributes(int index)            ///< number of attributes in specified vertex attribute
{
    return impl->getNbAttributes(index);
}
const char*  SofaPhysicsOutputMeshTetrahedron::getVAttributeName(int index)          ///< vertices attribute name
{
    return impl->getVAttributeName(index);
}
int          SofaPhysicsOutputMeshTetrahedron::getVAttributeSizePerVertex(int index) ///< vertices attribute #
{
    return impl->getVAttributeSizePerVertex(index);
}
const Real*  SofaPhysicsOutputMeshTetrahedron::getVAttributeValue(int index)         ///< vertices attribute (Vec#)
{
    return impl->getVAttributeValue(index);
}
int          SofaPhysicsOutputMeshTetrahedron::getVAttributeRevision(int index)      ///< changes each time vertices attribute is updated
{
    return impl->getVAttributeRevision(index);
}

unsigned int SofaPhysicsOutputMeshTetrahedron::getNbLines() ///< number of lines
{
    return impl->getNbLines();
}
const Index* SofaPhysicsOutputMeshTetrahedron::getLines()   ///< lines topology (2 indices / line)
{
    return impl->getLines();
}
int SofaPhysicsOutputMeshTetrahedron::getLinesRevision()    ///< changes each time lines data is updated
{
    return impl->getLinesRevision();
}

unsigned int SofaPhysicsOutputMeshTetrahedron::getNbTriangles() ///< number of triangles
{
    return impl->getNbTriangles();
}
const Index* SofaPhysicsOutputMeshTetrahedron::getTriangles()   ///< triangles topology (3 indices / triangle)
{
    return impl->getTriangles();
}
int SofaPhysicsOutputMeshTetrahedron::getTrianglesRevision()    ///< changes each time triangles data is updated
{
    return impl->getTrianglesRevision();
}
unsigned int SofaPhysicsOutputMeshTetrahedron::getNbTetrahedrons()    ///< number of Tetrahedrons
{
    return impl->getNbTetrahedrons();
}
const Index* SofaPhysicsOutputMeshTetrahedron::getTetrahedrons()   /// Tetrahedrons topology
{
    return impl->getTetrahedrons();
}
int SofaPhysicsOutputMeshTetrahedron::getTetrahedronsRevision() ///< changes each time quads data is updated
{
    return impl->getTetrahedronsRevision();
}

unsigned int SofaPhysicsOutputMeshTetrahedron::getNbQuads() ///< number of quads
{
    return impl->getNbQuads();
}
const Index* SofaPhysicsOutputMeshTetrahedron::getQuads()   ///< quads topology (4 indices / quad)
{
    return impl->getQuads();
}
int SofaPhysicsOutputMeshTetrahedron::getQuadsRevision()    ///< changes each time quads data is updated
{
    return impl->getQuadsRevision();
}

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;


SofaPhysicsOutputMeshTetrahedron::Impl::Impl()
{
}

SofaPhysicsOutputMeshTetrahedron::Impl::~Impl()
{
}

void SofaPhysicsOutputMeshTetrahedron::Impl::setObject(SofaOutputMeshTetrahedron* o)
{
    sObj = o;
    sVA.clear();
    sofa::core::objectmodel::BaseContext* context = sObj->getContext();

    sofa::helper::vector<SofaVAttribute::SPtr> vSE;
    context->get<SofaVAttribute>(&vSE,sofa::core::objectmodel::BaseContext::Local);
    for (unsigned int i = 0; i < vSE.size(); ++i)
    {
        SofaVAttribute::SPtr se = vSE[i];
        if (se->getSEType() == sofa::core::visual::ShaderElement::SE_ATTRIBUTE)
        {
            sVA.push_back(se);
        }
    }
}

const char* SofaPhysicsOutputMeshTetrahedron::Impl::getName() ///< (non-unique) name of this object
{
    if (!sObj) return "";
    return sObj->getName().c_str();
}

ID SofaPhysicsOutputMeshTetrahedron::Impl::getID() ///< unique ID of this object
{
    return sObj.get();
}

unsigned int SofaPhysicsOutputMeshTetrahedron::Impl::getNbVertices() ///< number of vertices
{
    if (!sObj) return 0;
    // we cannot use getVertices() method directly as we need the Data revision
    Data<ResizableExtVector<Coord> > *data = &(sObj->m_positions);
    return (unsigned int) data->getValue().size();
}
const Real* SofaPhysicsOutputMeshTetrahedron::Impl::getVPositions()  ///< vertices positions (Vec3)
{
    Data<ResizableExtVector<Coord> > * data = &(sObj->m_positions);
    return (const Real*) data->getValue().getData();
    //return (const Real*)( sObj->nodes->getX() );
    //Data<ResizableExtVector<Coord> > * data =
    //    (!sObj->m_vertPosIdx.getValue().empty()) ?
    //    &(sObj->m_vertices2) : &(sObj->m_positions);
    //return (const Real*) data->getValue().getData();
}
const Real* SofaPhysicsOutputMeshTetrahedron::Impl::getVNormals()    ///< vertices normals   (Vec3)
{
    // normal for Tetrahedron is not supported yet
    return NULL;
    /*Data<ResizableExtVector<Deriv> > * data = &(sObj->m_vnormals);
    return (const Real*) data->getValue().getData();*/
}

const Real* SofaPhysicsOutputMeshTetrahedron::Impl::getVTexCoords()  ///< vertices UVs       (Vec2)
{
    // texture coordinates for Tetrahedron is not supported yet
    return NULL;
    /*Data<ResizableExtVector<TexCoord> > * data = &(sObj->m_vtexcoords);
    return (const Real*) data->getValue().getData();*/
}

int SofaPhysicsOutputMeshTetrahedron::Impl::getTexCoordRevision()    ///< changes each time tex coord data are updated
{
    // texture coordinates for Tetrahedron is not supported yet
    return -1;
    //Data<ResizableExtVector<TexCoord> > * data = &(sObj->m_vtexcoords);
    //data->getValue(); // make sure the data is updated
    //return data->getCounter();
}

int SofaPhysicsOutputMeshTetrahedron::Impl::getVerticesRevision()    ///< changes each time vertices data are updated
{
    Data<ResizableExtVector<Coord> > * data = &(sObj->m_positions);
    data->updateIfDirty(); // make sure the data is updated
    return data->getCounter();
}


unsigned int SofaPhysicsOutputMeshTetrahedron::Impl::getNbVAttributes()                    ///< number of vertices attributes
{
    return sVA.size();
}

unsigned int SofaPhysicsOutputMeshTetrahedron::Impl::getNbAttributes(int index)            ///< number of attributes in specified vertex attribute
{
    if ((unsigned)index >= sVA.size())
      return 0;
    else 
      return sVA[index]->getSETotalSize();
     // return dynamic_cast< Data<ResizableExtVector<Real> >* >(sVA[index]->getSEValue())->getValue().size();
}

const char*  SofaPhysicsOutputMeshTetrahedron::Impl::getVAttributeName(int index)          ///< vertices attribute name
{
    // Quick fix: buffer to use for return value
    // May remove if getSEID() is made to return a non-temp std::string
    static std::string buffer;
  
    if ((unsigned)index >= sVA.size())
        return "";
    else {
        buffer= sVA[index]->getSEID();
        return buffer.c_str();
    }
}

int SofaPhysicsOutputMeshTetrahedron::Impl::getVAttributeSizePerVertex(int index) ///< vertices attribute #
{
    if ((unsigned)index >= sVA.size())
        return 0;
    else
        return sVA[index]->getSESizePerVertex();
}

const Real*  SofaPhysicsOutputMeshTetrahedron::Impl::getVAttributeValue(int index)         ///< vertices attribute (Vec#)
{
    if ((unsigned)index >= sVA.size())
        return NULL;
    else
        return (const Real*)((ResizableExtVector<Real>*)sVA[index]->getSEValue()->getValueVoidPtr())->getData();
}

int SofaPhysicsOutputMeshTetrahedron::Impl::getVAttributeRevision(int index)      ///< changes each time vertices attribute is updated
{
    if ((unsigned)index >= sVA.size())
        return 0;
    else
    {
        sVA[index]->getSEValue()->getValueVoidPtr(); // make sure the data is updated
        return sVA[index]->getSEValue()->getCounter();
    }
}


unsigned int SofaPhysicsOutputMeshTetrahedron::Impl::getNbLines() ///< number of lines
{
    return 0; // not yet supported
}
const Index* SofaPhysicsOutputMeshTetrahedron::Impl::getLines()   ///< lines topology (2 indices / line)
{
    return NULL;
}
int SofaPhysicsOutputMeshTetrahedron::Impl::getLinesRevision()    ///< changes each time lines data is updated
{
    return 0;
}

unsigned int SofaPhysicsOutputMeshTetrahedron::Impl::getNbTriangles() ///< number of triangles
{
    return sObj->m_topology->getNbTriangles();
    //Data<ResizableExtVector<Triangle> > * data = &(sObj->m_triangles);
    //return (unsigned int) data->getValue().size();
}
const Index* SofaPhysicsOutputMeshTetrahedron::Impl::getTriangles()   ///< triangles topology (3 indices / triangle)
{
    return NULL;
    //return (const Index*) sObj->topo->getTriangles();
    //Data<ResizableExtVector<Triangle> > * data = &(sObj->m_triangles);
    //return (const Index*) data->getValue().getData();
}
int SofaPhysicsOutputMeshTetrahedron::Impl::getTrianglesRevision()    ///< changes each time triangles data is updated
{
    return 0;
    //Data<ResizableExtVector<Triangle> > * data = &(sObj->m_triangles);
    //data->getValue(); // make sure the data is updated
    //return data->getCounter();
}

unsigned int SofaPhysicsOutputMeshTetrahedron::Impl::getNbQuads() ///< number of quads
{
    return sObj->m_topology->getNbQuads();
    /*Data<ResizableExtVector<Quad> > * data = &(sObj->m_quads);
    return (unsigned int) data->getValue().size();*/
}
const Index* SofaPhysicsOutputMeshTetrahedron::Impl::getQuads()   ///< quads topology (4 indices / quad)
{
    return NULL;
    /*return (const Index*)sObj->topo->getQuads();*/
    /*Data<ResizableExtVector<Quad> > * data = &(sObj->m_quads);
    return (const Index*) data->getValue().getData();*/
}
int SofaPhysicsOutputMeshTetrahedron::Impl::getQuadsRevision()    ///< changes each time quads data is updated
{
    return 0;
    //Data<ResizableExtVector<Quad> > * data = &(sObj->m_quads);
    //data->getValue(); // make sure the data is updated
    //return data->getCounter();
}
unsigned int SofaPhysicsOutputMeshTetrahedron::Impl::getNbTetrahedrons()  ///< number of Tetrahedrons
{
    return sObj->m_topology->getNbTetrahedra();
}
const Index* SofaPhysicsOutputMeshTetrahedron::Impl::getTetrahedrons()  ///< Tetrahedrons topology( 4 indices/Tetrahedron )
{
    Data<ResizableExtVector<Tetrahedron> > * data = &(sObj->m_tetrahedrons);
    return (const Index*) data->getValue().getData();
}
int SofaPhysicsOutputMeshTetrahedron::Impl::getTetrahedronsRevision()   ///< changes each time tetrahedrons data is updated
{
    Data<ResizableExtVector<Tetrahedron> > * data = &(sObj->m_tetrahedrons);
    return data->getCounter();
}