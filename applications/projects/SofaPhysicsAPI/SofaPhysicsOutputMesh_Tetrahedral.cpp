#include "SofaPhysicsAPI.h"
#include "SofaPhysicsOutputMesh_Tetrahedral_impl.h"

SofaPhysicsOutputMeshTetrahedral::SofaPhysicsOutputMeshTetrahedral()
    : impl(new Impl)
{
}

SofaPhysicsOutputMeshTetrahedral::~SofaPhysicsOutputMeshTetrahedral()
{
    delete impl;
}

const char* SofaPhysicsOutputMeshTetrahedral::getName() ///< (non-unique) name of this object
{
    return impl->getName();
}

ID SofaPhysicsOutputMeshTetrahedral::getID() ///< unique ID of this object
{
    return impl->getID();
}

unsigned int SofaPhysicsOutputMeshTetrahedral::getNbVertices() ///< number of vertices
{
    return impl->getNbVertices();
}
const Real* SofaPhysicsOutputMeshTetrahedral::getVPositions()  ///< vertices positions (Vec3)
{
    return impl->getVPositions();
}
const Real* SofaPhysicsOutputMeshTetrahedral::getVNormals()    ///< vertices normals   (Vec3)
{
    return impl->getVNormals();
}
const Real* SofaPhysicsOutputMeshTetrahedral::getVTexCoords()  ///< vertices UVs       (Vec2)
{
    return impl->getVTexCoords();
}
int SofaPhysicsOutputMeshTetrahedral::getTexCoordRevision()    ///< changes each time tex coord data are updated
{
    return impl->getTexCoordRevision();
}
int SofaPhysicsOutputMeshTetrahedral::getVerticesRevision()    ///< changes each time vertices data are updated
{
    return impl->getVerticesRevision();
}


unsigned int SofaPhysicsOutputMeshTetrahedral::getNbVAttributes()                    ///< number of vertices attributes
{
    return impl->getNbVAttributes();
}
const char*  SofaPhysicsOutputMeshTetrahedral::getVAttributeName(int index)          ///< vertices attribute name
{
    return impl->getVAttributeName(index);
}
int          SofaPhysicsOutputMeshTetrahedral::getVAttributeSizePerVertex(int index) ///< vertices attribute #
{
    return impl->getVAttributeSizePerVertex(index);
}
const Real*  SofaPhysicsOutputMeshTetrahedral::getVAttributeValue(int index)         ///< vertices attribute (Vec#)
{
    return impl->getVAttributeValue(index);
}
int          SofaPhysicsOutputMeshTetrahedral::getVAttributeRevision(int index)      ///< changes each time vertices attribute is updated
{
    return impl->getVAttributeRevision(index);
}

unsigned int SofaPhysicsOutputMeshTetrahedral::getNbLines() ///< number of lines
{
    return impl->getNbLines();
}
const Index* SofaPhysicsOutputMeshTetrahedral::getLines()   ///< lines topology (2 indices / line)
{
    return impl->getLines();
}
int SofaPhysicsOutputMeshTetrahedral::getLinesRevision()    ///< changes each time lines data is updated
{
    return impl->getLinesRevision();
}

unsigned int SofaPhysicsOutputMeshTetrahedral::getNbTriangles() ///< number of triangles
{
    return impl->getNbTriangles();
}
const Index* SofaPhysicsOutputMeshTetrahedral::getTriangles()   ///< triangles topology (3 indices / triangle)
{
    return impl->getTriangles();
}
int SofaPhysicsOutputMeshTetrahedral::getTrianglesRevision()    ///< changes each time triangles data is updated
{
    return impl->getTrianglesRevision();
}

unsigned int SofaPhysicsOutputMeshTetrahedral::getNbQuads() ///< number of quads
{
    return impl->getNbQuads();
}
const Index* SofaPhysicsOutputMeshTetrahedral::getQuads()   ///< quads topology (4 indices / quad)
{
    return impl->getQuads();
}
int SofaPhysicsOutputMeshTetrahedral::getQuadsRevision()    ///< changes each time quads data is updated
{
    return impl->getQuadsRevision();
}

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;


SofaPhysicsOutputMeshTetrahedral::Impl::Impl()
{
}

SofaPhysicsOutputMeshTetrahedral::Impl::~Impl()
{
}

void SofaPhysicsOutputMeshTetrahedral::Impl::setObject(SofaOutputMeshTetrahedral* o)
{
    sObj = o;
    //sVA.clear();
    sofa::core::objectmodel::BaseContext* context = sObj->getContext();
    // SofaVAttribute is not supported yet
    //sofa::helper::vector<SofaVAttribute::SPtr> vSE;
    //context->get<SofaVAttribute>(&vSE,sofa::core::objectmodel::BaseContext::Local);
    //for (unsigned int i = 0; i < vSE.size(); ++i)
    //{
    //    SofaVAttribute::SPtr se = vSE[i];
    //    if (se->getSEType() == sofa::core::visual::ShaderElement::SE_ATTRIBUTE)
    //    {
    //        sVA.push_back(se);
    //    }
    //}
}

const char* SofaPhysicsOutputMeshTetrahedral::Impl::getName() ///< (non-unique) name of this object
{
    if (!sObj) return "";
    return sObj->getName().c_str();
}

ID SofaPhysicsOutputMeshTetrahedral::Impl::getID() ///< unique ID of this object
{
    return sObj.get();
}

unsigned int SofaPhysicsOutputMeshTetrahedral::Impl::getNbVertices() ///< number of vertices
{
    if (!sObj) return 0;
    // we cannot use getVertices() method directly as we need the Data revision
    Data<ResizableExtVector<Coord> > *data = &(sObj->m_positions);
    return (unsigned int) data->getValue().size();
}
const Real* SofaPhysicsOutputMeshTetrahedral::Impl::getVPositions()  ///< vertices positions (Vec3)
{
    Data<ResizableExtVector<Coord> > * data = &(sObj->m_positions);
    return (const Real*) data->getValue().getData();
    //return (const Real*)( sObj->nodes->getX() );
    //Data<ResizableExtVector<Coord> > * data =
    //    (!sObj->m_vertPosIdx.getValue().empty()) ?
    //    &(sObj->m_vertices2) : &(sObj->m_positions);
    //return (const Real*) data->getValue().getData();
}
const Real* SofaPhysicsOutputMeshTetrahedral::Impl::getVNormals()    ///< vertices normals   (Vec3)
{
    // normal for Tetrahedral is not supported yet
    return NULL;
    /*Data<ResizableExtVector<Deriv> > * data = &(sObj->m_vnormals);
    return (const Real*) data->getValue().getData();*/
}

const Real* SofaPhysicsOutputMeshTetrahedral::Impl::getVTexCoords()  ///< vertices UVs       (Vec2)
{
    // texture coordinates for Tetrahedral is not supported yet
    return NULL;
    /*Data<ResizableExtVector<TexCoord> > * data = &(sObj->m_vtexcoords);
    return (const Real*) data->getValue().getData();*/
}

int SofaPhysicsOutputMeshTetrahedral::Impl::getTexCoordRevision()    ///< changes each time tex coord data are updated
{
    // texture coordinates for Tetrahedral is not supported yet
    return -1;
    //Data<ResizableExtVector<TexCoord> > * data = &(sObj->m_vtexcoords);
    //data->getValue(); // make sure the data is updated
    //return data->getCounter();
}

int SofaPhysicsOutputMeshTetrahedral::Impl::getVerticesRevision()    ///< changes each time vertices data are updated
{
    Data<ResizableExtVector<Coord> > * data = &(sObj->m_positions);
    data->updateIfDirty(); // make sure the data is updated
    return data->getCounter();
}


unsigned int SofaPhysicsOutputMeshTetrahedral::Impl::getNbVAttributes()                    ///< number of vertices attributes
{
    // no vertex attribute is supported yet
    return 0;
    /*return sVA.size();*/
}

const char*  SofaPhysicsOutputMeshTetrahedral::Impl::getVAttributeName(int index)          ///< vertices attribute name
{
    // no vertex attribute is supported yet
    return "";
    //// Quick fix: buffer to use for return value
    //// May remove if getSEID() is made to return a non-temp std::string
    //static std::string buffer;
  
    //if ((unsigned)index >= sVA.size())
    //    return "";
    //else {
    //    buffer= sVA[index]->getSEID();
    //    return buffer.c_str();
    //}
}

int SofaPhysicsOutputMeshTetrahedral::Impl::getVAttributeSizePerVertex(int index) ///< vertices attribute #
{
    // no vertex attribute is supported yet
    return 0;
    //if ((unsigned)index >= sVA.size())
    //    return 0;
    //else
    //    return sVA[index]->getSESizePerVertex();
}

const Real*  SofaPhysicsOutputMeshTetrahedral::Impl::getVAttributeValue(int index)         ///< vertices attribute (Vec#)
{
    // no vertex attribute is supported yet
    return 0;
    /*if ((unsigned)index >= sVA.size())
        return NULL;
    else
        return (Real*)((ResizableExtVector<Real>*)sVA[index]->getSEValue()->getValueVoidPtr())->getData();*/
}

int SofaPhysicsOutputMeshTetrahedral::Impl::getVAttributeRevision(int index)      ///< changes each time vertices attribute is updated
{
    // no vertex attribute is supported yet
    return 0;
    //if ((unsigned)index >= sVA.size())
    //    return 0;
    //else
    //{
    //    sVA[index]->getSEValue()->getValueVoidPtr(); // make sure the data is updated
    //    return sVA[index]->getSEValue()->getCounter();
    //}
}


unsigned int SofaPhysicsOutputMeshTetrahedral::Impl::getNbLines() ///< number of lines
{
    return 0; // not yet supported
}
const Index* SofaPhysicsOutputMeshTetrahedral::Impl::getLines()   ///< lines topology (2 indices / line)
{
    return NULL;
}
int SofaPhysicsOutputMeshTetrahedral::Impl::getLinesRevision()    ///< changes each time lines data is updated
{
    return 0;
}

unsigned int SofaPhysicsOutputMeshTetrahedral::Impl::getNbTriangles() ///< number of triangles
{
    return sObj->m_topology->getNbTriangles();
    //Data<ResizableExtVector<Triangle> > * data = &(sObj->m_triangles);
    //return (unsigned int) data->getValue().size();
}
const Index* SofaPhysicsOutputMeshTetrahedral::Impl::getTriangles()   ///< triangles topology (3 indices / triangle)
{
    return NULL;
    //return (const Index*) sObj->topo->getTriangles();
    //Data<ResizableExtVector<Triangle> > * data = &(sObj->m_triangles);
    //return (const Index*) data->getValue().getData();
}
int SofaPhysicsOutputMeshTetrahedral::Impl::getTrianglesRevision()    ///< changes each time triangles data is updated
{
    return 0;
    //Data<ResizableExtVector<Triangle> > * data = &(sObj->m_triangles);
    //data->getValue(); // make sure the data is updated
    //return data->getCounter();
}

unsigned int SofaPhysicsOutputMeshTetrahedral::Impl::getNbQuads() ///< number of quads
{
    return sObj->m_topology->getNbQuads();
    /*Data<ResizableExtVector<Quad> > * data = &(sObj->m_quads);
    return (unsigned int) data->getValue().size();*/
}
const Index* SofaPhysicsOutputMeshTetrahedral::Impl::getQuads()   ///< quads topology (4 indices / quad)
{
    return NULL;
    /*return (const Index*)sObj->topo->getQuads();*/
    /*Data<ResizableExtVector<Quad> > * data = &(sObj->m_quads);
    return (const Index*) data->getValue().getData();*/
}
int SofaPhysicsOutputMeshTetrahedral::Impl::getQuadsRevision()    ///< changes each time quads data is updated
{
    return 0;
    //Data<ResizableExtVector<Quad> > * data = &(sObj->m_quads);
    //data->getValue(); // make sure the data is updated
    //return data->getCounter();
}
