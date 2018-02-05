/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SofaPhysicsAPI.h"
#include "SofaPhysicsOutputMesh_impl.h"

SofaPhysicsOutputMesh::SofaPhysicsOutputMesh()
    : impl(new Impl)
{
}

SofaPhysicsOutputMesh::~SofaPhysicsOutputMesh()
{
    delete impl;
}

const char* SofaPhysicsOutputMesh::getName() ///< (non-unique) name of this object
{
    return impl->getName();
}

ID SofaPhysicsOutputMesh::getID() ///< unique ID of this object
{
    return impl->getID();
}

unsigned int SofaPhysicsOutputMesh::getNbVertices() ///< number of vertices
{
    return impl->getNbVertices();
}
const Real* SofaPhysicsOutputMesh::getVPositions()  ///< vertices positions (Vec3)
{
    return impl->getVPositions();
}
const Real* SofaPhysicsOutputMesh::getVNormals()    ///< vertices normals   (Vec3)
{
    return impl->getVNormals();
}
const Real* SofaPhysicsOutputMesh::getVTexCoords()  ///< vertices UVs       (Vec2)
{
    return impl->getVTexCoords();
}
int SofaPhysicsOutputMesh::getTexCoordRevision()    ///< changes each time tex coord data are updated
{
    return impl->getTexCoordRevision();
}
int SofaPhysicsOutputMesh::getVerticesRevision()    ///< changes each time vertices data are updated
{
    return impl->getVerticesRevision();
}


unsigned int SofaPhysicsOutputMesh::getNbVAttributes()                    ///< number of vertices attributes
{
    return impl->getNbVAttributes();
}

unsigned int SofaPhysicsOutputMesh::getNbAttributes(int index)            ///< number of attributes in specified vertex attribute
{
    return impl->getNbAttributes(index);
}

const char*  SofaPhysicsOutputMesh::getVAttributeName(int index)          ///< vertices attribute name
{
    return impl->getVAttributeName(index);
}
int          SofaPhysicsOutputMesh::getVAttributeSizePerVertex(int index) ///< vertices attribute #
{
    return impl->getVAttributeSizePerVertex(index);
}
const Real*  SofaPhysicsOutputMesh::getVAttributeValue(int index)         ///< vertices attribute (Vec#)
{
    return impl->getVAttributeValue(index);
}
int          SofaPhysicsOutputMesh::getVAttributeRevision(int index)      ///< changes each time vertices attribute is updated
{
    return impl->getVAttributeRevision(index);
}

unsigned int SofaPhysicsOutputMesh::getNbLines() ///< number of lines
{
    return impl->getNbLines();
}
const Index* SofaPhysicsOutputMesh::getLines()   ///< lines topology (2 indices / line)
{
    return impl->getLines();
}
int SofaPhysicsOutputMesh::getLinesRevision()    ///< changes each time lines data is updated
{
    return impl->getLinesRevision();
}

unsigned int SofaPhysicsOutputMesh::getNbTriangles() ///< number of triangles
{
    return impl->getNbTriangles();
}
const Index* SofaPhysicsOutputMesh::getTriangles()   ///< triangles topology (3 indices / triangle)
{
    return impl->getTriangles();
}
int SofaPhysicsOutputMesh::getTrianglesRevision()    ///< changes each time triangles data is updated
{
    return impl->getTrianglesRevision();
}

unsigned int SofaPhysicsOutputMesh::getNbQuads() ///< number of quads
{
    return impl->getNbQuads();
}
const Index* SofaPhysicsOutputMesh::getQuads()   ///< quads topology (4 indices / quad)
{
    return impl->getQuads();
}
int SofaPhysicsOutputMesh::getQuadsRevision()    ///< changes each time quads data is updated
{
    return impl->getQuadsRevision();
}

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;


SofaPhysicsOutputMesh::Impl::Impl()
	: sObj(NULL)
{
}

SofaPhysicsOutputMesh::Impl::~Impl()
{
}

void SofaPhysicsOutputMesh::Impl::setObject(SofaOutputMesh* o)
{
	if (!sObj)
		return;

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

const char* SofaPhysicsOutputMesh::Impl::getName() ///< (non-unique) name of this object
{
    if (!sObj) return "";
    return sObj->getName().c_str();
}

ID SofaPhysicsOutputMesh::Impl::getID() ///< unique ID of this object
{
    return sObj.get();
}

unsigned int SofaPhysicsOutputMesh::Impl::getNbVertices() ///< number of vertices
{
    if (!sObj) return 0;
    // we cannot use getVertices() method directly as we need the Data revision
    Data<ResizableExtVector<Coord> > * data =
        (!sObj->m_vertPosIdx.getValue().empty()) ?
        &(sObj->m_vertices2) : &(sObj->m_positions);
    return (unsigned int) data->getValue().size();
}
const Real* SofaPhysicsOutputMesh::Impl::getVPositions()  ///< vertices positions (Vec3)
{
    Data<ResizableExtVector<Coord> > * data =
        (!sObj->m_vertPosIdx.getValue().empty()) ?
        &(sObj->m_vertices2) : &(sObj->m_positions);
    return (const Real*) data->getValue().getData();
}
const Real* SofaPhysicsOutputMesh::Impl::getVNormals()    ///< vertices normals   (Vec3)
{
    Data<ResizableExtVector<Deriv> > * data = &(sObj->m_vnormals);
    return (const Real*) data->getValue().getData();
}

const Real* SofaPhysicsOutputMesh::Impl::getVTexCoords()  ///< vertices UVs       (Vec2)
{
    Data<ResizableExtVector<TexCoord> > * data = &(sObj->m_vtexcoords);
    return (const Real*) data->getValue().getData();
}

int SofaPhysicsOutputMesh::Impl::getTexCoordRevision()    ///< changes each time tex coord data are updated
{
    Data<ResizableExtVector<TexCoord> > * data = &(sObj->m_vtexcoords);
    data->getValue(); // make sure the data is updated
    return data->getCounter();
}

int SofaPhysicsOutputMesh::Impl::getVerticesRevision()    ///< changes each time vertices data are updated
{
    Data<ResizableExtVector<Coord> > * data =
        (!sObj->m_vertPosIdx.getValue().empty()) ?
        &(sObj->m_vertices2) : &(sObj->m_positions);
    data->getValue(); // make sure the data is updated
    return data->getCounter();
}


unsigned int SofaPhysicsOutputMesh::Impl::getNbVAttributes()                    ///< number of vertices attributes
{
    return sVA.size();
}

unsigned int SofaPhysicsOutputMesh::Impl::getNbAttributes(int index)            ///< number of attributes in specified vertex attribute
{
  if ((unsigned)index >= sVA.size())
    return 0;
  else 
    return dynamic_cast< Data<ResizableExtVector<Real> >* >(sVA[index]->getSEValue())->getValue().size();
}

const char*  SofaPhysicsOutputMesh::Impl::getVAttributeName(int index)          ///< vertices attribute name
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

int          SofaPhysicsOutputMesh::Impl::getVAttributeSizePerVertex(int index) ///< vertices attribute #
{
    if ((unsigned)index >= sVA.size())
        return 0;
    else
        return sVA[index]->getSESizePerVertex();
}

const Real*  SofaPhysicsOutputMesh::Impl::getVAttributeValue(int index)         ///< vertices attribute (Vec#)
{
    if ((unsigned)index >= sVA.size())
        return NULL;
    else
        return (const Real*)((ResizableExtVector<Real>*)sVA[index]->getSEValue()->getValueVoidPtr())->getData();
}

int          SofaPhysicsOutputMesh::Impl::getVAttributeRevision(int index)      ///< changes each time vertices attribute is updated
{
    if ((unsigned)index >= sVA.size())
        return 0;
    else
    {
        sVA[index]->getSEValue()->getValueVoidPtr(); // make sure the data is updated
        return sVA[index]->getSEValue()->getCounter();
    }
}


unsigned int SofaPhysicsOutputMesh::Impl::getNbLines() ///< number of lines
{
    return 0; // not yet supported
}
const Index* SofaPhysicsOutputMesh::Impl::getLines()   ///< lines topology (2 indices / line)
{
    return NULL;
}
int SofaPhysicsOutputMesh::Impl::getLinesRevision()    ///< changes each time lines data is updated
{
    return 0;
}

unsigned int SofaPhysicsOutputMesh::Impl::getNbTriangles() ///< number of triangles
{
    Data<ResizableExtVector<Triangle> > * data = &(sObj->m_triangles);
    return (unsigned int) data->getValue().size();
}
const Index* SofaPhysicsOutputMesh::Impl::getTriangles()   ///< triangles topology (3 indices / triangle)
{
    Data<ResizableExtVector<Triangle> > * data = &(sObj->m_triangles);
    return (const Index*) data->getValue().getData();
}
int SofaPhysicsOutputMesh::Impl::getTrianglesRevision()    ///< changes each time triangles data is updated
{
    Data<ResizableExtVector<Triangle> > * data = &(sObj->m_triangles);
    data->getValue(); // make sure the data is updated
    return data->getCounter();
}

unsigned int SofaPhysicsOutputMesh::Impl::getNbQuads() ///< number of quads
{
    Data<ResizableExtVector<Quad> > * data = &(sObj->m_quads);
    return (unsigned int) data->getValue().size();
}
const Index* SofaPhysicsOutputMesh::Impl::getQuads()   ///< quads topology (4 indices / quad)
{
    Data<ResizableExtVector<Quad> > * data = &(sObj->m_quads);
    return (const Index*) data->getValue().getData();
}
int SofaPhysicsOutputMesh::Impl::getQuadsRevision()    ///< changes each time quads data is updated
{
    Data<ResizableExtVector<Quad> > * data = &(sObj->m_quads);
    data->getValue(); // make sure the data is updated
    return data->getCounter();
}
