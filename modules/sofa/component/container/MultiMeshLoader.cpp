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
#include <sofa/component/container/MultiMeshLoader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/Topology.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace container
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MultiMeshLoader)

int MultiMeshLoaderClass = core::RegisterObject("Generic multiple Mesh Loader")
        .add< MultiMeshLoader >()
        ;

MultiMeshLoader::MultiMeshLoader()
    : filenameList(initData(&filenameList,"filenamelist","list of the filenames of the objects"))
    , nbPointsPerMesh(initData(&nbPointsPerMesh,"nbPointsPerMesh","the number of points per mesh"))
    , nbPointsTotal(initData(&nbPointsTotal,"nbPointsTotal","the total number of points loaded"))
{
    nbPointsPerMesh.setReadOnly(true);
    nbPointsTotal.setReadOnly(true);
}


void MultiMeshLoader::init()
{
    helper::WriteAccessor<Data<helper::vector<sofa::defaulttype::Vector3> > > _vertices(this->vertices);
    _vertices.resize(seqPoints.size());
    for (unsigned int i=0 ; i<seqPoints.size() ; i++)		 for (unsigned int j=0 ; j<3 ; j++) _vertices[i][j]=seqPoints[i][j];
}

void MultiMeshLoader::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->BaseObject::parse(arg);
    clear();
    for (unsigned int i=0 ; i<filenameList.getValue().size() ; i++)
    {
        if (filenameList.getValue()[i] != "")
        {
            pushMesh(filenameList.getValue()[i].c_str());
        }
    }
    nbPointsTotal.setValue(seqPoints.size());
}

void MultiMeshLoader::pushMesh(const char* filename)
{
    currentMeshIndex = seqPoints.size();
    load(filename);
    nbPointsPerMesh.beginEdit()->push_back(seqPoints.size() - currentMeshIndex);
    nbPointsPerMesh.endEdit();
}


bool MultiMeshLoader::load(const char* filename)
{
    //clear();
    if (!MeshTopologyLoader::load(filename))
    {
        serr << "Unable to load Mesh "<<filename << sendl;
        return false;
    }

    return true;
}

void MultiMeshLoader::addPoint(double px, double py, double pz)
{
    seqPoints.push_back(helper::make_array((SReal)px, (SReal)py, (SReal)pz));
}

void MultiMeshLoader::addLine( int a, int b )
{
    seqEdges.push_back(Edge(currentMeshIndex+a,currentMeshIndex+b));
}

void MultiMeshLoader::addTriangle( int a, int b, int c )
{
    seqTriangles.push_back( Triangle(currentMeshIndex+a,currentMeshIndex+b,currentMeshIndex+c) );
}

void MultiMeshLoader::addTetra( int a, int b, int c, int d )
{
    seqTetrahedra.push_back( Tetra(currentMeshIndex+a,currentMeshIndex+b,currentMeshIndex+c,currentMeshIndex+d) );
}

void MultiMeshLoader::addQuad(int p1, int p2, int p3, int p4)
{
    if (triangulate.getValue())
    {
        addTriangle(p1,p2,p3);
        addTriangle(p1,p3,p4);
    }
    else
        seqQuads.push_back(Quad(currentMeshIndex+p1,currentMeshIndex+p2,currentMeshIndex+p3,currentMeshIndex+p4));
}

void MultiMeshLoader::addCube(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8)
{
#ifdef SOFA_NEW_HEXA
    seqHexahedra.push_back(Hexa(currentMeshIndex+p1,currentMeshIndex+p2,currentMeshIndex+p3,currentMeshIndex+p4,currentMeshIndex+p5,currentMeshIndex+p6,currentMeshIndex+p7,currentMeshIndex+p8));
#else
    seqHexahedra.push_back(Hexa(currentMeshIndex+p1,currentMeshIndex+p2,currentMeshIndex+p4,currentMeshIndex+p3,currentMeshIndex+p5,currentMeshIndex+p6,currentMeshIndex+p8,currentMeshIndex+p7));
#endif
}

}

} // namespace component

} // namespace sofa

