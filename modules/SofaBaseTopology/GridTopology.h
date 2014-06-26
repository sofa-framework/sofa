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
#ifndef SOFA_COMPONENT_TOPOLOGY_GRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_GRIDTOPOLOGY_H

#include <SofaBaseTopology/MeshTopology.h>
#include <sofa/core/DataEngine.h>
namespace sofa
{

namespace component
{

namespace topology
{

/** Define a regular grid topology, with no spatial information.
  */
class SOFA_BASE_TOPOLOGY_API GridTopology : public MeshTopology
{
public:
    SOFA_CLASS(GridTopology,MeshTopology);
    typedef sofa::defaulttype::Vec3i Vec3i;
    typedef sofa::defaulttype::Vector2 Vector2;
    typedef sofa::defaulttype::ResizableExtVector<Vector2> TextCoords2D;
    friend class GridUpdate;
private:
    class GridUpdate : public sofa::core::DataEngine
    {
    public:
        typedef MeshTopology::Edge Edge;
        typedef MeshTopology::Quad Quad;
        typedef MeshTopology::Hexa Hexa;
        SOFA_CLASS(GridUpdate,sofa::core::DataEngine);
        GridUpdate(GridTopology* t);
        void update();
    protected:
        void updateEdges();
        void updateQuads();
        void updateHexas();
    private:
        GridTopology* topology;
    };
protected:
    GridTopology();
    GridTopology(int nx, int ny, int nz);
    GridTopology(Vec3i nXnYnZ );
public:
    virtual void init();

    /// BaseObject method should be overwritten by children
    virtual void reinit(){}

    /// Set grid resolution, given the number of vertices
    void setSize(int nx, int ny, int nz);
    /// set grid resolution, given the number of vertices
    void setNumVertices( Vec3i nXnYnZ );
    /// Set grid resolution, given the number of vertices
    void setNumVertices(int nx, int ny, int nz);

    void parse(core::objectmodel::BaseObjectDescription* arg)
    {
        this->MeshTopology::parse(arg);

        if (arg->getAttribute("nx")!=NULL && arg->getAttribute("ny")!=NULL && arg->getAttribute("nz")!=NULL )
        {
            const char* nx = arg->getAttribute("nx");
            const char* ny = arg->getAttribute("ny");
            const char* nz = arg->getAttribute("nz");
            n.setValue(Vec3i(atoi(nx),atoi(ny),atoi(nz)));
        }

        this->setSize();
    }

    int getNx() const { return n.getValue()[0]; }
    int getNy() const { return n.getValue()[1]; }
    int getNz() const { return n.getValue()[2]; }

    void setNx(int n_) { (*n.beginEdit())[0] = n_; setSize(); }
    void setNy(int n_) { (*n.beginEdit())[1] = n_; setSize(); }
    void setNz(int n_) { (*n.beginEdit())[2] = n_; setSize(); }

    //int getNbPoints() const { return n.getValue()[0]*n.getValue()[1]*n.getValue()[2]; }

    virtual int getNbHexahedra() { return (n.getValue()[0]-1)*(n.getValue()[1]-1)*(n.getValue()[2]-1); }

    /*
    int getNbQuads() {
    if (n.getValue()[2] == 1)
    return (n.getValue()[0]-1)*(n.getValue()[1]-1);
    else if (n.getValue()[1] == 1)
    return (n.getValue()[0]-1)*(n.getValue()[2]-1);
    else
    return (n.getValue()[1]-1)*(n.getValue()[2]-1);
    }
    */

    Hexa getHexaCopy(int i);
    Hexa getHexahedron(int x, int y, int z);

#ifndef SOFA_NEW_HEXA
    Cube getCubeCopy(int i) { return getHexaCopy(i); }
    Cube getCube(int x, int y, int z) { return getHexahedron(x,y,z); }
#endif

    Quad getQuadCopy(int i);
    Quad getQuad(int x, int y, int z);

    int point(int x, int y, int z) const { return x+n.getValue()[0]*(y+n.getValue()[1]*z); }
    int hexa(int x, int y, int z) const { return x+(n.getValue()[0]-1)*(y+(n.getValue()[1]-1)*z); }
    int cube(int x, int y, int z) const { return hexa(x,y,z); }

    // Method to create grid texture coordinates, should be overwritten by children
    virtual void createTexCoords(){}

protected:
    Data< Vec3i > n;
    Data <bool> p_createTexCoords;
    Data <TextCoords2D> m_texCoords;

    virtual void setSize();
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
