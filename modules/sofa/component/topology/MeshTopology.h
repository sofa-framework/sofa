/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_MESHTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_MESHTOPOLOGY_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sofa/core/componentmodel/topology/Topology.h>
#include <sofa/component/topology/BaseMeshTopology.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using helper::vector;
using helper::fixed_array;

class MeshTopology : public BaseMeshTopology
{
public:
    MeshTopology();

    virtual void clear();

    virtual bool load(const char* filename);

    virtual int getNbPoints() const;

    // Complete sequence accessors

    virtual const SeqEdges& getEdges();
    virtual const SeqTriangles& getTriangles();
    virtual const SeqQuads& getQuads();
    virtual const SeqTetras& getTetras();
    virtual const SeqHexas& getHexas();

    // Random accessors

    virtual int getNbEdges();
    virtual int getNbTriangles();
    virtual int getNbQuads();
    virtual int getNbTetras();
    virtual int getNbHexas();

    virtual Edge getEdge(EdgeID i);
    virtual Triangle getTriangle(TriangleID i);
    virtual Quad getQuad(QuadID i);
    virtual Tetra getTetra(TetraID i);
    virtual Hexa getHexa(HexaID i);

    // Points accessors (not always available)

    virtual bool hasPos() const;
    virtual double getPX(int i) const;
    virtual double getPY(int i) const;
    virtual double getPZ(int i) const;
    virtual std::string getFilename() const {return filename.getValue();}

    // for procedural creation without file loader
    void addPoint(double px, double py, double pz);
    void addEdge( int a, int b );
    void addTriangle( int a, int b, int c );
    void addTetrahedron( int a, int b, int c, int d );

    // get the current revision of this mesh (use to detect changes)
    int getRevision() const { return revision; }

    /// return true if the given cube is active, i.e. it contains or is surrounded by mapped points.
    /// @deprecated
    virtual bool isCubeActive(int /*index*/) { return true; }

    void parse(core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("filename"))
        {
            filename.setValue( arg->getAttribute("filename") );
            this->load(arg->getAttribute("filename"));
        }
        arg->removeAttribute("filename");
        this->core::componentmodel::topology::Topology::parse(arg);
    }

protected:
    int nbPoints;
    vector< fixed_array<double,3> > seqPoints;

    Data<SeqEdges> seqEdges;
    bool validEdges;

    //SeqTriangles   seqTriangles;
    Data<SeqTriangles> seqTriangles;
    bool         validTriangles;
    SeqQuads       seqQuads;
    bool         validQuads;

    SeqTetras      seqTetras;
    bool         validTetras;
    SeqCubes       seqHexas;
    bool         validHexas;

    int revision;

    Data< std::string > filename;

    void invalidate();

    virtual void updateEdges()     { }
    virtual void updateTriangles() { }
    virtual void updateQuads()     { }
    virtual void updateTetras()    { }
    virtual void updateHexas()     { }

    class Loader;
    friend class Loader;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
