#ifndef SOFA_COMPONENT_TOPOLOGY_MESHTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_MESHTOPOLOGY_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sofa/core/componentmodel/topology/Topology.h>
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

class MeshTopology : public core::componentmodel::topology::Topology
{
public:

    typedef int index_type;

    typedef Vec<2,index_type> Line;
    typedef Vec<3,index_type> Triangle;
    typedef Vec<4,index_type> Quad;
    typedef Vec<4,index_type> Tetra;
    typedef Vec<8,index_type> Cube;

    typedef vector<Line> SeqLines;
    typedef vector<Triangle> SeqTriangles;
    typedef vector<Quad> SeqQuads;
    typedef vector<Tetra> SeqTetras;
    typedef vector<Cube> SeqCubes;

    MeshTopology();
    //virtual const char* getTypeName() const { return "Mesh"; }

    virtual void clear();

    virtual bool load(const char* filename);

    int getNbPoints() const;

    // Complete sequence accessors

    const SeqLines& getLines();
    const SeqTriangles& getTriangles();
    const SeqQuads& getQuads();
    const SeqTetras& getTetras();
    const SeqCubes& getCubes();

    // Random accessors

    int getNbLines();
    int getNbTriangles();
    int getNbQuads();
    int getNbTetras();
    int getNbCubes();

    const Line& getLine(index_type i);
    const Triangle& getTriangle(index_type i);
    const Quad& getQuad(index_type i);
    const Tetra& getTetra(index_type i);
    const Cube& getCube(index_type i);

    // Points accessors (not always available)

    virtual bool hasPos() const;
    virtual double getPX(int i) const;
    virtual double getPY(int i) const;
    virtual double getPZ(int i) const;

    // for procedural creation without file loader
    void addPoint(double px, double py, double pz);
    void addLine( int a, int b );
    void addTriangle( int a, int b, int c );
    void addTetrahedron( int a, int b, int c, int d );

    // get the current revision of this mesh (use to detect changes)
    int getRevision() const { return revision; }
protected:
    int nbPoints;
    vector< fixed_array<double,3> > seqPoints;

    DataField<SeqLines> seqLines;
    bool validLines;

    //SeqTriangles   seqTriangles;
    DataField<SeqTriangles> seqTriangles;
    bool         validTriangles;
    SeqQuads       seqQuads;
    bool         validQuads;

    SeqTetras      seqTetras;
    bool         validTetras;
    SeqCubes       seqCubes;
    bool         validCubes;

    int revision;
    void invalidate();

    virtual void updateLines()     { }
    virtual void updateTriangles() { }
    virtual void updateQuads()     { }
    virtual void updateTetras()    { }
    virtual void updateCubes()     { }

    class Loader;
    friend class Loader;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
