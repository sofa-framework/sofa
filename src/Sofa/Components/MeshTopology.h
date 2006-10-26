#ifndef SOFA_COMPONENTS_MESHTOPOLOGY_H
#define SOFA_COMPONENTS_MESHTOPOLOGY_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include "Sofa/Core/Topology.h"
#include "Common/fixed_array.h"
#include "Common/vector.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class MeshTopology : public Core::Topology
{
public:
    typedef int index_type;

    typedef fixed_array<index_type, 2> Line;
    typedef fixed_array<index_type, 3> Triangle;
    typedef fixed_array<index_type, 4> Quad;
    typedef fixed_array<index_type, 4> Tetra;
    typedef fixed_array<index_type, 8> Cube;

    typedef vector<Line> SeqLines;
    typedef vector<Triangle> SeqTriangles;
    typedef vector<Quad> SeqQuads;
    typedef vector<Tetra> SeqTetras;
    typedef vector<Cube> SeqCubes;

    MeshTopology();
    virtual const char* getTypeName() const { return "Mesh"; }

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
    void addTriangle( int a, int b, int c );
    void addTetrahedron( int a, int b, int c, int d );

protected:
    int nbPoints;
    vector< fixed_array<double,3> > seqPoints;

    SeqLines seqLines;
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

    void invalidate();

    virtual void updateLines()     { }
    virtual void updateTriangles() { }
    virtual void updateQuads()     { }
    virtual void updateTetras()    { }
    virtual void updateCubes()     { }

    class Loader;
    friend class Loader;
};

} // namespace Components

} // namespace Sofa

#endif
