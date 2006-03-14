#include <iostream>
#include "MeshTopology.h"
#include "MeshTopologyLoader.h"
#include "XML/TopologyNode.h"
#include "Common/fixed_array.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

void create(MeshTopology*& obj, XML::Node<Core::Topology>* arg)
{
    obj = new MeshTopology();
    if (arg->getAttribute("filename"))
        obj->load(arg->getAttribute("filename"));
}

SOFA_DECL_CLASS(MeshTopology)

Creator<XML::TopologyNode::Factory, MeshTopology> MeshTopologyClass("Mesh");

MeshTopology::MeshTopology()
    : nbPoints(0), validLines(false), validTriangles(false), validQuads(false), validTetras(false), validCubes(false)
{
}


class MeshTopology::Loader : public MeshTopologyLoader
{
public:
    MeshTopology* dest;
    Loader(MeshTopology* dest) : dest(dest) {}
    virtual void addLine(int p1, int p2)
    {
        dest->seqLines.push_back(make_array(p1,p2));
    }
    virtual void addTriangle(int p1, int p2, int p3)
    {
        dest->seqTriangles.push_back(make_array(p1,p2,p3));
    }
    virtual void addQuad(int p1, int p2, int p3, int p4)
    {
        dest->seqQuads.push_back(make_array(p1,p2,p3,p4));
    }
    virtual void addTetra(int p1, int p2, int p3, int p4)
    {
        dest->seqTetras.push_back(make_array(p1,p2,p3,p4));
    }
    virtual void addCube(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8)
    {
        dest->seqCubes.push_back(make_array(p1,p2,p3,p4,p5,p6,p7,p8));
    }
};

void MeshTopology::clear()
{
    nbPoints = 0;
    seqLines.clear();
    seqTriangles.clear();
    seqQuads.clear();
    seqTetras.clear();
    seqCubes.clear();
    invalidate();
}

bool MeshTopology::load(const char* filename)
{
    clear();
    Loader loader(this);
    return loader.load(filename);
}

const MeshTopology::SeqLines& MeshTopology::getLines()
{
    if (!validLines)
    {
        updateLines();
        validLines = true;
    }
    return seqLines;
}

const MeshTopology::SeqTriangles& MeshTopology::getTriangles()
{
    if (!validTriangles)
    {
        updateTriangles();
        validTriangles = true;
    }
    return seqTriangles;
}

const MeshTopology::SeqQuads& MeshTopology::getQuads()
{
    if (!validQuads)
    {
        updateQuads();
        validQuads = true;
    }
    return seqQuads;
}

const MeshTopology::SeqTetras& MeshTopology::getTetras()
{
    if (!validTetras)
    {
        updateTetras();
        validTetras = true;
    }
    return seqTetras;
}

const MeshTopology::SeqCubes& MeshTopology::getCubes()
{
    if (!validCubes)
    {
        updateCubes();
        validCubes = true;
    }
    return seqCubes;
}

int MeshTopology::getNbPoints()
{
    return nbPoints;
}

int MeshTopology::getNbLines()
{
    return getLines().size();
}

int MeshTopology::getNbTriangles()
{
    return getTriangles().size();
}

int MeshTopology::getNbQuads()
{
    return getQuads().size();
}

int MeshTopology::getNbTetras()
{
    return getTetras().size();
}

int MeshTopology::getNbCubes()
{
    return getCubes().size();
}

const MeshTopology::Line& MeshTopology::getLine(index_type i)
{
    return getLines()[i];
}

const MeshTopology::Triangle& MeshTopology::getTriangle(index_type i)
{
    return getTriangles()[i];
}

const MeshTopology::Quad& MeshTopology::getQuad(index_type i)
{
    return getQuads()[i];
}

const MeshTopology::Tetra& MeshTopology::getTetra(index_type i)
{
    return getTetras()[i];
}

const MeshTopology::Cube& MeshTopology::getCube(index_type i)
{
    return getCubes()[i];
}

void MeshTopology::invalidate()
{
    validLines = false;
    validTriangles = false;
    validQuads = false;
    validTetras = false;
    validQuads = false;
}

} // namespace Components

} // namespace Sofa
