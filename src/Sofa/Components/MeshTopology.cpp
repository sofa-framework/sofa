#include <iostream>
#include "MeshTopology.h"
#include "Common/fixed_array.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

MeshTopology::MeshTopology()
    : nbPoints(0), validLines(false), validTriangles(false), validQuads(false), validTetras(false), validCubes(false)
{
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
