#include <iostream>
#include "Common/Mesh.h"
#include "MeshTopology.h"
#include "MeshTopologyLoader.h"
#include "Common/ObjectFactory.h"
#include "Common/fixed_array.h"

#include <set>

namespace Sofa
{

namespace Components
{

using namespace Common;

void create(MeshTopology*& obj, ObjectDescription* arg)
{
    obj = new MeshTopology();
    if (arg->getAttribute("filename"))
        obj->load(arg->getAttribute("filename"));
}

SOFA_DECL_CLASS(MeshTopology)

Creator<ObjectFactory, MeshTopology> MeshTopologyClass("Mesh");

MeshTopology::MeshTopology()
    : nbPoints(0), validLines(false), validTriangles(false), validQuads(false), validTetras(false), validCubes(false)
{
}


class MeshTopology::Loader : public MeshTopologyLoader
{
public:
    MeshTopology* dest;
    Loader(MeshTopology* dest) : dest(dest) {}
    virtual void addPoint(double px, double py, double pz)
    {
        dest->seqPoints.push_back(make_array(px, py, pz));
        if (dest->seqPoints.size() > (unsigned)dest->nbPoints)
            dest->nbPoints = dest->seqPoints.size();
    }
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

    if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".obj"))
    {
        Mesh* mesh = Mesh::Create(filename);
        if (mesh==NULL) return false;

        loader.setNbPoints(mesh->getVertices().size());
        for (unsigned int i=0; i<mesh->getVertices().size(); i++)
        {
            loader.addPoint(mesh->getVertices()[i][0],mesh->getVertices()[i][1],mesh->getVertices()[i][2]);
        }

        std::set<std::pair<int,int>> edges;

        const std::vector< std::vector < std::vector <int> > > & facets = mesh->getFacets();
        for (unsigned int i=0; i<facets.size(); i++)
        {
            const std::vector<int>& facet = facets[i][0];
            if (facet.size()==4)
            {
                // Quat
                loader.addQuad(facet[0],facet[1],facet[2],facet[3]);
            }
            else
            {
                // Triangularize
                for (unsigned int j=2; j<facet.size(); j++)
                    loader.addTriangle(facet[0],facet[j-1],facet[j]);
            }
            // Add edges
            for (unsigned int j=0; j<facet.size(); j++)
            {
                int i1 = facet[j];
                int i2 = facet[(j+1)%facet.size()];
                if (edges.count(std::make_pair(i1,i2))!=0)
                {
                    std::cerr << "ERROR: Duplicate edge.\n";
                }
                else if (edges.count(std::make_pair(i2,i1))==0)
                {
                    loader.addLine(i1,i2);
                    edges.insert(std::make_pair(i1,i2));
                }
            }
        }
        delete mesh;
    }
    else
    {
        if (!loader.load(filename))
            return false;
    }
    return true;
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

bool MeshTopology::hasPos()
{
    return !seqPoints.empty();
}

double MeshTopology::getPX(int i)
{
    return ((unsigned)i<seqPoints.size()?seqPoints[i][0]:0.0);
}

double MeshTopology::getPY(int i)
{
    return ((unsigned)i<seqPoints.size()?seqPoints[i][1]:0.0);
}

double MeshTopology::getPZ(int i)
{
    return ((unsigned)i<seqPoints.size()?seqPoints[i][2]:0.0);
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
