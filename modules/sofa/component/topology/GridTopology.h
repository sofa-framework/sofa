#ifndef SOFA_COMPONENT_TOPOLOGY_GRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_GRIDTOPOLOGY_H

#include <sofa/component/topology/MeshTopology.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

class GridTopology : public MeshTopology
{
public:
    GridTopology();

    GridTopology(int nx, int ny, int nz);

    void setSize(int nx, int ny, int nz);

    void parse(core::objectmodel::BaseObjectDescription* arg)
    {
        this->MeshTopology::parse(arg);
        this->setSize();
    }

    int getNx() const { return nx.getValue(); }
    int getNy() const { return ny.getValue(); }
    int getNz() const { return nz.getValue(); }

    void setNx(int n) { nx.setValue(n); setSize(); }
    void setNy(int n) { ny.setValue(n); setSize(); }
    void setNz(int n) { nz.setValue(n); setSize(); }

    //int getNbPoints() const { return nx.getValue()*ny.getValue()*nz.getValue(); }

    int getNbCubes() { return (nx.getValue()-1)*(ny.getValue()-1)*(nz.getValue()-1); }

    int getNbQuads()
    {
        if (nz.getValue() == 1)
            return (nx.getValue()-1)*(ny.getValue()-1);
        else if (ny.getValue() == 1)
            return (nx.getValue()-1)*(nz.getValue()-1);
        else
            return (ny.getValue()-1)*(nz.getValue()-1);
    }

    Cube getCube(int i);
    Cube getCube(int x, int y, int z);

    Quad getQuad(int i);
    Quad getQuad(int x, int y, int z);

    int point(int x, int y, int z) const { return x+nx.getValue()*(y+ny.getValue()*z); }
    int cube(int x, int y, int z) const { return x+(nx.getValue()-1)*(y+(ny.getValue()-1)*z); }

protected:
    DataField<int> nx;
    DataField<int> ny;
    DataField<int> nz;

    virtual void setSize();
    void updateLines();
    void updateQuads();
    void updateCubes();
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
