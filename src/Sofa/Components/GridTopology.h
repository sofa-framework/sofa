#ifndef SOFA_COMPONENTS_GRIDTOPOLOGY_H
#define SOFA_COMPONENTS_GRIDTOPOLOGY_H

#include "MeshTopology.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class GridTopology : public MeshTopology
{
public:
    GridTopology();

    GridTopology(int nx, int ny, int nz);

    void setSize(int nx, int ny, int nz);

    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }

    //virtual int getNbPoints() const { return nx*ny*nz; }

    virtual int getNbCubes() { return (nx-1)*(ny-1)*(nz-1); }

    virtual int getNbQuads()
    {
        if (nz == 1)
            return (nx-1)*(ny-1);
        else if (ny == 1)
            return (nx-1)*(nz-1);
        else
            return (ny-1)*(nz-1);
    }

    virtual Cube getCube(int i) const;
    virtual Cube getCube(int x, int y, int z) const;

    virtual Quad getQuad(int i) const;
    virtual Quad getQuad(int x, int y, int z) const;

    int point(int x, int y, int z) const { return x+nx*(y+ny*z); }
    int cube(int x, int y, int z) const { return x+(nx-1)*(y+(ny-1)*z); }

protected:
    int nx;
    int ny;
    int nz;

    void updateLines();
    void updateQuads();
    void updateCubes();
};

} // namespace Components

} // namespace Sofa

#endif
