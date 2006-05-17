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

    int getNbPoints() const { return nx*ny*nz; }

    int getNbCubes() const { return (nx-1)*(ny-1)*(nz-1); }

    int getNbQuads() const
    {
        if (nz == 1)
            return (nx-1)*(ny-1);
        else if (ny == 1)
            return (nx-1)*(nz-1);
        else
            return (ny-1)*(nz-1);
    }

    Cube getCube(int i);
    Cube getCube(int x, int y, int z);

    Quad getQuad(int i);
    Quad getQuad(int x, int y, int z);

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
