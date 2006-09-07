#include "GridTopology.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(GridTopology)

void create(GridTopology*& obj, ObjectDescription* arg)
{
    const char* nx = arg->getAttribute("nx");
    const char* ny = arg->getAttribute("ny");
    const char* nz = arg->getAttribute("nz");
    if (!nx || !ny || !nz)
    {
        std::cerr << "GridTopology requires nx, ny and nz attributes\n";
    }
    else
    {
        obj = new GridTopology(atoi(nx),atoi(ny),atoi(nz));
    }
}

Creator<ObjectFactory, GridTopology> GridTopologyClass("Grid");

GridTopology::GridTopology()
    : nx(0), ny(0), nz(0)
{
}

GridTopology::GridTopology(int nx, int ny, int nz)
    : nx(nx), ny(ny), nz(nz)
{
    nbPoints = nx*ny*nz;
}

void GridTopology::setSize(int nx, int ny, int nz)
{
    if (nx == this->nx && ny == this->ny && nz == this->nz)
        return;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    this->nbPoints = nx*ny*nz;
    invalidate();
}

void GridTopology::updateLines()
{
    seqLines.clear();
    seqLines.reserve((nx-1)*ny*nz+nx*(ny-1)*nz+nx*ny*(nz-1));
    // lines along X
    for (int z=0; z<nz; z++)
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx-1; x++)
                seqLines.push_back(make_array(point(x,y,z),point(x+1,y,z)));
    // lines along Y
    for (int z=0; z<nz; z++)
        for (int y=0; y<ny-1; y++)
            for (int x=0; x<nx; x++)
                seqLines.push_back(make_array(point(x,y,z),point(x,y+1,z)));
    // lines along Z
    for (int z=0; z<nz-1; z++)
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx; x++)
                seqLines.push_back(make_array(point(x,y,z),point(x,y,z+1)));
}

void GridTopology::updateQuads()
{
    seqQuads.clear();
    seqQuads.reserve((nx-1)*(ny-1)*nz+(nx-1)*ny*(nz-1)+nx*(ny-1)*(nz-1));
    // quads along XY plane
    for (int z=0; z<nz; z++)
        for (int y=0; y<ny-1; y++)
            for (int x=0; x<nx-1; x++)
                seqQuads.push_back(make_array(point(x,y,z),point(x+1,y,z),point(x+1,y+1,z),point(x,y+1,z)));
    // quads along XZ plane
    for (int z=0; z<nz-1; z++)
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx-1; x++)
                seqQuads.push_back(make_array(point(x,y,z),point(x+1,y,z),point(x+1,y,z+1),point(x,y,z+1)));
    // quads along YZ plane
    for (int z=0; z<nz-1; z++)
        for (int y=0; y<ny-1; y++)
            for (int x=0; x<nx; x++)
                seqQuads.push_back(make_array(point(x,y,z),point(x,y+1,z),point(x,y+1,z+1),point(x,y,z+1)));
}

void GridTopology::updateCubes()
{
    seqCubes.clear();
    seqCubes.reserve((nx-1)*(ny-1)*(nz-1));
    for (int z=0; z<nz-1; z++)
        for (int y=0; y<ny-1; y++)
            for (int x=0; x<nx-1; x++)
                seqCubes.push_back(make_array(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
                        point(x  ,y+1,z  ),point(x+1,y+1,z  ),
                        point(x  ,y  ,z+1),point(x+1,y  ,z+1),
                        point(x  ,y+1,z+1),point(x+1,y+1,z+1)));
}

GridTopology::Cube GridTopology::getCube(int i) const
{
    int x = i%(nx-1); i/=(nx-1);
    int y = i%(ny-1); i/=(ny-1);
    int z = i;
    return getCube(x,y,z);
}

GridTopology::Cube GridTopology::getCube(int x, int y, int z) const
{
    return make_array(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
            point(x  ,y+1,z  ),point(x+1,y+1,z  ),
            point(x  ,y  ,z+1),point(x+1,y  ,z+1),
            point(x  ,y+1,z+1),point(x+1,y+1,z+1));
}



GridTopology::Quad GridTopology::getQuad(int i) const
{
    if (nx == 1)
    {
        int y = i%(ny-1);
        i/=(ny-1);
        int z = i%(nz-1);

        return getQuad(1,y,z);
    }
    else if (ny == 1)
    {
        int x = i%(nx-1);
        i/=(nx-1);
        int z = i%(nz-1);

        return getQuad(x,1,z);
    }
    else
    {
        int x = i%(nx-1);
        i/=(nx-1);
        int y = i%(ny-1);

        return getQuad(x,y,1);
    }
}

GridTopology::Quad GridTopology::getQuad(int x, int y, int /*z*/) const
{
    /*
    	if (x == -1)
    		return make_array(point(1, y, z),point(1, y+1, z),
    			point(1, y+1, z+1),point(1, y, z+1));

    	else if (y == -1)
    		return make_array(point(x, 1, z),point(x+1, 1, z),
    			point(x+1, 1, z+1),point(x, 1, z+1));

    	else
    		return make_array(point(x, y, 1),point(x+1, y, 1),
    			point(x+1, y+1, 1),point(x, y+1, 1));
    */
    return make_array(point(x, y, 1),point(x+1, y, 1),
            point(x+1, y+1, 1),point(x, y+1, 1));
}

} // namespace Components

} // namespace Sofa
