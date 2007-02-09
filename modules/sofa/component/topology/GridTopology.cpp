#include <sofa/component/topology/GridTopology.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(GridTopology)

int GridTopologyClass = core::RegisterObject("Base class fo a regular grid in 3D")
        .addAlias("Grid")
        .add< GridTopology >()
        ;

GridTopology::GridTopology()
    : nx(dataField(&nx,0,"nx","x grid resolution")), ny(dataField(&ny,0,"ny","y grid resolution")), nz(dataField(&nz,0,"nz","z grid resolution"))
{
}

GridTopology::GridTopology(int _nx, int _ny, int _nz)
    : nx(dataField(&nx,_nx,"nx","x grid resolution")), ny(dataField(&ny,_ny,"ny","y grid resolution")), nz(dataField(&nz,_nz,"nz","z grid resolution"))
{
    nbPoints = _nx*_ny*_nz;
}

void GridTopology::setSize(int nx, int ny, int nz)
{
    if (nx == this->nx.getValue() && ny == this->ny.getValue() && nz == this->nz.getValue())
        return;
    this->nx.setValue(nx);
    this->ny.setValue(ny);
    this->nz.setValue(nz);
    setSize();
}

void GridTopology::setSize()
{
    this->nbPoints = nx.getValue()*ny.getValue()*nz.getValue();
    invalidate();
}

void GridTopology::updateLines()
{
    SeqLines& lines = *seqLines.beginEdit();
    lines.clear();
    lines.reserve((nx.getValue()-1)*ny.getValue()*nz.getValue()+nx.getValue()*(ny.getValue()-1)*nz.getValue()+nx.getValue()*ny.getValue()*(nz.getValue()-1));
    // lines along X
    for (int z=0; z<nz.getValue(); z++)
        for (int y=0; y<ny.getValue(); y++)
            for (int x=0; x<nx.getValue()-1; x++)
                lines.push_back(Line(point(x,y,z),point(x+1,y,z)));
    // lines along Y
    for (int z=0; z<nz.getValue(); z++)
        for (int y=0; y<ny.getValue()-1; y++)
            for (int x=0; x<nx.getValue(); x++)
                lines.push_back(Line(point(x,y,z),point(x,y+1,z)));
    // lines along Z
    for (int z=0; z<nz.getValue()-1; z++)
        for (int y=0; y<ny.getValue(); y++)
            for (int x=0; x<nx.getValue(); x++)
                lines.push_back(Line(point(x,y,z),point(x,y,z+1)));
    seqLines.endEdit();
}

void GridTopology::updateQuads()
{
    seqQuads.clear();
    seqQuads.reserve((nx.getValue()-1)*(ny.getValue()-1)*nz.getValue()+(nx.getValue()-1)*ny.getValue()*(nz.getValue()-1)+nx.getValue()*(ny.getValue()-1)*(nz.getValue()-1));
    // quads along XY plane
    for (int z=0; z<nz.getValue(); z++)
        for (int y=0; y<ny.getValue()-1; y++)
            for (int x=0; x<nx.getValue()-1; x++)
                seqQuads.push_back(Quad(point(x,y,z),point(x+1,y,z),point(x+1,y+1,z),point(x,y+1,z)));
    // quads along XZ plane
    for (int z=0; z<nz.getValue()-1; z++)
        for (int y=0; y<ny.getValue(); y++)
            for (int x=0; x<nx.getValue()-1; x++)
                seqQuads.push_back(Quad(point(x,y,z),point(x+1,y,z),point(x+1,y,z+1),point(x,y,z+1)));
    // quads along YZ plane
    for (int z=0; z<nz.getValue()-1; z++)
        for (int y=0; y<ny.getValue()-1; y++)
            for (int x=0; x<nx.getValue(); x++)
                seqQuads.push_back(Quad(point(x,y,z),point(x,y+1,z),point(x,y+1,z+1),point(x,y,z+1)));
}

void GridTopology::updateCubes()
{
    seqCubes.clear();
    seqCubes.reserve((nx.getValue()-1)*(ny.getValue()-1)*(nz.getValue()-1));
    for (int z=0; z<nz.getValue()-1; z++)
        for (int y=0; y<ny.getValue()-1; y++)
            for (int x=0; x<nx.getValue()-1; x++)
                seqCubes.push_back(Cube(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
                        point(x  ,y+1,z  ),point(x+1,y+1,z  ),
                        point(x  ,y  ,z+1),point(x+1,y  ,z+1),
                        point(x  ,y+1,z+1),point(x+1,y+1,z+1)));
}

GridTopology::Cube GridTopology::getCube(int i)
{
    int x = i%(nx.getValue()-1); i/=(nx.getValue()-1);
    int y = i%(ny.getValue()-1); i/=(ny.getValue()-1);
    int z = i;
    return getCube(x,y,z);
}

GridTopology::Cube GridTopology::getCube(int x, int y, int z)
{
    return Cube(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
            point(x  ,y+1,z  ),point(x+1,y+1,z  ),
            point(x  ,y  ,z+1),point(x+1,y  ,z+1),
            point(x  ,y+1,z+1),point(x+1,y+1,z+1));
}

GridTopology::Quad GridTopology::getQuad(int i)
{
    if (nx.getValue() == 1)
    {
        int y = i%(ny.getValue()-1);
        i/=(ny.getValue()-1);
        int z = i%(nz.getValue()-1);

        return getQuad(1,y,z);
    }
    else if (ny.getValue() == 1)
    {
        int x = i%(nx.getValue()-1);
        i/=(nx.getValue()-1);
        int z = i%(nz.getValue()-1);

        return getQuad(x,1,z);
    }
    else
    {
        int x = i%(nx.getValue()-1);
        i/=(nx.getValue()-1);
        int y = i%(ny.getValue()-1);

        return getQuad(x,y,1);
    }
}

GridTopology::Quad GridTopology::getQuad(int x, int y, int /*z*/)
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
    return Quad(point(x, y, 1),point(x+1, y, 1),
            point(x+1, y+1, 1),point(x, y+1, 1));
}

} // namespace topology

} // namespace component

} // namespace sofa

