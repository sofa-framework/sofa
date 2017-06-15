/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaVolumetricData/DistanceGrid.h>
#include <sofa/core/visual/VisualParams.h>
#include <fstream>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/template.h>

#include <sofa/helper/io/Mesh.h>

#ifdef SOFA_HAVE_MINIFLOWVR
#include <flowvr/render/mesh.h>
#endif

#include <fstream>
#include <sstream>

#include <sofa/helper/logging/Messaging.h>

#define FMM_VERBOSE false

namespace sofa
{

using helper::rabs;
using helper::rmax;

namespace component
{

namespace container
{

namespace _distancegrid_
{

using namespace defaulttype;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const Coord calcCellWidth(const int nx, const int ny,const int nz,
                          const Coord& pmin, const Coord& pmax)
{
   return Coord((pmax[0]-pmin[0])/(nx-1), (pmax[1]-pmin[1])/(ny-1),(pmax[2]-pmin[2])/(nz-1)) ;
}

const Coord calcInvCellWidth(const int nx, const int ny,const int nz,
                             const Coord& pmin, const Coord& pmax)
{
    return Coord((nx-1)/(pmax[0]-pmin[0]), (ny-1)/(pmax[1]-pmin[1]),(nz-1)/(pmax[2]-pmin[2])) ;
}

int validateDim(int n)
{
    if(n<0){
        dmsg_warning("DistanceGrid") << "Invalid dimension" << n << ". Dimension for a distance grid must be of size >= 0. Use '0' instead. " ;
        n=0;
    }
    return n;
}

DistanceGrid::DistanceGrid(int nx, int ny, int nz, Coord pmin, Coord pmax)
    : meshPts(new DefaultAllocator<Coord>)
    , m_nbRef(1)
    , m_nx(validateDim(nx)), m_ny(validateDim(ny)), m_nz(validateDim(nz))
    , m_nxny(m_nx*m_ny), m_nxnynz(m_nx*m_ny*m_nz)
    , m_dists(m_nx*m_ny*m_nz, new DefaultAllocator<SReal>)
    , m_pmin(pmin), m_pmax(pmax)
    , m_cellWidth   (calcCellWidth(m_nx,m_ny,m_nz,pmin,pmax))
    , m_invCellWidth(calcInvCellWidth(m_nx,m_ny,m_nz,pmin,pmax))
    , m_cubeDim(0)
{
}

DistanceGrid::DistanceGrid(int nx, int ny, int nz,
                           Coord pmin, Coord pmax, ExtVectorAllocator<SReal>* alloc)
    : meshPts(new DefaultAllocator<Coord>)
    , m_nbRef(1)
    , m_nx(validateDim(nx)), m_ny(validateDim(ny)), m_nz(validateDim(nz))
    , m_nxny(m_nx*m_ny), m_nxnynz(m_nx*m_ny*m_nz)
    , m_dists(m_nx*m_ny*m_nz, alloc)
    , m_pmin(pmin), m_pmax(pmax)
    , m_cellWidth   (calcCellWidth(m_nx,m_ny,m_nz,pmin,pmax))
    , m_invCellWidth(calcInvCellWidth(m_nx,m_ny,m_nz,pmin,pmax))
    , m_cubeDim(0)
{
}

DistanceGrid::~DistanceGrid()
{
    std::map<DistanceGridParams, DistanceGrid*>& shared = getShared();
    std::map<DistanceGridParams, DistanceGrid*>::iterator it = shared.begin();
    while (it != shared.end() && it->second != this) ++it;
    if (it != shared.end())
        shared.erase(it); // remove this grid from the list of already loaded grids
}

/// Add one reference to this grid. Note that loadShared already does this.
DistanceGrid* DistanceGrid::addRef()
{
    ++m_nbRef;
    return this;
}

/// Release one reference, deleting this grid if this is the last
bool DistanceGrid::release()
{
    if (--m_nbRef != 0)
        return false;
    delete this;
    return true;
}

//todo(dmarchal) we should make a loader for that...
DistanceGrid* DistanceGrid::load(const std::string& filename,
                                 double scale, double sampling,
                                 int nx, int ny, int nz, Coord pmin, Coord pmax)
{
    double absscale=fabs(scale);
    if (filename == "#cube")
    {
        float dim = (float)scale;
        int np = 5;
        Coord bbmin(-dim, -dim, -dim), bbmax(dim,dim,dim);
        if (pmin[0]<=pmax[0])
        {
            pmin = bbmin;
            pmax = bbmax;
            Coord margin = (bbmax-bbmin)*0.1;
            pmin -= margin;
            pmax += margin;
        }
        else
        {
            for (int c=0; c<3; c++)
            {
                if (bbmin[c] < pmin[c]) pmin[c] = bbmin[c];
                if (bbmax[c] > pmax[c]) pmax[c] = bbmax[c];
            }
        }
        DistanceGrid* grid = new DistanceGrid(nx, ny, nz, pmin, pmax);
        grid->calcCubeDistance(dim, np);
        if (sampling)
            grid->sampleSurface(sampling);
        return grid;
    }
    else if (filename.length()>4 && filename.substr(filename.length()-4) == ".raw")
    {
        DistanceGrid* grid = new DistanceGrid(nx, ny, nz, pmin, pmax);
        std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
        in.read((char*)&(grid->m_dists[0]), grid->m_nxnynz*sizeof(SReal));
        if (scale != 1.0)
        {
            for (int i=0; i< grid->m_nxnynz; i++)
                grid->m_dists[i] *= (float)scale;
        }
        grid->computeBBox();
        if (sampling)
            grid->sampleSurface(sampling);
        return grid;
    }
    else if (filename.length()>4 && filename.substr(filename.length()-4) == ".vtk")
    {
        return loadVTKFile(filename, scale, sampling);
    }
    else if (filename.length()>6 && filename.substr(filename.length()-6) == ".fmesh")
    {
#ifdef SOFA_HAVE_MINIFLOWVR
        flowvr::render::Mesh mesh;
        if (!mesh.load(filename.c_str()))
        {
            msg_error("DistanceGrid")<<"loading FlowVR mesh file "<<filename;
            return NULL;
        }

        if (!mesh.getAttrib(flowvr::render::Mesh::MESH_DISTMAP))
        {
            msg_error("DistanceGrid")<<"FlowVR mesh "<<filename<<" does not contain distance information. Please use flowvr-distmap.";
            return NULL;
        }
        nx = mesh.distmap->nx;
        ny = mesh.distmap->ny;
        nz = mesh.distmap->nz;
        ftl::Vec3f fpmin = ftl::transform(mesh.distmap->mat,ftl::Vec3f(0,0,0))*(float)absscale;
        ftl::Vec3f fpmax = ftl::transform(mesh.distmap->mat,ftl::Vec3f((float)(nx-1),(float)(ny-1),(float)(nz-1)))*(float)absscale;
        pmin = Coord(fpmin.ptr());
        pmax = Coord(fpmax.ptr());
        DistanceGrid* grid = new DistanceGrid(nx, ny, nz, pmin, pmax);
        for (int i=0; i< grid->m_nxnynz; i++)
            grid->m_dists[i] = mesh.distmap->data[i]*scale;
        if (sampling)
            grid->sampleSurface(sampling);
        else if (mesh.getAttrib(flowvr::render::Mesh::MESH_POINTS_GROUP))
        {
            int nbpos = 0;
            for (int i=0; i<mesh.nbg(); i++)
            {
                if (mesh.getGP0(i) >= 0)
                    ++nbpos;
            }
            grid->meshPts.resize(nbpos);
            int p = 0;
            for (int i=0; i<mesh.nbg(); i++)
            {
                int p0 = mesh.getGP0(i);
                if (p0 >= 0)
                    grid->meshPts[p++] = Coord(mesh.getPP(p0).ptr())*absscale;
            }
        }
        else
        {
            int nbpos = mesh.nbp();
            grid->meshPts.resize(nbpos);
            for (int i=0; i<nbpos; i++)
                grid->meshPts[i] = Coord(mesh.getPP(i).ptr())*absscale;
        }
        if (scale < 0)
        {
            grid->m_bbmin = grid->m_pmin;
            grid->m_bbmax = grid->m_pmax;
        }
        else
            grid->computeBBox();
        return grid;
#else
        msg_error("DistanceGrid")<<"Loading a .fmesh file requires the FlowVR library (activatable with the CMake option 'SOFA_BUILD_MINIFLOWVR')";
        return NULL;
#endif
    }
    else if (filename.length()>4 && filename.substr(filename.length()-4) == ".obj")
    {
        Mesh* mesh = Mesh::Create(filename);
        const helper::vector<Vector3> & vertices = mesh->getVertices();

        Coord bbmin, bbmax;
        if (!vertices.empty())
        {
            bbmin = vertices[0];
            bbmax = bbmin;
            for(unsigned int i=1; i<vertices.size(); i++)
            {
                for (int c=0; c<3; c++)
                    if (vertices[i][c] < bbmin[c]) bbmin[c] = (SReal)vertices[i][c];
                    else if (vertices[i][c] > bbmax[c]) bbmax[c] = (SReal)vertices[i][c];
            }
            bbmin *= absscale;
            bbmax *= absscale;
        }

        if (pmin[0]<=pmax[0])
        {
            pmin = bbmin;
            pmax = bbmax;
            Coord margin = (bbmax-bbmin)*0.1;
            pmin -= margin;
            pmax += margin;
        }
        else if (!vertices.empty())
        {
            for (int c=0; c<3; c++)
            {
                if (bbmin[c] < pmin[c]) pmin[c] = bbmin[c];
                if (bbmax[c] > pmax[c]) pmax[c] = bbmax[c];
            }
        }
        DistanceGrid* grid = new DistanceGrid(nx, ny, nz, pmin, pmax);
        grid->calcDistance(mesh, scale);
        if (sampling)
            grid->sampleSurface(sampling);
        else
        {
            grid->meshPts.resize(vertices.size());
            for(unsigned int i=0; i<vertices.size(); i++)
                grid->meshPts[i] = vertices[i]*absscale;
        }
        grid->computeBBox();
        delete mesh;
        return grid;
    }
    else
    {
         msg_error("DistanceGrid")<< "Unknown extension: "<<filename;
        return NULL;
    }
}

bool DistanceGrid::save(const std::string& filename)
{
    /// !!!TODO!!! ///
    if (filename.length()>4 && filename.substr(filename.length()-4) == ".raw")
    {
        std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
        out.write((char*)&(m_dists[0]), m_nxnynz*sizeof(SReal));
    }
    else
    {
        msg_error("DistanceGrid")<<" save(): Unsupported extension: "<<filename;
        return false;
    }
    return true;
}


template<class T> bool readData(std::istream& in, int dataSize, bool binary, DistanceGrid::VecSReal& data, double scale)
{
    if (binary)
    {
        T* buffer = new T[dataSize];
        in.read((char*)buffer, dataSize * sizeof(T));
        if (in.eof() || in.bad())
        {
            delete[] buffer;
            return false;
        }
        else
        {
            for (int i=0; i<dataSize; ++i)
                data[i] = (SReal)(buffer[i]*scale);
        }
        delete[] buffer;
        return true;
    }
    else
    {
        int i = 0;
        std::string line;
        T buffer;
        while(i < dataSize && !in.eof() && !in.bad())
        {
            std::getline(in, line);
            std::istringstream ln(line);
            while (i < dataSize && ln >> buffer)
            {
                data[i] = (SReal)(buffer*scale);
                ++i;
            }
        }
        return (i == dataSize);
    }
}

DistanceGrid* DistanceGrid::loadVTKFile(const std::string& filename, double scale, double sampling)
{
    // Format doc: http://www.vtk.org/pdf/file-formats.pdf
    // http://www.cacr.caltech.edu/~slombey/asci/vtk/vtk_formats.simple.html

    std::ifstream inVTKFile(filename.c_str(), std::ifstream::in & std::ifstream::binary);
    if( !inVTKFile.is_open() )
    {
        return NULL;
    }
    std::string line;

    // Part 1
    std::getline(inVTKFile, line);
    if (std::string(line,0,23) != "# vtk DataFile Version ") return NULL;
    std::string version(line,23);

    // Part 2
    std::string header;
    std::getline(inVTKFile, header);

    // Part 3
    std::getline(inVTKFile, line);

    bool binary;
    if (line == "BINARY") binary = true;
    else if (line == "ASCII") binary = false;
    else return NULL;

    // Part 4
    std::getline(inVTKFile, line);
    if (line != "DATASET STRUCTURED_POINTS")
    {
        return NULL;
    }

    msg_info("DistanceGrid")<< (binary ? "Binary" : "Text") << " VTK File " << filename << " (version " << version << "): " << header;
    int dataSize = 0;
    int nx = 0, ny = 0, nz = 0;
    Coord origin, spacing(1.0f,1.0f,1.0f);
    while(!inVTKFile.eof())
    {
        std::getline(inVTKFile, line);
        std::istringstream ln(line);
        std::string kw;
        ln >> kw;
        if (kw == "DIMENSIONS")
        {
            ln >> nx >> ny >> nz;
        }
        else if (kw == "SPACING")
        {
            ln >> spacing[0] >> spacing[1] >> spacing[2];
            spacing *= scale;
        }
        else if (kw == "ORIGIN")
        {
            ln >> origin[0] >> origin[1] >> origin[2];
            origin *= scale;
        }
        else if (kw == "CELL_DATA")
        {
            //section = CellData;
            ln >> dataSize;
        }
        else if (kw == "POINT_DATA")
        {
            //section = PointData;
            ln >> dataSize;
        }
        else if (kw == "SCALARS")
        {
            std::string name, typestr;
            ln >> name >> typestr;
            msg_info("DistanceGrid")<< "Found " << typestr << " data: " << name;
            std::getline(inVTKFile, line); // lookup_table, ignore
            msg_info("DistanceGrid")<< "Loading " << nx<<"x"<<ny<<"x"<<nz << " volume...";
            DistanceGrid* grid = new DistanceGrid(nx, ny, nz, origin, origin + Coord(spacing[0] * nx, spacing[1] * ny, spacing[2]*nz));
            bool ok = true;
            if (typestr == "char") ok = readData<char>(inVTKFile, dataSize, binary, grid->m_dists, scale);
            else if (typestr == "unsigned_char") ok = readData<unsigned char>(inVTKFile, dataSize, binary, grid->m_dists, scale);
            else if (typestr == "short") ok = readData<short>(inVTKFile, dataSize, binary, grid->m_dists, scale);
            else if (typestr == "unsigned_short") ok = readData<unsigned short>(inVTKFile, dataSize, binary, grid->m_dists, scale);
            else if (typestr == "int") ok = readData<int>(inVTKFile, dataSize, binary, grid->m_dists, scale);
            else if (typestr == "unsigned_int") ok = readData<unsigned int>(inVTKFile, dataSize, binary, grid->m_dists, scale);
            else if (typestr == "long") ok = readData<long long>(inVTKFile, dataSize, binary, grid->m_dists, scale);
            else if (typestr == "unsigned_long") ok = readData<unsigned long long>(inVTKFile, dataSize, binary, grid->m_dists, scale);
            else if (typestr == "float") ok = readData<float>(inVTKFile, dataSize, binary, grid->m_dists, scale);
            else if (typestr == "double") ok = readData<double>(inVTKFile, dataSize, binary, grid->m_dists, scale);
            else
            {
                msg_error("DistanceGrid")<< "Invalid type " << typestr;
                ok = false;
            }
            if (!ok)
            {
                delete grid;
                return NULL;
            }
            msg_info("DistanceGrid")<< "Volume data loading OK.";
            grid->computeBBox();
            if (sampling)
                grid->sampleSurface(sampling);
            return grid; // we read one scalar field, stop here.
        }
    }
    return NULL;
}

template<class T>
void * readData(std::istream& in, int dataSize, bool binary)
{
    T* buffer = new T[dataSize];
    if (binary)
    {
        in.read((char*)buffer, dataSize * sizeof(T));
        if (in.eof() || in.bad())
        {
            delete[] buffer;
            return NULL;
        }
    }
    else
    {
        int i = 0;
        std::string line;
        while(i < dataSize && !in.eof() && !in.bad())
        {
            std::getline(in, line);
            std::istringstream ln(line);
            while (i < dataSize && ln >> buffer[i])
                ++i;
        }
        if (i < dataSize)
        {
            delete[] buffer;
            return NULL;
        }
    }
    return buffer;
}


template<int U, int V>
bool pointInTriangle(const Coord& p, const Coord& p0, const Coord& p1, const Coord& p2)
{
    SReal u0 = p [U] - p0[U], v0 = p [V] - p0[V];
    SReal u1 = p1[U] - p0[U], v1 = p1[V] - p0[V];
    SReal u2 = p2[U] - p0[U], v2 = p2[V] - p0[V];
    SReal alpha, beta;
    if (u1 == 0)
    {
        beta = u0/u2;
        if ( beta < 0 || beta > 1 ) return false;
        alpha = (v0 - beta*v2)/v1;
        if ( alpha < 0 || (alpha+beta) > 1 ) return false;
    }
    else
    {
        beta = (v0*u1 - u0*v1)/(v2*u1 - u2*v1);
        if ( beta < 0 || beta > 1 ) return false;
        alpha = (u0 - beta*u2)/u1;
        if ( alpha < 0 || (alpha+beta) > 1 ) return false;
    }
    return true;
}

int DistanceGrid::index(const Coord& p, Coord& coefs) const
{
    coefs[0] = (p[0]-m_pmin[0])*m_invCellWidth[0];
    coefs[1] = (p[1]-m_pmin[1])*m_invCellWidth[1];
    coefs[2] = (p[2]-m_pmin[2])*m_invCellWidth[2];
    int x = helper::rfloor(coefs[0]);
    if (x<0) x=0; else if (x>=m_nx-1) x=m_nx-2;
    coefs[0] -= x;
    int y = helper::rfloor(coefs[1]);
    if (y<0) y=0; else if (y>=m_ny-1) y=m_ny-2;
    coefs[1] -= y;
    int z = helper::rfloor(coefs[2]);
    if (z<0) z=0; else if (z>=m_nz-1) z=m_nz-2;
    coefs[2] -= z;
    return x+m_nx*(y+m_ny*(z));
}



void DistanceGrid::computeBBox()
{
    if (!meshPts.empty())
    {
        m_bbmin = meshPts[0];
        m_bbmax = m_bbmin;
        for(unsigned int i=1; i<meshPts.size(); i++)
        {
            for (int c=0; c<3; c++)
                if (meshPts[i][c] < m_bbmin[c]) m_bbmin[c] = (SReal)meshPts[i][c];
                else if (meshPts[i][c] > m_bbmax[c]) m_bbmax[c] = (SReal)meshPts[i][c];
        }
    }
    else
    {
        m_bbmin = m_pmin;
        m_bbmax = m_pmax;
        /// \todo compute the SReal bbox from the grid content
    }
}


/// Compute distance field for a cube of the given half-size.
/// Also create a mesh of points using np points per axis
void DistanceGrid::calcCubeDistance(SReal dim, int np)
{
    m_cubeDim = dim;
    if (np > 1)
    {
        int nbp = np*np*np - (np-2)*(np-2)*(np-2);
        meshPts.resize(nbp);

        for (int i=0,z=0; z<np; z++)
            for (int y=0; y<np; y++)
                for (int x=0; x<np; x++)
                    if (z==0 || z==np-1 || y==0 || y==np-1 || x==0 || x==np-1)
                        meshPts[i++] = Coord(x*dim*2/(np-1) - dim, y*dim*2/(np-1) - dim, z*dim*2/(np-1) - dim);
    }

    SReal dim2 = dim; //*0.75f; // add some 'roundness' to the cubes corner

    for (int i=0,z=0; z<m_nz; z++)
        for (int y=0; y<m_ny; y++)
            for (int x=0; x<m_nx; x++,i++)
            {
                Coord p = coord(x,y,z);
                Coord s = p;
                bool out = false;
                for (int c=0; c<3; c++)
                {
                    if (s[c] < -dim2) { s[c] = -dim2; out = true; }
                    else if (s[c] >  dim2) { s[c] =  dim2; out = true; }
                }
                SReal d;
                if (out)
                    d = (p - s).norm();
                else
                    d = rmax(rmax(rabs(s[0]),rabs(s[1])),rabs(s[2])) - dim2;
                m_dists[i] = d - (dim-dim2);
            }
    m_bbmin = Coord(-dim,-dim,-dim);
    m_bbmax = Coord( dim, dim, dim);
}

/// Compute distance field from given mesh
void DistanceGrid::calcDistance(sofa::helper::io::Mesh* mesh, double scale)
{
    m_fmm_status.resize(m_nxnynz);
    m_fmm_heap.resize(m_nxnynz);
    m_fmm_heap_size = 0;
    dmsg_info("DistanceGrid")<< "FMM: Init.";

    std::fill(m_fmm_status.begin(), m_fmm_status.end(), FMM_FAR);
    std::fill(m_dists.begin(), m_dists.end(), maxDist());

    const helper::vector<Vector3> & vertices = mesh->getVertices();
    const helper::vector<helper::vector<helper::vector<int> > > & facets = mesh->getFacets();

    // Initialize distance of edges crossing triangles
    dmsg_info("DistanceGrid")<< "FMM: Initialize distance of edges crossing triangles.";

    for (unsigned int i=0; i<facets.size(); i++)
    {
        const helper::vector<int>& pts = facets[i][0];
        const int pt0 = 0;
        const Coord p0 = vertices[pts[pt0]]*scale;
        for (unsigned int pt2=2; pt2<pts.size(); pt2++)
        {
            const int pt1 = pt2-1;
            const Coord p1 = vertices[pts[pt1]]*scale;
            const Coord p2 = vertices[pts[pt2]]*scale;
            Coord bbmin = p0, bbmax = p0;
            for (int c=0; c<3; c++)
                if (p1[c] < bbmin[c]) bbmin[c] = p1[c];
                else if (p1[c] > bbmax[c]) bbmax[c] = p1[c];
            for (int c=0; c<3; c++)
                if (p2[c] < bbmin[c]) bbmin[c] = p2[c];
                else if (p2[c] > bbmax[c]) bbmax[c] = p2[c];

            Coord normal = (p1-p0).cross(p2-p0);
            normal.normalize();
            SReal d = -(p0*normal);
            int nedges = 0;
            int ix0 = ix(bbmin)-1; if (ix0 < 0) ix0 = 0;
            int iy0 = iy(bbmin)-1; if (iy0 < 0) iy0 = 0;
            int iz0 = iz(bbmin)-1; if (iz0 < 0) iz0 = 0;
            int ix1 = ix(bbmax)+2; if (ix1 >= m_nx) ix1 = m_nx-1;
            int iy1 = iy(bbmax)+2; if (iy1 >= m_ny) iy1 = m_ny-1;
            int iz1 = iz(bbmax)+2; if (iz1 >= m_nz) iz1 = m_nz-1;
            for (int z=iz0; z<iz1; z++)
                for (int y=iy0; y<iy1; y++)
                    for (int x=ix0; x<ix1; x++)
                    {
                        Coord pos = coord(x,y,z);
                        int ind = index(x,y,z);
                        SReal dist = pos*normal + d;
                        //if (rabs(dist) > cellWidth) continue; // no edge from this point can cross the plane

                        // X edge
                        if (rabs(normal[0]) > 1e-6)
                        {
                            SReal dist1 = -dist / normal[0];
                            int ind2 = ind+1;
                            if (dist1 >= -0.01*m_cellWidth[0] && dist1 <= 1.01*m_cellWidth[0])
                            {
                                // edge crossed plane
                                if (pointInTriangle<1,2>(pos,p0,p1,p2))
                                {
                                    // edge crossed triangle
                                    ++nedges;
                                    SReal dist2 = m_cellWidth[0] - dist1;
                                    if (normal[0]<0)
                                    {
                                        // p1 is in outside, p2 inside
                                        if (dist1 < (m_dists[ind]))
                                        {
                                            // nearest triangle
                                            m_dists[ind] = dist1;
                                            m_fmm_status[ind] = FMM_KNOWN_OUT;
                                        }
                                        if (dist2 < (m_dists[ind2]))
                                        {
                                            // nearest triangle
                                            m_dists[ind2] = dist2;
                                            m_fmm_status[ind2] = FMM_KNOWN_IN;
                                        }
                                    }
                                    else
                                    {
                                        // p1 is in inside, p2 outside
                                        if (dist1 < (m_dists[ind]))
                                        {
                                            // nearest triangle
                                            m_dists[ind] = dist1;
                                            m_fmm_status[ind] = FMM_KNOWN_IN;
                                        }
                                        if (dist2 < (m_dists[ind2]))
                                        {
                                            // nearest triangle
                                            m_dists[ind2] = dist2;
                                            m_fmm_status[ind2] = FMM_KNOWN_OUT;
                                        }
                                    }
                                }
                            }
                        }

                        // Y edge
                        if (rabs(normal[1]) > 1e-6)
                        {
                            SReal dist1 = -dist / normal[1];
                            int ind2 = ind+m_nx;
                            if (dist1 >= -0.01*m_cellWidth[1] && dist1 <= 1.01*m_cellWidth[1])
                            {
                                // edge crossed plane
                                if (pointInTriangle<2,0>(pos,p0,p1,p2))
                                {
                                    // edge crossed triangle
                                    ++nedges;
                                    SReal dist2 = m_cellWidth[1] - dist1;
                                    if (normal[1]<0)
                                    {
                                        // p1 is in outside, p2 inside
                                        if (dist1 < (m_dists[ind]))
                                        {
                                            // nearest triangle
                                            m_dists[ind] = dist1;
                                            m_fmm_status[ind] = FMM_KNOWN_OUT;
                                        }
                                        if (dist2 < (m_dists[ind2]))
                                        {
                                            // nearest triangle
                                            m_dists[ind2] = dist2;
                                            m_fmm_status[ind2] = FMM_KNOWN_IN;
                                        }
                                    }
                                    else
                                    {
                                        // p1 is in inside, p2 outside
                                        if (dist1 < (m_dists[ind]))
                                        {
                                            // nearest triangle
                                            m_dists[ind] = dist1;
                                            m_fmm_status[ind] = FMM_KNOWN_IN;
                                        }
                                        if (dist2 < (m_dists[ind2]))
                                        {
                                            // nearest triangle
                                            m_dists[ind2] = dist2;
                                            m_fmm_status[ind2] = FMM_KNOWN_OUT;
                                        }
                                    }
                                }
                            }
                        }

                        // Z edge
                        if (rabs(normal[2]) > 1e-6)
                        {
                            SReal dist1 = -dist / normal[2];
                            int ind2 = ind+m_nxny;
                            if (dist1 >= -0.01*m_cellWidth[2] && dist1 <= 1.01*m_cellWidth[2])
                            {
                                // edge crossed plane
                                if (pointInTriangle<0,1>(pos,p0,p1,p2))
                                {
                                    // edge crossed triangle
                                    ++nedges;
                                    SReal dist2 = m_cellWidth[2] - dist1;
                                    if (normal[2]<0)
                                    {
                                        // p1 is in outside, p2 inside
                                        if (dist1 < (m_dists[ind]))
                                        {
                                            // nearest triangle
                                            m_dists[ind] = dist1;
                                            m_fmm_status[ind] = FMM_KNOWN_OUT;
                                        }
                                        if (dist2 < (m_dists[ind2]))
                                        {
                                            // nearest triangle
                                            m_dists[ind2] = dist2;
                                            m_fmm_status[ind2] = FMM_KNOWN_IN;
                                        }
                                    }
                                    else
                                    {
                                        // p1 is in inside, p2 outside
                                        if (dist1 < (m_dists[ind]))
                                        {
                                            // nearest triangle
                                            m_dists[ind] = dist1;
                                            m_fmm_status[ind] = FMM_KNOWN_IN;
                                        }
                                        if (dist2 < (m_dists[ind2]))
                                        {
                                            // nearest triangle
                                            m_dists[ind2] = dist2;
                                            m_fmm_status[ind2] = FMM_KNOWN_OUT;
                                        }
                                    }
                                }
                            }
                        }
                    }
         }
    }

    // Update known points neighbors
    for (int z=0, ind=0; z<m_nz; z++)
        for (int y=0; y<m_ny; y++)
            for (int x=0; x<m_nx; x++, ind++)
            {
                if (m_fmm_status[ind] < FMM_FAR)
                {
                    int ind2;
                    SReal dist1 = m_dists[ind];
                    SReal dist2 = dist1+m_cellWidth[0];
                    // X-1
                    if (x>0)
                    {
                        ind2 = ind-1;
                        if (x>0 && m_fmm_status[ind2] >= FMM_FAR && (m_dists[ind2]) > dist2)
                        {
                            m_dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                    // X+1
                    if (x<m_nx-1)
                    {
                        ind2 = ind+1;
                        if (x>0 && m_fmm_status[ind2] >= FMM_FAR && (m_dists[ind2]) > dist2)
                        {
                            m_dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                    dist2 = dist1+m_cellWidth[1];
                    // Y-1
                    if (y>0)
                    {
                        ind2 = ind-m_nx;
                        if (x>0 && m_fmm_status[ind2] >= FMM_FAR && (m_dists[ind2]) > dist2)
                        {
                            m_dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                    // Y+1
                    if (y<m_ny-1)
                    {
                        ind2 = ind+m_nx;
                        if (x>0 && m_fmm_status[ind2] >= FMM_FAR && (m_dists[ind2]) > dist2)
                        {
                            m_dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                    dist2 = dist1+m_cellWidth[2];
                    // Z-1
                    if (z>0)
                    {
                        ind2 = ind-m_nxny;
                        if (x>0 && m_fmm_status[ind2] >= FMM_FAR && (m_dists[ind2]) > dist2)
                        {
                            m_dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                    // Z+1
                    if (z<m_nz-1)
                    {
                        ind2 = ind+m_nxny;
                        if (x>0 && m_fmm_status[ind2] >= FMM_FAR && (m_dists[ind2]) > dist2)
                        {
                            m_dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                }
            }

    // March through the heap
    while (m_fmm_heap_size > 0)
    {
        int ind = fmm_pop();
        int nbin = 0, nbout = 0;
        int x = ind%m_nx;
        int y = (ind/m_nx)%m_ny;
        int z = ind/m_nxny;

        int ind2;
        SReal dist1 = m_dists[ind];
        SReal dist2 = dist1+m_cellWidth[0];
        // X-1
        if (x>0)
        {
            ind2 = ind-1;
            if (m_fmm_status[ind2] < FMM_FAR)
            {
                if (m_fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((m_dists[ind2]) > dist2)
            {
                m_dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        // X+1
        if (x<m_nx-1)
        {
            ind2 = ind+1;
            if (m_fmm_status[ind2] < FMM_FAR)
            {
                if (m_fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((m_dists[ind2]) > dist2)
            {
                m_dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        dist2 = dist1+m_cellWidth[1];
        // Y-1
        if (y>0)
        {
            ind2 = ind-m_nx;
            if (m_fmm_status[ind2] < FMM_FAR)
            {
                if (m_fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((m_dists[ind2]) > dist2)
            {
                m_dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        // Y+1
        if (y<m_ny-1)
        {
            ind2 = ind+m_nx;
            if (m_fmm_status[ind2] < FMM_FAR)
            {
                if (m_fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((m_dists[ind2]) > dist2)
            {
                m_dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        dist2 = dist1+m_cellWidth[2];
        // Z-1
        if (z>0)
        {
            ind2 = ind-m_nxny;
            if (m_fmm_status[ind2] < FMM_FAR)
            {
                if (m_fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((m_dists[ind2]) > dist2)
            {
                m_dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        // Z+1
        if (z<m_nz-1)
        {
            ind2 = ind+m_nxny;
            if (m_fmm_status[ind2] < FMM_FAR)
            {
                if (m_fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((m_dists[ind2]) > dist2)
            {
                m_dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        if (nbin && nbout)
        {
            msg_warning("DistanceGrid")<< "FMM WARNING: in/out conflict at cell "<<x<<" "<<y<<" "<<z<<" ( "<<nbin<<" in, "<<nbout<<" out), dist = "<<m_dists[ind];
        }
        if (nbin > nbout)
            m_fmm_status[ind] = FMM_KNOWN_IN;
        else
            m_fmm_status[ind] = FMM_KNOWN_OUT;
    }

    // Finalize distances
    int nbin = 0;
    for (int z=0, ind=0; z<m_nz; z++)
        for (int y=0; y<m_ny; y++)
            for (int x=0; x<m_nx; x++, ind++)
            {
                if (m_fmm_status[ind] == FMM_KNOWN_IN)
                {
                    m_dists[ind] = -m_dists[ind];
                    ++nbin;
                }
                else if (m_fmm_status[ind] != FMM_KNOWN_OUT)
                {
                }
            }
    msg_info("DistanceGrid")<< "FMM: DONE. "<< nbin << " points inside ( " << (nbin*100)/size() <<" % )";
}

inline void DistanceGrid::fmm_swap(int entry1, int entry2)
{
    int ind1 = m_fmm_heap[entry1];
    int ind2 = m_fmm_heap[entry2];
    m_fmm_heap[entry1] = ind2;
    m_fmm_heap[entry2] = ind1;
    m_fmm_status[ind1] = entry2 + FMM_FRONT0;
    m_fmm_status[ind2] = entry1 + FMM_FRONT0;
}

int DistanceGrid::fmm_pop()
{

    int res = m_fmm_heap[0];

    if(FMM_VERBOSE)
        msg_info("DistanceGrid")<< "fmm_pop -> <"<<(res%m_nx)<<','<<((res/m_nx)%m_ny)<<','<<(res/m_nxny)<<">="<<m_dists[res];

    --m_fmm_heap_size;
    if (m_fmm_heap_size>0)
    {
        fmm_swap(0, m_fmm_heap_size);
        int i=0;
        SReal phi = (m_dists[m_fmm_heap[i]]);
        while (i*2+1 < m_fmm_heap_size)
        {
            SReal phi1 = (m_dists[m_fmm_heap[i*2+1]]);
            if (i*2+2 < m_fmm_heap_size)
            {
                SReal phi2 = (m_dists[m_fmm_heap[i*2+2]]);
                if (phi1 < phi)
                {
                    if (phi1 < phi2)
                    {
                        fmm_swap(i, i*2+1);
                        i = i*2+1;
                    }
                    else
                    {
                        fmm_swap(i, i*2+2);
                        i = i*2+2;
                    }
                }
                else if (phi2 < phi)
                {
                    fmm_swap(i, i*2+2);
                    i = i*2+2;
                }
                else break;
            }
            else if (phi1 < phi)
            {
                fmm_swap(i, i*2+1);
                i = i*2+1;
            }
            else break;
        }
    }

    if(FMM_VERBOSE){
        std::stringstream tmp;
        tmp << "fmm_heap = [";
        for (int i=0; i<m_fmm_heap_size; i++)
            tmp << " <"<<(m_fmm_heap[i]%m_nx)<<','<<((m_fmm_heap[i]/m_nx)%m_ny)<<','<<(m_fmm_heap[i]/m_nxny)<<">="<<m_dists[m_fmm_heap[i]];
        msg_info("DistanceGrid") << tmp.str() ;
    }

    return res;
}

void DistanceGrid::fmm_push(int index)
{
    SReal phi = (m_dists[index]);
    int i;
    if (m_fmm_status[index] >= FMM_FRONT0)
    {
        i = m_fmm_status[index] - FMM_FRONT0;

        if(FMM_VERBOSE)
           dmsg_info("DistanceGrid") << "fmm update <"<<(index%m_nx)<<','<<((index/m_nx)%m_ny)<<','<<(index/m_nxny)<<">="<<m_dists[index]<<" from entry "<<i ;

        while (i>0 && phi < (m_dists[m_fmm_heap[(i-1)/2]]))
        {
            fmm_swap(i,(i-1)/2);
            i = (i-1)/2;
        }
        while (i*2+1 < m_fmm_heap_size)
        {
            SReal phi1 = (m_dists[m_fmm_heap[i*2+1]]);
            if (i*2+2 < m_fmm_heap_size)
            {
                SReal phi2 = (m_dists[m_fmm_heap[i*2+2]]);
                if (phi1 < phi)
                {
                    if (phi1 < phi2)
                    {
                        fmm_swap(i, i*2+1);
                        i = i*2+1;
                    }
                    else
                    {
                        fmm_swap(i, i*2+2);
                        i = i*2+2;
                    }
                }
                else if (phi2 < phi)
                {
                    fmm_swap(i, i*2+2);
                    i = i*2+2;
                }
                else break;
            }
            else if (phi1 < phi)
            {
                fmm_swap(i, i*2+1);
                i = i*2+1;
            }
            else break;
        }
    }
    else
    {
        if(FMM_VERBOSE)
           dmsg_info("DistanceGrid") << "fmm push <"<<(index%m_nx)<<','<<((index/m_nx)%m_ny)<<','<<(index/m_nxny)<<">="<<m_dists[index] ;

        i = m_fmm_heap_size;
        ++m_fmm_heap_size;
        m_fmm_heap[i] = index;
        m_fmm_status[index] = i;
        while (i>0 && phi < (m_dists[m_fmm_heap[(i-1)/2]]))
        {
            fmm_swap(i,(i-1)/2);
            i = (i-1)/2;
        }
    }

    if(FMM_VERBOSE){
        std::stringstream tmp;
        tmp << "fmm_heap = [";
        for (int i=0; i<m_fmm_heap_size; i++)
            tmp << " <"<<(m_fmm_heap[i]%m_nx)<<','<<((m_fmm_heap[i]/m_nx)%m_ny)<<','<<(m_fmm_heap[i]/m_nxny)<<">="<<m_dists[m_fmm_heap[i]];
        msg_info("DistanceGrid") << tmp.str() ;
    }
}

/// Sample the surface with points approximately separated by the given sampling distance (expressed in voxels if the value is negative)
void DistanceGrid::sampleSurface(double sampling)
{
    msg_info("DistanceGrid")<< "sample surface with sampling distance " << sampling;
    std::vector<Coord> pts;
    if (sampling <= -1.0 && sampling == floor(sampling))
    {
        int stepX, stepY, stepZ;
        stepX = stepY = stepZ = (int)(-sampling);
        msg_info("DistanceGrid")<< "sampling steps: " << stepX << " " << stepY << " " << stepZ;

        SReal maxD = (SReal)sqrt((m_cellWidth[0]*stepX)*(m_cellWidth[0]*stepX) + (m_cellWidth[1]*stepY)*(m_cellWidth[1]*stepY) + (m_cellWidth[2]*stepZ)*(m_cellWidth[2]*stepZ));
        for (int z=1; z<m_nz-1; z+=stepZ)
            for (int y=1; y<m_ny-1; y+=stepY)
                for (int x=1; x<m_nx-1; x+=stepX)
                {
                    SReal d = m_dists[index(x,y,z)];
                    if (rabs(d) > maxD) continue;

                    Vector3 pos = coord(x,y,z);
                    Vector3 n = grad(index(x,y,z), Coord()); // note that there are some redundant computations between interp() and grad()
                    n.normalize();
                    pos -= n * (d * 0.99); // push pos back to the surface
                    d = interp(pos);
                    int it = 1;
                    while (rabs(d) > 0.01f*maxD && it < 10)
                    {
                        n = grad(pos);
                        n.normalize();
                        pos -= n * (d * 0.99); // push pos back to the surface
                        d = interp(pos);
                        ++it;
                    }
                    if (it == 10 && rabs(d) > 0.1f*maxD)
                    {
                        msg_warning("DistanceGrid")
                                << "Failed to converge at ("<<x<<","<<y<<","<<z<<"):"
                                << " pos0 = " << coord(x,y,z) << " d0 = " << m_dists[index(x,y,z)] << " grad0 = " << grad(index(x,y,z), Coord())
                                << " pos = " << pos << " d = " << d << " grad = " << n;
                        continue;
                    }
                    Coord p = pos;
                    pts.push_back(p);
                }
    }
    else
    {
        if (sampling < 0) sampling = m_cellWidth[0] * (-sampling);
        SReal maxD = (SReal)(sqrt(3.0)*sampling);
        int nstepX = (int)ceil((m_pmax[0] - m_pmin[0])/sampling);
        int nstepY = (int)ceil((m_pmax[1] - m_pmin[1])/sampling);
        int nstepZ = (int)ceil((m_pmax[2] - m_pmin[2])/sampling);
        Coord p0 = m_pmin + ((m_pmax-m_pmin) - Coord((nstepX)*sampling, (nstepY)*sampling, (nstepZ)*sampling))*0.5f;
        msg_info("DistanceGrid")<< "sampling bbox " << m_pmin << " - " << m_pmax << " starting at " << p0 << " with number of steps: " << nstepX << " " << nstepY << " " << nstepZ;

        for (int z=0; z<=nstepZ; z++)
            for (int y=0; y<=nstepY; y++)
                for (int x=0; x<=nstepX; x++)
                {
                    Coord pos = p0 + Coord(x*sampling, y*sampling, z*sampling);
                    if (!inGrid(pos)) continue;
                    SReal d = interp(pos);
                    if (rabs(d) > maxD) continue;
                    Vector3 n = grad(pos);
                    n.normalize();
                    pos -= n * (d * 0.99); // push pos back to the surface
                    d = interp(pos);
                    int it = 1;
                    while (rabs(d) > 0.01f*maxD && it < 10)
                    {
                        n = grad(pos);
                        n.normalize();
                        pos -= n * (d * 0.99); // push pos back to the surface
                        d = interp(pos);
                        ++it;
                    }
                    if (it == 10 && rabs(d) > 0.1f*maxD)
                    {
                        msg_warning("DistanceGrid")<< "Failed to converge at ("<<x<<","<<y<<","<<z<<"):"
                                << " pos0 = " << coord(x,y,z) << " d0 = " << m_dists[index(x,y,z)] << " grad0 = " << grad(index(x,y,z), Coord())
                                << " pos = " << pos << " d = " << d << " grad = " << n;
                        continue;
                    }
                    Coord p = pos;
                    pts.push_back(p);
                }
    }
    msg_info("DistanceGrid")<< pts.size() << " sampling points created.";
    meshPts.resize(pts.size());
    for (unsigned int p=0; p<pts.size(); ++p)
        meshPts[p] = pts[p];
}


DistanceGrid* DistanceGrid::loadShared(const std::string& filename,
                                       double scale, double sampling, int nx, int ny, int nz, Coord pmin, Coord pmax)
{
    DistanceGridParams params;
    params.filename = filename;
    params.scale = scale;
    params.sampling = sampling;
    params.nx = nx;
    params.ny = ny;
    params.nz = nz;
    params.pmin = pmin;
    params.pmax = pmax;
    std::map<DistanceGridParams, DistanceGrid*>& shared = getShared();
    std::map<DistanceGridParams, DistanceGrid*>::iterator it = shared.find(params);
    if (it != shared.end())
        return it->second->addRef();
    else
    {
        return shared[params] = load(filename, scale, sampling, nx, ny, nz, pmin, pmax);
    }
}


SReal DistanceGrid::quickeval(const Coord& x) const
{
    SReal d;
    if (inGrid(x))
    {
        d = m_dists[index(x)] - m_cellWidth[0]; // we underestimate the distance
    }
    else
    {
        Coord xclamp = clamp(x);
        d = m_dists[index(xclamp)] - m_cellWidth[0]; // we underestimate the distance
        d = helper::rsqrt((x-xclamp).norm2() + d*d);
    }
    return d;
}

SReal DistanceGrid::eval2(const Coord& x) const
{
    SReal d2;
    if (inGrid(x))
    {
        SReal d = interp(x);
        d2 = d*d;
    }
    else
    {
        Coord xclamp = clamp(x);
        SReal d = interp(xclamp);
        d2 = ((x-xclamp).norm2() + d*d); // we underestimate the distance
    }
    return d2;
}

SReal DistanceGrid::quickeval2(const Coord& x) const
{
    SReal d2;
    if (inGrid(x))
    {
        SReal d = m_dists[index(x)] - m_cellWidth[0]; // we underestimate the distance
        d2 = d*d;
    }
    else
    {
        Coord xclamp = clamp(x);
        SReal d = m_dists[index(xclamp)] - m_cellWidth[0]; // we underestimate the distance
        d2 = ((x-xclamp).norm2() + d*d);
    }
    return d2;
}

SReal DistanceGrid::interp(int index, const Coord& coefs) const
{
    return interp(coefs[2],interp(coefs[1],interp(coefs[0],m_dists[index          ],m_dists[index+1        ]),
            interp(coefs[0],m_dists[index  +m_nx     ],m_dists[index+1+m_nx     ])),
            interp(coefs[1],interp(coefs[0],m_dists[index     +m_nxny],m_dists[index+1   +m_nxny]),
                    interp(coefs[0],m_dists[index  +m_nx+m_nxny],m_dists[index+1+m_nx+m_nxny])));
}


SReal DistanceGrid::interp(const Coord& p) const
{
    Coord coefs;
    int i = index(p, coefs);
    return interp(i, coefs);
}

Coord DistanceGrid::grad(int index, const Coord& coefs) const
{
    // val = dist[0][0][0] * (1-x) * (1-y) * (1-z)
    //     + dist[1][0][0] * (  x) * (1-y) * (1-z)
    //     + dist[0][1][0] * (1-x) * (  y) * (1-z)
    //     + dist[1][1][0] * (  x) * (  y) * (1-z)
    //     + dist[0][0][1] * (1-x) * (1-y) * (  z)
    //     + dist[1][0][1] * (  x) * (1-y) * (  z)
    //     + dist[0][1][1] * (1-x) * (  y) * (  z)
    //     + dist[1][1][1] * (  x) * (  y) * (  z)
    // dval / dx = (dist[1][0][0]-dist[0][0][0]) * (1-y) * (1-z)
    //           + (dist[1][1][0]-dist[0][1][0]) * (  y) * (1-z)
    //           + (dist[1][0][1]-dist[0][0][1]) * (1-y) * (  z)
    //           + (dist[1][1][1]-dist[0][1][1]) * (  y) * (  z)
    const SReal dist000 = m_dists[index          ];
    const SReal dist100 = m_dists[index+1        ];
    const SReal dist010 = m_dists[index  +m_nx     ];
    const SReal dist110 = m_dists[index+1+m_nx     ];
    const SReal dist001 = m_dists[index     +m_nxny];
    const SReal dist101 = m_dists[index+1   +m_nxny];
    const SReal dist011 = m_dists[index  +m_nx+m_nxny];
    const SReal dist111 = m_dists[index+1+m_nx+m_nxny];
    return Coord(
            interp(coefs[2],interp(coefs[1],dist100-dist000,dist110-dist010),interp(coefs[1],dist101-dist001,dist111-dist011)), //*invCellWidth[0],
            interp(coefs[2],interp(coefs[0],dist010-dist000,dist110-dist100),interp(coefs[0],dist011-dist001,dist111-dist101)), //*invCellWidth[1],
            interp(coefs[1],interp(coefs[0],dist001-dist000,dist101-dist100),interp(coefs[0],dist011-dist010,dist111-dist110))); //*invCellWidth[2]);
}

Coord DistanceGrid::grad(const Coord& p) const
{
    Coord coefs;
    int i = index(p, coefs);
    return grad(i, coefs);
}

SReal DistanceGrid::eval(const Coord& x) const
{
    SReal d;
    if (inGrid(x))
    {
        d = interp(x);
    }
    else
    {
        Coord xclamp = clamp(x);
        d = interp(xclamp);
        d = helper::rsqrt((x-xclamp).norm2() + d*d); // we underestimate the distance
    }
    return d;
}

bool DistanceGrid::DistanceGridParams::operator==(const DistanceGridParams& v) const
{
    if (!(filename == v.filename)) return false;
    if (!(scale    == v.scale   )) return false;
    if (!(sampling == v.sampling)) return false;
    if (!(nx       == v.nx      )) return false;
    if (!(ny       == v.ny      )) return false;
    if (!(nz       == v.nz      )) return false;
    if (!(pmin[0]  == v.pmin[0] )) return false;
    if (!(pmin[1]  == v.pmin[1] )) return false;
    if (!(pmin[2]  == v.pmin[2] )) return false;
    if (!(pmax[0]  == v.pmax[0] )) return false;
    if (!(pmax[1]  == v.pmax[1] )) return false;
    if (!(pmax[2]  == v.pmax[2] )) return false;
    return true;
}

bool DistanceGrid::DistanceGridParams::operator<(const DistanceGridParams& v) const
{
    if (filename < v.filename) return false;
    if (filename > v.filename) return true;
    if (scale    < v.scale   ) return false;
    if (scale    > v.scale   ) return true;
    if (sampling < v.sampling) return false;
    if (sampling > v.sampling) return true;
    if (nx       < v.nx      ) return false;
    if (nx       > v.nx      ) return true;
    if (ny       < v.ny      ) return false;
    if (ny       > v.ny      ) return true;
    if (nz       < v.nz      ) return false;
    if (nz       > v.nz      ) return true;
    if (pmin[0]  < v.pmin[0] ) return false;
    if (pmin[0]  > v.pmin[0] ) return true;
    if (pmin[1]  < v.pmin[1] ) return false;
    if (pmin[1]  > v.pmin[1] ) return true;
    if (pmin[2]  < v.pmin[2] ) return false;
    if (pmin[2]  > v.pmin[2] ) return true;
    if (pmax[0]  < v.pmax[0] ) return false;
    if (pmax[0]  > v.pmax[0] ) return true;
    if (pmax[1]  < v.pmax[1] ) return false;
    if (pmax[1]  > v.pmax[1] ) return true;
    if (pmax[2]  < v.pmax[2] ) return false;
    if (pmax[2]  > v.pmax[2] ) return true;
    return false;
}

bool DistanceGrid::DistanceGridParams::operator>(const DistanceGridParams& v) const
{
    if (filename > v.filename) return false;
    if (filename < v.filename) return true;
    if (scale    > v.scale   ) return false;
    if (scale    < v.scale   ) return true;
    if (sampling < v.sampling) return false;
    if (sampling > v.sampling) return true;
    if (nx       > v.nx      ) return false;
    if (nx       < v.nx      ) return true;
    if (ny       > v.ny      ) return false;
    if (ny       < v.ny      ) return true;
    if (nz       > v.nz      ) return false;
    if (nz       < v.nz      ) return true;
    if (pmin[0]  > v.pmin[0] ) return false;
    if (pmin[0]  < v.pmin[0] ) return true;
    if (pmin[1]  > v.pmin[1] ) return false;
    if (pmin[1]  < v.pmin[1] ) return true;
    if (pmin[2]  > v.pmin[2] ) return false;
    if (pmin[2]  < v.pmin[2] ) return true;
    if (pmax[0]  > v.pmax[0] ) return false;
    if (pmax[0]  < v.pmax[0] ) return true;
    if (pmax[1]  > v.pmax[1] ) return false;
    if (pmax[1]  < v.pmax[1] ) return true;
    if (pmax[2]  > v.pmax[2] ) return false;
    if (pmax[2]  < v.pmax[2] ) return true;
    return false;
}

std::map<DistanceGrid::DistanceGridParams, DistanceGrid*>& DistanceGrid::getShared()
{
    static std::map<DistanceGridParams, DistanceGrid*> instance;
    return instance;
}

} // namespace _distancegrid_

} // namespace container

} // namespace component

} // namespace sofa
