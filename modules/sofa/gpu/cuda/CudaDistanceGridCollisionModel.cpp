#ifdef SOFA_HAVE_GLEW
#include <GL/glew.h>
#endif
#include "CudaDistanceGridCollisionModel.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/collision/CubeModel.h>
#include <fstream>
#include <GL/gl.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaDistanceGridCollisionModel)

//int CudaDistanceGridCollisionModelClass = core::RegisterObject("GPU-based grid distance field using CUDA")
//.add< CudaDistanceGridCollisionModel >()
//.addAlias("CudaDistanceGrid")
//;

using namespace defaulttype;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

CudaDistanceGrid::CudaDistanceGrid(int nx, int ny, int nz, Coord pmin, Coord pmax)
    : nbRef(1), nx(nx), ny(ny), nz(nz), nxny(nx*ny), nxnynz(nx*ny*nz)
    , pmin(pmin), pmax(pmax)
    , cellWidth   ((pmax[0]-pmin[0])/(nx-1), (pmax[1]-pmin[1])/(ny-1),(pmax[2]-pmin[2])/(nz-1))
    , invCellWidth((nx-1)/(pmax[0]-pmin[0]), (ny-1)/(pmax[1]-pmin[1]),(nz-1)/(pmax[2]-pmin[2]))
    , cubeDim(0)
{
    dists.resize(nxnynz);
}

/// Add one reference to this grid. Note that loadShared already does this.
CudaDistanceGrid* CudaDistanceGrid::addRef()
{
    ++nbRef;
    return this;
}

/// Release one reference, deleting this grid if this is the last
bool CudaDistanceGrid::release()
{
    if (--nbRef != 0)
        return false;
    delete this;
    return true;
}

CudaDistanceGrid* CudaDistanceGrid::load(const std::string& filename, double scale, int nx, int ny, int nz, Coord pmin, Coord pmax)
{
    if (filename == "#cube")
    {
        float dim = (float)scale;
        int np = 5;
        Coord bbmin(-dim, -dim, -dim), bbmax(dim,dim,dim);
        std::cout << "bbox = <"<<bbmin<<">-<"<<bbmax<<">"<<std::endl;
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
        std::cout << "Creating cube distance grid in <"<<pmin<<">-<"<<pmax<<">"<<std::endl;
        CudaDistanceGrid* grid = new CudaDistanceGrid(nx, ny, nz, pmin, pmax);
        grid->calcCubeDistance(dim, np);
        std::cout << "Distance grid creation DONE."<<std::endl;
        return grid;
    }
    else if (filename.length()>4 && filename.substr(filename.length()-4) == ".raw")
    {
        CudaDistanceGrid* grid = new CudaDistanceGrid(nx, ny, nz, pmin, pmax);
        std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
        in.read((char*)&(grid->dists[0]), grid->nxnynz*sizeof(Real));
        if (scale != 1.0)
        {
            for (int i=0; i< grid->nxnynz; i++)
                grid->dists[i] *= (float)scale;
        }
        grid->computeBBox();
        return grid;
    }
#ifdef SOFA_HAVE_FLOWVR
    else if (filename.length()>6 && filename.substr(filename.length()-6) == ".fmesh")
    {
        flowvr::render::Mesh mesh;
        if (!mesh.load(filename.c_str()))
        {
            std::cerr << "ERROR loading FlowVR mesh file "<<filename<<std::endl;
            return NULL;
        }
        //std::cout << "bbox = "<<mesh.bb<<std::endl;

        if (!mesh.getAttrib(flowvr::render::Mesh::MESH_DISTMAP))
        {
            std::cerr << "ERROR: FlowVR mesh "<<filename<<" does not contain distance information. Please use flowvr-distmap."<<std::endl;
            return NULL;
        }
        nx = mesh.distmap->nx;
        ny = mesh.distmap->ny;
        nz = mesh.distmap->nz;
        ftl::Vec3f fpmin = ftl::transform(mesh.distmap->mat,ftl::Vec3f(0,0,0))*scale;
        ftl::Vec3f fpmax = ftl::transform(mesh.distmap->mat,ftl::Vec3f(nx-1,ny-1,nz-1))*scale;
        pmin = Coord(fpmin.ptr());
        pmax = Coord(fpmax.ptr());
        std::cout << "Copying "<<nx<<"x"<<ny<<"x"<<nz<<" distance grid in <"<<pmin<<">-<"<<pmax<<">"<<std::endl;
        CudaDistanceGrid* grid = new CudaDistanceGrid(nx, ny, nz, pmin, pmax);
        for (int i=0; i< grid->nxnynz; i++)
            grid->dists[i] = mesh.distmap->data[i]*scale;

        if (mesh.getAttrib(flowvr::render::Mesh::MESH_POINTS_GROUP))
        {
            int nbpos = 0;
            for (int i=0; i<mesh.nbg(); i++)
            {
                if (mesh.getGP0(i) >= 0)
                    ++nbpos;
            }
            std::cout << "Copying "<<nbpos<<" mesh vertices."<<std::endl;
            grid->meshPts.resize(nbpos);
            int p = 0;
            for (int i=0; i<mesh.nbg(); i++)
            {
                int p0 = mesh.getGP0(i);
                if (p0 >= 0)
                    grid->meshPts[p++] = Coord(mesh.getPP(p0).ptr())*scale;
            }
        }
        else
        {
            int nbpos = mesh.nbp();
            std::cout << "Copying "<<nbpos<<" mesh vertices."<<std::endl;
            grid->meshPts.resize(nbpos);
            for (int i=0; i<nbpos; i++)
                grid->meshPts[i] = Coord(mesh.getPP(i).ptr())*scale;
        }
        grid->computeBBox();
        std::cout << "Distance grid creation DONE."<<std::endl;
        return grid;
    }
#endif
    else if (filename.length()>4 && filename.substr(filename.length()-4) == ".obj")
    {
        sofa::helper::io::Mesh* mesh = sofa::helper::io::Mesh::Create(filename);
        const sofa::helper::vector<Vector3> & vertices = mesh->getVertices();

        std::cout << "Computing bbox."<<std::endl;
        Coord bbmin, bbmax;
        if (!vertices.empty())
        {
            bbmin = vertices[0];
            bbmax = bbmin;
            for(unsigned int i=1; i<vertices.size(); i++)
            {
                for (int c=0; c<3; c++)
                    if (vertices[i][c] < bbmin[c]) bbmin[c] = (Real)vertices[i][c];
                    else if (vertices[i][c] > bbmax[c]) bbmax[c] = (Real)vertices[i][c];
            }
            bbmin *= scale;
            bbmax *= scale;
        }
        std::cout << "bbox = <"<<bbmin<<">-<"<<bbmax<<">"<<std::endl;

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
        std::cout << "Creating distance grid in <"<<pmin<<">-<"<<pmax<<">"<<std::endl;
        CudaDistanceGrid* grid = new CudaDistanceGrid(nx, ny, nz, pmin, pmax);
        std::cout << "Copying "<<vertices.size()<<" mesh vertices."<<std::endl;
        grid->meshPts.resize(vertices.size());
        for(unsigned int i=0; i<vertices.size(); i++)
            grid->meshPts[i] = vertices[i]*scale;
        const sofa::helper::vector<sofa::helper::vector<sofa::helper::vector<int> > > & facets = mesh->getFacets();
        int nbt = 0;
        int nbq = 0;
        for (unsigned int i=0; i<facets.size(); i++)
        {
            const sofa::helper::vector<int>& pts = facets[i][0];
            if (pts.size() == 4)
                ++nbq;
            else if (pts.size() >= 3)
                nbt += pts.size()-2;
        }
        grid->meshTriangles.resize(nbt);
        grid->meshQuads.resize(nbq);
        nbt=0;
        nbq=0;
        for (unsigned int i=0; i<facets.size(); i++)
        {
            const sofa::helper::vector<int>& pts = facets[i][0];
            if (pts.size() == 4)
                grid->meshQuads[nbq++] = sofa::component::topology::MeshTopology::Quad(pts[0],pts[1],pts[2],pts[3]);
            else if (pts.size() >= 3)
                for (unsigned int j=2; j<pts.size(); j++)
                    grid->meshTriangles[nbt++] = sofa::component::topology::MeshTopology::Triangle(pts[0],pts[j-1],pts[j]);
        }
        std::cout << "Computing distance field."<<std::endl;
        grid->calcDistance();
        grid->computeBBox();
        std::cout << "Distance grid creation DONE."<<std::endl;
        delete mesh;
        return grid;
    }
    else
    {
        std::cerr << "Unknown extension: "<<filename<<std::endl;
        return NULL;
    }
}

bool CudaDistanceGrid::save(const std::string& filename)
{
    /// !!!TODO!!! ///
    if (filename.length()>4 && filename.substr(filename.length()-4) == ".raw")
    {
        std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
        out.write((const char*)&(dists[0]), nxnynz*sizeof(Real));
    }
    else
    {
        std::cerr << " CudaDistanceGrid::save(): Unsupported extension: "<<filename<<std::endl;
        return false;
    }
    return true;
}


void CudaDistanceGrid::computeBBox()
{
    if (!meshPts.empty())
    {
        bbmin = meshPts[0];
        bbmax = bbmin;
        for(unsigned int i=1; i<meshPts.size(); i++)
        {
            for (int c=0; c<3; c++)
                if (meshPts[i][c] < bbmin[c]) bbmin[c] = (Real)meshPts[i][c];
                else if (meshPts[i][c] > bbmax[c]) bbmax[c] = (Real)meshPts[i][c];
        }
    }
    else
    {
        bbmin = pmin;
        bbmax = pmax;
        /// \TODO compute the real bbox from the grid content
    }
}


/// Compute distance field for a cube of the given half-size.
/// Also create a mesh of points using np points per axis
void CudaDistanceGrid::calcCubeDistance(Real dim, int np)
{
    cubeDim = dim;
    if (np > 1)
    {
        int nbp = np*np*np - (np-2)*(np-2)*(np-2);
        //std::cout << "Copying "<<nbp<<" cube vertices."<<std::endl;
        meshPts.resize(nbp);

        for (int i=0,z=0; z<np; z++)
            for (int y=0; y<np; y++)
                for (int x=0; x<np; x++)
                    if (z==0 || z==np-1 || y==0 || y==np-1 || x==0 || x==np-1)
                        meshPts[i++] = Coord(x*dim*2/(np-1) - dim, y*dim*2/(np-1) - dim, z*dim*2/(np-1) - dim);
    }

    //std::cout << "Computing distance field."<<std::endl;

    Real dim2 = dim; //*0.75f; // add some 'roundness' to the cubes corner

    for (int i=0,z=0; z<nz; z++)
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx; x++,i++)
            {
                Coord p = coord(x,y,z);
                Coord s = p;
                bool out = false;
                for (int c=0; c<3; c++)
                {
                    if (s[c] < -dim2) { s[c] = -dim2; out = true; }
                    else if (s[c] >  dim2) { s[c] =  dim2; out = true; }
                }
                Real d;
                if (out)
                    d = (p - s).norm();
                else
                    d = rmax(rmax(rabs(s[0]),rabs(s[1])),rabs(s[2])) - dim2;
                dists[i] = d - (dim-dim2);
            }
    //computeBBox();
    bbmin = Coord(-dim,-dim,-dim);
    bbmax = Coord( dim, dim, dim);
}

/// Compute distance field from given mesh
void CudaDistanceGrid::calcDistance()
{
#ifdef SOFA_HAVE_GLEW

    if (GLEW_EXT_framebuffer_object && GLEW_ARB_vertex_buffer_object)
    {
        static GLuint fbid = 0;
        if (!fbid)
        {
            glGenFramebuffersEXT(1,&fbid);
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbid);
        }

        /*
        if (!texcolor)
        glGenTextures(1, &texcolor);

        glBindTexture(target, texcolor);

        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexImage2D(target, 0, GL_RGBA8, nx, ny, 0,
        	 GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        if (!texdepth)
        glGenTextures(1, &texdepth);
            glBindTexture(target, texdepth);

        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexImage2D(target, 0, GL_DEPTH_COMPONENT24, nx, ny, 0,
        	 GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, target, texcolor, 0);*/
    }
    else
    {
        std::cerr << "ERROR: Unsupported OpenGL extensions EXT_framebuffer_object ARB_vertex_buffer_object" << std::endl;
    }

#else
    std::cerr << "ERROR: CudaDistanceGrid::calcDistance requires GLEW to access OpenGL extensions" << std::endl;

#endif
}

CudaDistanceGrid* CudaDistanceGrid::loadShared(const std::string& filename, double scale, int nx, int ny, int nz, Coord pmin, Coord pmax)
{
    CudaDistanceGridParams params;
    params.filename = filename;
    params.scale = scale;
    params.nx = nx;
    params.ny = ny;
    params.nz = nz;
    params.pmin = pmin;
    params.pmax = pmax;
    std::map<CudaDistanceGridParams, CudaDistanceGrid*>& shared = getShared();
    std::map<CudaDistanceGridParams, CudaDistanceGrid*>::iterator it = shared.find(params);
    if (it != shared.end())
        return it->second->addRef();
    else
    {
        return shared[params] = load(filename, scale, nx, ny, nz, pmin, pmax);
    }
}

std::map<CudaDistanceGrid::CudaDistanceGridParams, CudaDistanceGrid*>& CudaDistanceGrid::getShared()
{
    static std::map<CudaDistanceGridParams, CudaDistanceGrid*> instance;
    return instance;
}

} // namespace cuda

} // namespace gpu

} // namespace sofa
