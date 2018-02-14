/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifdef SOFA_HAVE_GLEW
#include <GL/glew.h>
#endif
#ifdef SOFA_HAVE_MINIFLOWVR
    #include <flowvr/render/mesh.h>
#endif // SOFA_HAVE_MINIFLOWVR
#include "CudaDistanceGridCollisionModel.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <fstream>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/rmath.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaDistanceGridCollisionModel)

int CudaRigidDistanceGridCollisionModelClass = core::RegisterObject("GPU-based grid distance field using CUDA")
        .add< CudaRigidDistanceGridCollisionModel >()
        .addAlias("CudaDistanceGridCollisionModel")
        .addAlias("CudaRigidDistanceGrid")
        .addAlias("CudaDistanceGrid")
        ;

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

CudaDistanceGrid::~CudaDistanceGrid()
{
    std::map<CudaDistanceGridParams, CudaDistanceGrid*>& shared = getShared();
    std::map<CudaDistanceGridParams, CudaDistanceGrid*>::iterator it = shared.begin();
    while (it != shared.end() && it->second != this) ++it;
    if (it != shared.end())
        shared.erase(it); // remove this grid from the list of already loaded grids
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

CudaDistanceGrid* CudaDistanceGrid::load(const std::string& filename, double scale, double sampling, int nx, int ny, int nz, Coord pmin, Coord pmax)
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
        if (sampling)
            grid->sampleSurface(sampling);
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
        if (sampling)
            grid->sampleSurface(sampling);
        return grid;
    }
#ifdef SOFA_HAVE_MINIFLOWVR
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
        ftl::Vec3f fpmin = ftl::transform(mesh.distmap->mat,ftl::Vec3f(0,0,0))*(float)scale;
        ftl::Vec3f fpmax = ftl::transform(mesh.distmap->mat,ftl::Vec3f((float)(nx-1),(float)(ny-1),(float)(nz-1)))*(float)scale;
        pmin = Coord(fpmin.ptr());
        pmax = Coord(fpmax.ptr());
        std::cout << "Copying "<<nx<<"x"<<ny<<"x"<<nz<<" distance grid in <"<<pmin<<">-<"<<pmax<<">"<<std::endl;
        CudaDistanceGrid* grid = new CudaDistanceGrid(nx, ny, nz, pmin, pmax);
        for (int i=0; i< grid->nxnynz; i++)
            grid->dists[i] = mesh.distmap->data[i]*(float)scale;

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
#endif // SOFA_HAVE_MINIFLOWVR
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
                grid->meshQuads[nbq++] = sofa::core::topology::BaseMeshTopology::Quad(pts[0],pts[1],pts[2],pts[3]);
            else if (pts.size() >= 3)
                for (unsigned int j=2; j<pts.size(); j++)
                    grid->meshTriangles[nbt++] = sofa::core::topology::BaseMeshTopology::Triangle(pts[0],pts[j-1],pts[j]);
        }
        std::cout << "Computing distance field."<<std::endl;
        grid->calcDistance();
        if (sampling)
            grid->sampleSurface(sampling);
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

/// Sample the surface with points approximately separated by the given sampling distance (expressed in voxels if the value is negative)
void CudaDistanceGrid::sampleSurface(double sampling)
{
    std::cout << "CudaDistanceGrid: sample surface with sampling distance " << sampling << std::endl;
    int stepX, stepY, stepZ;
    if (sampling < 0)
    {
        stepX = stepY = stepZ = (int)(-sampling);
    }
    else
    {
        stepX = (int)(sampling/cellWidth[0]);
        stepY = (int)(sampling/cellWidth[1]);
        stepZ = (int)(sampling/cellWidth[2]);
    }
    if (stepX < 1) stepX = 1;
    if (stepY < 1) stepY = 1;
    if (stepZ < 1) stepZ = 1;
    std::cout << "CudaDistanceGrid: sampling steps: " << stepX << " " << stepY << " " << stepZ << std::endl;

    SReal maxD = (SReal)sqrt((cellWidth[0]*stepX)*(cellWidth[0]*stepX) + (cellWidth[1]*stepY)*(cellWidth[1]*stepY) + (cellWidth[2]*stepZ)*(cellWidth[2]*stepZ));
    std::vector<Coord> pts;
    for (int z=1; z<nz-1; z+=stepZ)
        for (int y=1; y<ny-1; y+=stepY)
            for (int x=1; x<nx-1; x+=stepX)
            {
                SReal d = dists[index(x,y,z)];
                if (helper::rabs(d) > maxD) continue;

                Vector3 pos = coord(x,y,z);
                Vector3 n = grad(index(x,y,z), Coord()); // note that there are some redundant computations between interp() and grad()
                n.normalize();
                pos -= n * (d * 0.99); // push pos back to the surface
                d = interp(pos);
                int it = 1;
                while (helper::rabs(d) > 0.01f*maxD && it < 10)
                {
                    n = grad(pos);
                    n.normalize();
                    pos -= n * (d * 0.99); // push pos back to the surface
                    d = interp(pos);
                    ++it;
                }
                if (it == 10 && helper::rabs(d) > 0.1f*maxD)
                {
                    std::cout << "Failed to converge at ("<<x<<","<<y<<","<<z<<"):"
                            << " pos0 = " << coord(x,y,z) << " d0 = " << dists[index(x,y,z)] << " grad0 = " << grad(index(x,y,z), Coord())
                            << " pos = " << pos << " d = " << d << " grad = " << n << std::endl;
                    continue;
                }
                Coord p = pos;
                pts.push_back(p);
            }
    std::cout << "CudaDistanceGrid: " << pts.size() << " sampling points created." << std::endl;
    meshPts.resize(pts.size());
    for (unsigned int p=0; p<pts.size(); ++p)
        meshPts[p] = pts[p];
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
                    d = helper::rmax(helper::rmax(helper::rabs(s[0]),helper::rabs(s[1])),helper::rabs(s[2])) - dim2;
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

CudaDistanceGrid* CudaDistanceGrid::loadShared(const std::string& filename, double scale, double sampling, int nx, int ny, int nz, Coord pmin, Coord pmax)
{
    CudaDistanceGridParams params;
    params.filename = filename;
    params.scale = scale;
    params.sampling = sampling;
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
        return shared[params] = load(filename, scale, sampling, nx, ny, nz, pmin, pmax);
    }
}

std::map<CudaDistanceGrid::CudaDistanceGridParams, CudaDistanceGrid*>& CudaDistanceGrid::getShared()
{
    static std::map<CudaDistanceGridParams, CudaDistanceGrid*> instance;
    return instance;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

CudaRigidDistanceGridCollisionModel::CudaRigidDistanceGridCollisionModel()
    : modified(true)
    , fileCudaRigidDistanceGrid( initData( &fileCudaRigidDistanceGrid, "fileCudaRigidDistanceGrid", "load distance grid from specified file"))
    , scale( initData( &scale, 1.0, "scale", "scaling factor for input file"))
    , sampling( initData( &sampling, 0.0, "sampling", "if not zero: sample the surface with points approximately separated by the given sampling distance (expressed in voxels if the value is negative)"))
    , box( initData( &box, "box", "Field bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , nx( initData( &nx, 64, "nx", "number of values on X axis") )
    , ny( initData( &ny, 64, "ny", "number of values on Y axis") )
    , nz( initData( &nz, 64, "nz", "number of values on Z axis") )
    , dumpfilename( initData( &dumpfilename, "dumpfilename","write distance grid to specified file"))
    , usePoints( initData( &usePoints, true, "usePoints", "use mesh vertices for collision detection"))
{
    rigid = NULL;
    addAlias(&fileCudaRigidDistanceGrid,"filename");
}

CudaRigidDistanceGridCollisionModel::~CudaRigidDistanceGridCollisionModel()
{
    for (unsigned int i=0; i<elems.size(); i++)
    {
        if (elems[i].grid!=NULL) elems[i].grid->release();
        if (elems[i].prevGrid!=NULL) elems[i].prevGrid->release();
    }
}

void CudaRigidDistanceGridCollisionModel::init()
{
    std::cout << "> CudaRigidDistanceGridCollisionModel::init()"<<std::endl;
    this->core::CollisionModel::init();
    rigid = dynamic_cast< core::behavior::MechanicalState<RigidTypes>* > (getContext()->getMechanicalState());

    CudaDistanceGrid* grid = NULL;
    if (fileCudaRigidDistanceGrid.getValue().empty())
    {
        if (elems.size()==0 || elems[0].grid==NULL)
            std::cerr << "ERROR: CudaRigidDistanceGridCollisionModel requires an input filename.\n";
        // else the grid has already been set
        return;
    }
    std::cout << "CudaRigidDistanceGridCollisionModel: creating "<<nx.getValue()<<"x"<<ny.getValue()<<"x"<<nz.getValue()<<" DistanceGrid from file "<<fileCudaRigidDistanceGrid.getValue();
    if (scale.getValue()!=1.0) std::cout<<" scale="<<scale.getValue();
    if (sampling.getValue()!=0.0) std::cout<<" sampling="<<sampling.getValue();
    if (box.getValue()[0][0]<box.getValue()[1][0]) std::cout<<" bbox=<"<<box.getValue()[0]<<">-<"<<box.getValue()[0]<<">";
    std::cout << std::endl;
    grid = CudaDistanceGrid::loadShared(fileCudaRigidDistanceGrid.getFullPath(), scale.getValue(), sampling.getValue(), nx.getValue(),ny.getValue(),nz.getValue(),box.getValue()[0],box.getValue()[1]);

    resize(1);
    elems[0].grid = grid;
    if (grid && !dumpfilename.getValue().empty())
    {
        std::cout << "CudaRigidDistanceGridCollisionModel: dump grid to "<<dumpfilename.getValue()<<std::endl;
        grid->save(dumpfilename.getFullPath());
    }
    std::cout << "< CudaRigidDistanceGridCollisionModel::init()"<<std::endl;
}

void CudaRigidDistanceGridCollisionModel::resize(int s)
{
    this->core::CollisionModel::resize(s);
    elems.resize(s);
}

void CudaRigidDistanceGridCollisionModel::setGrid(CudaDistanceGrid* surf, int index)
{
    if (elems[index].grid == surf) return;
    if (elems[index].grid!=NULL) elems[index].grid->release();
    elems[index].grid = surf->addRef();
    modified = true;
}

void CudaRigidDistanceGridCollisionModel::setNewState(int index, double dt, CudaDistanceGrid* grid, const Matrix3& rotation, const Vector3& translation)
{
    grid->addRef();
    if (elems[index].prevGrid!=NULL)
        elems[index].prevGrid->release();
    elems[index].prevGrid = elems[index].grid;
    elems[index].grid = grid;
    elems[index].prevRotation = elems[index].rotation;
    elems[index].rotation = rotation;
    elems[index].prevTranslation = elems[index].translation;
    elems[index].translation = translation;
    if (!elems[index].isTransformed)
    {
        Matrix3 I; I.identity();
        if (!(rotation == I) || !(translation == Vector3()))
            elems[index].isTransformed = true;
    }
    elems[index].prevDt = dt;
    modified = true;
}

using sofa::component::collision::CubeModel;

/// Create or update the bounding volume hierarchy.
void CudaRigidDistanceGridCollisionModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = this->createPrevious<CubeModel>();

    if (!modified && !isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    updateGrid();

    cubeModel->resize(size);
    for (int i=0; i<size; i++)
    {
        //static_cast<DistanceGridCollisionElement*>(elems[i])->recalcBBox();
        Vector3 emin, emax;
        if (rigid)
        {
            const RigidTypes::Coord& xform = rigid->read(core::ConstVecCoordId::position())->getValue()[i];
            elems[i].translation = xform.getCenter();
            xform.getOrientation().toMatrix(elems[i].rotation);
            elems[i].isTransformed = true;
        }
        if (elems[i].isTransformed)
        {
            //std::cout << "Grid "<<i<<" transformation: <"<<elems[i].rotation<<"> x + <"<<elems[i].translation<<">"<<std::endl;
            Vector3 corner = elems[i].translation + elems[i].rotation * elems[i].grid->getBBCorner(0);
            emin = corner;
            emax = emin;
            for (int j=1; j<8; j++)
            {
                corner = elems[i].translation + elems[i].rotation * elems[i].grid->getBBCorner(j);
                for(int c=0; c<3; c++)
                    if (corner[c] < emin[c]) emin[c] = corner[c];
                    else if (corner[c] > emax[c]) emax[c] = corner[c];
            }
        }
        else
        {
            emin = elems[i].grid->getBBMin();
            emax = elems[i].grid->getBBMax();
        }
        cubeModel->setParentOf(i, emin, emax); // define the bounding box of the current element
        //std::cout << "Grid "<<i<<" within  <"<<emin<<">-<"<<emax<<">"<<std::endl;
    }
    cubeModel->computeBoundingTree(maxDepth);
    modified = false;
}

void CudaRigidDistanceGridCollisionModel::updateGrid()
{
}

void CudaRigidDistanceGridCollisionModel::draw(const core::visual::VisualParams* vparams)
{
    if (!isActive()) return;
    if (vparams->displayFlags().getShowCollisionModels())
    {
        if (vparams->displayFlags().getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDisable(GL_LIGHTING);
        glColor4fv(getColor4f());
        glPointSize(3);
        for (unsigned int i=0; i<elems.size(); i++)
        {
            draw(vparams,i);
        }
        glPointSize(1);
        if (vparams->displayFlags().getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (getPrevious()!=NULL)
        getPrevious()->draw(vparams);
}

void CudaRigidDistanceGridCollisionModel::draw(const core::visual::VisualParams* ,int index)
{
    if (elems[index].isTransformed)
    {
        glPushMatrix();
        // float m[16];
        // (*rigid->getX())[index].writeOpenGlMatrix( m );
        // glMultMatrixf(m);
        Matrix4 m;
        m.identity();
        m = elems[index].rotation;
        m.transpose();
        m[3] = Vector4(elems[index].translation,1.0);
        helper::gl::glMultMatrix(m.ptr());
    }

    CudaDistanceGrid* grid = getGrid(index);
    CudaDistanceGrid::Coord corners[8];
    for(unsigned int i=0; i<8; i++)
        corners[i] = grid->getCorner(i);
    //glEnable(GL_BLEND);
    //glDepthMask(0);
    if (!isMoving())
        glColor4f(0.25f, 0.25f, 0.25f, 0.1f);
    else
        glColor4f(0.5f, 0.5f, 0.5f, 0.1f);
    glBegin(GL_LINES);
    {
        glVertex3fv(corners[0].ptr()); glVertex3fv(corners[4].ptr());
        glVertex3fv(corners[1].ptr()); glVertex3fv(corners[5].ptr());
        glVertex3fv(corners[2].ptr()); glVertex3fv(corners[6].ptr());
        glVertex3fv(corners[3].ptr()); glVertex3fv(corners[7].ptr());
        glVertex3fv(corners[0].ptr()); glVertex3fv(corners[2].ptr());
        glVertex3fv(corners[1].ptr()); glVertex3fv(corners[3].ptr());
        glVertex3fv(corners[4].ptr()); glVertex3fv(corners[6].ptr());
        glVertex3fv(corners[5].ptr()); glVertex3fv(corners[7].ptr());
        glVertex3fv(corners[0].ptr()); glVertex3fv(corners[1].ptr());
        glVertex3fv(corners[2].ptr()); glVertex3fv(corners[3].ptr());
        glVertex3fv(corners[4].ptr()); glVertex3fv(corners[5].ptr());
        glVertex3fv(corners[6].ptr()); glVertex3fv(corners[7].ptr());
    }
    glEnd();
    glDisable(GL_BLEND);
    glDepthMask(1);
    for(unsigned int i=0; i<8; i++)
        corners[i] = grid->getBBCorner(i);
    //glEnable(GL_BLEND);
    //glDepthMask(0);

    if (!isMoving())
        glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
    else
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_LINES);
    {
        glVertex3fv(corners[0].ptr()); glVertex3fv(corners[4].ptr());
        glVertex3fv(corners[1].ptr()); glVertex3fv(corners[5].ptr());
        glVertex3fv(corners[2].ptr()); glVertex3fv(corners[6].ptr());
        glVertex3fv(corners[3].ptr()); glVertex3fv(corners[7].ptr());
        glVertex3fv(corners[0].ptr()); glVertex3fv(corners[2].ptr());
        glVertex3fv(corners[1].ptr()); glVertex3fv(corners[3].ptr());
        glVertex3fv(corners[4].ptr()); glVertex3fv(corners[6].ptr());
        glVertex3fv(corners[5].ptr()); glVertex3fv(corners[7].ptr());
        glVertex3fv(corners[0].ptr()); glVertex3fv(corners[1].ptr());
        glVertex3fv(corners[2].ptr()); glVertex3fv(corners[3].ptr());
        glVertex3fv(corners[4].ptr()); glVertex3fv(corners[5].ptr());
        glVertex3fv(corners[6].ptr()); glVertex3fv(corners[7].ptr());
    }
    glEnd();

    const float mindist = -(grid->getPMax()-grid->getPMin()).norm()*0.1f;
    const float maxdist = (grid->getPMax()-grid->getPMin()).norm()*0.025f;

    if (grid->meshPts.empty())
    {
        glBegin(GL_POINTS);
        {
            for (int z=0, ind=0; z<grid->getNz(); z++)
                for (int y=0; y<grid->getNy(); y++)
                    for (int x=0; x<grid->getNx(); x++, ind++)
                    {
                        CudaDistanceGrid::Coord p = grid->coord(x,y,z);
                        CudaDistanceGrid::Real d = (*grid)[ind];
                        if (d < mindist || d > maxdist) continue;
                        d /= maxdist;
                        if (d<0)
                            glColor3d(1+d*0.25, 0, 1+d);
                        else
                            glColor3d(0, 1-d*0.25, 1-d);
                        glVertex3fv(p.ptr());
                    }
        }
        glEnd();
    }
    else
    {
        glColor3d(1, 1 ,1);
        glBegin(GL_POINTS);
        for (unsigned int i=0; i<grid->meshPts.size(); i++)
        {
            glVertex3fv(grid->meshPts[i].ptr());
        }
        glEnd();
        glBegin(GL_LINES);
        for (unsigned int i=0; i<grid->meshPts.size(); i++)
        {
            CudaDistanceGrid::Coord p ( grid->meshPts[i].ptr() );
            glColor3d(1, 1 ,1);
            CudaDistanceGrid::Coord grad = grid->grad(p);
            grad.normalize();
            for (int j = -2; j <= 2; j++)
            {
                CudaDistanceGrid::Coord p2 = p + grad * (j*maxdist/2);
                CudaDistanceGrid::Real d = grid->eval(p2);
                //if (rabs(d) > maxdist) continue;
                d /= maxdist;
                if (d<0)
                    glColor3d(1+d*0.25, 0, 1+d);
                else
                    glColor3d(0, 1-d*0.25, 1-d);
                glVertex3fv(p2.ptr());
                if (j>-2 && j < 2)
                    glVertex3fv(p2.ptr());
            }
        }
        glEnd();
    }
    if (elems[index].isTransformed)
    {
        glPopMatrix();
    }
}




} // namespace cuda

} // namespace gpu

} // namespace sofa
