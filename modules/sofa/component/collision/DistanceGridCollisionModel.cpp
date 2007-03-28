#include <sofa/component/collision/DistanceGridCollisionModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/collision/CubeModel.h>
#include <fstream>
#include <GL/gl.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(DistanceGridCollisionModel)

int DistanceGridCollisionModelClass = core::RegisterObject("Grid-based distance field, based on SLC code ( http://sourceforge.net/projects/bfast/ )")
        .add< DistanceGridCollisionModel >()
        .addAlias("DistanceGrid")
        ;

using namespace defaulttype;

void DistanceGridCollisionModel::draw(int index)
{
    DistanceGrid* grid = getGrid(index);
    DistanceGrid::Coord corners[8];
    for(unsigned int i=0; i<8; i++)
        corners[i] = grid->getCorner(i);
    //glEnable(GL_BLEND);
    //glDepthMask(0);
    if (isStatic())
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
    glBegin(GL_POINTS);
    {

        for (int z=0, ind=0; z<grid->getNz(); z++)
            for (int y=0; y<grid->getNy(); y++)
                for (int x=0; x<grid->getNx(); x++, ind++)
                {
                    DistanceGrid::Coord p = grid->coord(x,y,z);
                    DistanceGrid::Real d = (*grid)[ind];
                    if (rabs(d) > 4) continue;
                    if (d<0)
                        glColor3d(1-d*0.25, 0, 1-d);
                    else
                        glColor3d(0, 1-d*0.25, 1-d);
                    glVertex3fv(p.ptr());
                }
    }
    glEnd();
}

DistanceGridCollisionModel::DistanceGridCollisionModel()
    : filename( dataField( &filename, "filename","load distance grid from specified file"))
    , box( dataField( &box, "box", "Field bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , nx( dataField( &nx, 64, "nx", "number of values on X axis") )
    , ny( dataField( &ny, 64, "ny", "number of values on Y axis") )
    , nz( dataField( &nz, 64, "nz", "number of values on Z axis") )
    , dumpfilename( dataField( &dumpfilename, "dumpfilename","write distance grid to specified file"))
{
    mstate = NULL;
    rigid = NULL;
    mesh = NULL;
    previous = NULL;
    next = NULL;
    static_ = false;
}

DistanceGridCollisionModel::~DistanceGridCollisionModel()
{
    for (unsigned int i=0; i<elems.size(); i++)
        if (elems[i]!=NULL) delete elems[i];
}

void DistanceGridCollisionModel::init()
{
    std::cout << "> DistanceGridCollisionModel::init()"<<std::endl;
    this->core::CollisionModel::init();
    mstate = dynamic_cast< core::componentmodel::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());
    rigid = dynamic_cast< core::componentmodel::behavior::MechanicalState<RigidTypes>* > (getContext()->getMechanicalState());
    mesh = dynamic_cast< topology::MeshTopology* > (getContext()->getTopology());

    DistanceGrid* grid = NULL;
    if (filename.getValue().empty())
    {
        if (mstate==NULL)
        {
            std::cerr << "ERROR: DistanceGridCollisionModel requires a Vec3 Mechanical Model or a filename.\n";
            return;
        }
        /*
        if (mstate->getX()->size()!=1)
        {
            std::cerr << "ERROR: DistanceGridCollisionModel requires a Vec3 Mechanical Model with 1 element.\n";
            return;
        }
        */
        if (mesh==NULL)
        {
            std::cerr << "ERROR: DistanceGridCollisionModel requires a Mesh Topology or a filename.\n";
            return;
        }

        if (mesh->getNbTriangles()==0 || !mesh->hasPos())
        {
            std::cerr << "ERROR: DistanceGridCollisionModel requires a Mesh Topology with triangles and vertice positions or a filename.\n";
            return;
        }
        grid = new DistanceGrid(nx.getValue(),ny.getValue(),nz.getValue(),box.getValue()[0],box.getValue()[1]);
    }
    else
    {
        std::cout << "DistanceGridCollisionModel: creating DistanceGrid from file "<<filename.getValue()<<std::endl;
        grid = DistanceGrid::loadShared(filename.getValue(), nx.getValue(),ny.getValue(),nz.getValue(),box.getValue()[0],box.getValue()[1]);
    }
    resize(1);
    elems[0] = grid;
    if (grid && !dumpfilename.getValue().empty())
    {
        std::cout << "DistanceGridCollisionModel: dump grid to "<<dumpfilename.getValue()<<std::endl;
        grid->save(dumpfilename.getValue());
    }
    std::cout << "< DistanceGridCollisionModel::init()"<<std::endl;
}

void DistanceGridCollisionModel::resize(int s)
{
    this->core::CollisionModel::resize(s);
    elems.resize(s);
}

DistanceGrid* DistanceGridCollisionModel::getGrid(int index)
{
    return elems[index];
}

void DistanceGridCollisionModel::setGrid(DistanceGrid* surf, int index)
{
    elems[index] = surf;
}

void DistanceGridCollisionModel::updateGrid()
{
    if (elems.size()<1) return;
    //DistanceGrid* grid = elems[0];
    if (filename.getValue().empty())
    {
        /*
                grid->builtFromTriangles = true;
                if (mstate != NULL)
                {
                    const Vec3Types::VecCoord& x = *mstate->getX();
                    unsigned int nbp = x.size();
                    grid->meshPts.setNumVecs(nbp);
                    for (unsigned int i=0;i<nbp;i++)
                    {
                        Vec3d p = x[i];
                        grid->meshPts[i][0] = p[0];
                        grid->meshPts[i][1] = p[1];
                        grid->meshPts[i][2] = p[2];
                    }
                }
                else
                {
                    unsigned int nbp = mesh->getNbPoints();
                    grid->meshPts.setNumVecs(nbp);
                    for (unsigned int i=0;i<nbp;i++)
                    {
                        grid->meshPts[i][0] = mesh->getPX(i);
                        grid->meshPts[i][1] = mesh->getPY(i);
                        grid->meshPts[i][2] = mesh->getPZ(i);
                    }
                }
                const topology::MeshTopology::SeqTriangles& tris = mesh->getTriangles();
                grid->triangles.setNumTriangles(tris.size());
                for (unsigned int i=0;i<tris.size();i++)
                {
                    grid->triangles[i].a = tris[i][0];
                    grid->triangles[i].b = tris[i][1];
                    grid->triangles[i].c = tris[i][2];
                }
        */
    }
    /*
        // Compute mesh bounding box
        BfastVector3 bbmin;
        BfastVector3 bbmax;
        unsigned int nbp = grid->meshPts.numVecs();
        for (unsigned int i=0;i<nbp;i++)
        {
            const BfastVector3& p = grid->meshPts[i];
            if (!i || p[0] < bbmin[0]) bbmin[0] = p[0];
            if (!i || p[0] > bbmax[0]) bbmax[0] = p[0];
            if (!i || p[1] < bbmin[1]) bbmin[1] = p[1];
            if (!i || p[1] > bbmax[1]) bbmax[1] = p[1];
            if (!i || p[2] < bbmin[2]) bbmin[2] = p[2];
            if (!i || p[2] > bbmax[2]) bbmax[2] = p[2];
        }
        // Grid's bounding box should be square
        BfastReal width = bbmax[0] - bbmin[0];
        if (bbmax[1] - bbmin[1] > width) width = bbmax[1] - bbmin[1];
        if (bbmax[2] - bbmin[2] > width) width = bbmax[2] - bbmin[2];
        width *= (1+2*border); // Build a bigger grid
        BfastReal xcenter = (bbmin[0] + bbmax[0])/2;
        BfastReal ycenter = (bbmin[1] + bbmax[1])/2;
        BfastReal zcenter = (bbmin[2] + bbmax[2])/2;
        bbmin[0] = xcenter - width/2; bbmax[0] = xcenter + width/2;
        bbmin[1] = ycenter - width/2; bbmax[1] = ycenter + width/2;
        bbmin[2] = zcenter - width/2; bbmax[2] = zcenter + width/2;

        std::cout << "Building Distance Grid with " << grid->meshPts.numVecs() << " vertices and " << grid->triangles.numTriangles() << " triangles, bbox=<"<<bbmin[0]<<','<<bbmin[1]<<','<<bbmin[2]<<">-<"<<bbmax[0]<<','<<bbmax[1]<<','<<bbmax[2]<<">." << std::endl;
        grid->computeNormals();
        grid->computePsuedoNormals();
        delete grid->tree;
        grid->tree = new DtTree;
        buildTree(grid->tree, bbmin, bbmax, depth, &(grid->meshPts), &(grid->triangles), &(grid->faceNormals));
        grid->redistance();
        std::cout << "Grid built: " << grid->tree->cells.numCells() << " cells." << std::endl;
    */
}

void DistanceGridCollisionModel::draw()
{
    if (!isActive() || !getContext()->getShowCollisionModels()) return;

    if (rigid!=NULL)
    {
        glPushMatrix();
        float m[16];
        (*rigid->getX())[0].writeOpenGlMatrix( m );
        glMultMatrixf(m);
    }

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDisable(GL_LIGHTING);
    if (isStatic())
        glColor3f(0.5, 0.5, 0.5);
    else
        glColor3f(1.0, 0.0, 0.0);
    glPointSize(3);
    for (unsigned int i=0; i<elems.size(); i++)
    {
        draw(i);
    }
    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (getPrevious()!=NULL && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();

    if (rigid!=NULL)
    {
        glPopMatrix();
    }
}

/// Create or update the bounding volume hierarchy.
void DistanceGridCollisionModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = this->createPrevious<CubeModel>();

    if (isStatic() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    if (filename.getValue().empty())
        updateGrid();

    for (unsigned int i=0; i<elems.size(); i++)
    {
        //static_cast<DistanceGridCollisionElement*>(elems[i])->recalcBBox();
        Vector3 emin, emax;
        for (int c = 0; c < 3; c++)
        {
            emin[c] = elems[i]->getPMin()[c];
            emax[c] = elems[i]->getPMax()[c];
        }
        cubeModel->setParentOf(i, emin, emax); // define the bounding box of the current triangle
    }
    cubeModel->computeBoundingTree(maxDepth);
}

DistanceGrid::DistanceGrid(int nx, int ny, int nz, Coord pmin, Coord pmax)
    : nx(nx), ny(ny), nz(nz), nxny(nx*ny), nxnynz(nx*ny*nz),
      pmin(pmin), pmax(pmax),
      cellWidth   ((pmax[0]-pmin[0])/(nx-1), (pmax[1]-pmin[1])/(ny-1),(pmax[2]-pmin[2])/(nz-1)),
      invCellWidth((nx-1)/(pmax[0]-pmin[0]), (ny-1)/(pmax[1]-pmin[1]),(nz-1)/(pmax[2]-pmin[2]))
{
    dists.resize(nxnynz);
}

DistanceGrid* DistanceGrid::load(const std::string& filename, int nx, int ny, int nz, Coord pmin, Coord pmax)
{
    if (filename.length()>4 && filename.substr(filename.length()-4) == ".raw")
    {
        DistanceGrid* grid = new DistanceGrid(nx, ny, nz, pmin, pmax);
        std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
        in.read((char*)&(grid->dists[0]), grid->nxnynz*sizeof(Real));
    }
    else if (filename.length()>4 && filename.substr(filename.length()-4) == ".obj")
    {
        sofa::helper::io::Mesh* mesh = sofa::helper::io::Mesh::Create(filename);
        const sofa::helper::vector<Vector3> & vertices = mesh->getVertices();

        if (pmin[0]==pmax[0])
        {
            std::cout << "Computing bbox."<<std::endl;
            if (!vertices.empty())
            {
                pmin = vertices[0];
                pmax = pmin;
                for(unsigned int i=1; i<vertices.size(); i++)
                {
                    for (int c=0; c<3; c++)
                        if (vertices[i][c] < pmin[c]) pmin[c] = vertices[i][c];
                        else if (vertices[i][c] > pmax[c]) pmax[c] = vertices[i][c];
                }
            }
            std::cout << "bbox = "<<pmin<<" "<<pmax<<std::endl;
        }
        std::cout << "Creating distance grid."<<std::endl;
        DistanceGrid* grid = new DistanceGrid(nx, ny, nz, pmin, pmax);
        std::cout << "Copying mesh vertices."<<std::endl;
        grid->meshPts.resize(vertices.size());
        for(unsigned int i=0; i<vertices.size(); i++)
            grid->meshPts[i] = vertices[i];
        std::cout << "Computing distance field."<<std::endl;
        grid->calcDistance(mesh);
        std::cout << "Distance grid creation DONE."<<std::endl;
        delete mesh;
        return grid;
    }
    else
    {
        std::cerr << "Unknown extension: "<<filename<<std::endl;
    }


    return NULL;
}

bool DistanceGrid::save(const std::string& filename)
{
    /// !!!TODO!!! ///
    if (filename.length()>4 && filename.substr(filename.length()-4) == ".raw")
    {
        std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
        out.write((char*)&(dists[0]), nxnynz*sizeof(Real));
    }
    else
    {
        std::cerr << " DistanceGrid::save(): Unsupported extension: "<<filename<<std::endl;
        return false;
    }
    return true;
}


template<int U, int V>
bool pointInTriangle(const DistanceGrid::Coord& p, const DistanceGrid::Coord& p0, const DistanceGrid::Coord& p1, const DistanceGrid::Coord& p2)
{
    DistanceGrid::Real u0 = p [U] - p0[U], v0 = p [V] - p0[V];
    DistanceGrid::Real u1 = p1[U] - p0[U], v1 = p1[V] - p0[V];
    DistanceGrid::Real u2 = p2[U] - p0[U], v2 = p2[V] - p0[V];
    DistanceGrid::Real alpha, beta;
    //return true;
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

/// Compute distance field from given mesh
void DistanceGrid::calcDistance(sofa::helper::io::Mesh* mesh)
{
    fmm_status.resize(nxnynz);
    fmm_heap.resize(nxnynz);
    fmm_heap_size = 0;
    std::cout << "FMM: Init."<<std::endl;

    std::fill(fmm_status.begin(), fmm_status.end(), FMM_FAR);
    std::fill(dists.begin(), dists.end(), maxDist());

    const sofa::helper::vector<Vector3> & vertices = mesh->getVertices();
    const sofa::helper::vector<sofa::helper::vector<sofa::helper::vector<int> > > & facets = mesh->getFacets();

    // Initialize distance of edges crossing triangles
    std::cout << "FMM: Initialize distance of edges crossing triangles."<<std::endl;

    for (unsigned int i=0; i<facets.size(); i++)
    {
        const sofa::helper::vector<int>& pts = facets[i][0];
        const int pt0 = 0;
        const Coord& p0 = vertices[pts[pt0]];
        for (unsigned int pt2=2; pt2<pts.size(); pt2++)
        {
            const int pt1 = pt2-1;
            const Coord& p1 = vertices[pts[pt1]];
            const Coord& p2 = vertices[pts[pt2]];
            Coord bbmin = p0, bbmax = p0;
            for (int c=0; c<3; c++)
                if (p1[c] < bbmin[c]) bbmin[c] = p1[c];
                else if (p1[c] > bbmax[c]) bbmax[c] = p1[c];
            for (int c=0; c<3; c++)
                if (p2[c] < bbmin[c]) bbmin[c] = p2[c];
                else if (p2[c] > bbmax[c]) bbmax[c] = p2[c];

            Coord normal = (p1-p0).cross(p2-p0);
            normal.normalize();
            Real d = -(p0*normal);

            int ix0 = ix(bbmin)-1; if (ix0 < 0) ix0 = 0;
            int iy0 = iy(bbmin)-1; if (iy0 < 0) iy0 = 0;
            int iz0 = iz(bbmin)-1; if (iz0 < 0) iz0 = 0;
            int ix1 = ix(bbmax)+2; if (ix1 >= nx) ix1 = nx-1;
            int iy1 = iy(bbmax)+2; if (iy1 >= ny) iy1 = ny-1;
            int iz1 = iz(bbmax)+2; if (iz1 >= nz) iz1 = nz-1;
            for (int z=iz0; z<iz1; z++)
                for (int y=iy0; y<iy1; y++)
                    for (int x=ix0; x<ix1; x++)
                    {
                        Coord pos = coord(x,y,z);
                        int ind = index(x,y,z);
                        Real dist = pos*normal + d;
                        //if (rabs(dist) > cellWidth) continue; // no edge from this point can cross the plane

                        // X edge
                        if (rabs(normal[0]) > 1e-6)
                        {
                            Real dist1 = -dist / normal[0];
                            int ind2 = ind+1;
                            if (dist1 >= -0.01*cellWidth[0] && dist1 <= 1.01*cellWidth[0])
                            {
                                // edge crossed plane
                                if (pointInTriangle<1,2>(pos,p0,p1,p2))
                                {
                                    // edge crossed triangle
                                    Real dist2 = cellWidth[0] - dist1;
                                    if (normal[0]<0)
                                    {
                                        // p1 is in outside, p2 inside
                                        if (dist1 < (dists[ind]))
                                        {
                                            // nearest triangle
                                            dists[ind] = dist1;
                                            fmm_status[ind] = FMM_KNOWN_OUT;
                                        }
                                        if (dist2 < (dists[ind2]))
                                        {
                                            // nearest triangle
                                            dists[ind2] = dist2;
                                            fmm_status[ind2] = FMM_KNOWN_IN;
                                        }
                                    }
                                    else
                                    {
                                        // p1 is in inside, p2 outside
                                        if (dist1 < (dists[ind]))
                                        {
                                            // nearest triangle
                                            dists[ind] = dist1;
                                            fmm_status[ind] = FMM_KNOWN_IN;
                                        }
                                        if (dist2 < (dists[ind2]))
                                        {
                                            // nearest triangle
                                            dists[ind2] = dist2;
                                            fmm_status[ind2] = FMM_KNOWN_OUT;
                                        }
                                    }
                                }
                            }
                        }

                        // Y edge
                        if (rabs(normal[1]) > 1e-6)
                        {
                            Real dist1 = -dist / normal[1];
                            int ind2 = ind+nx;
                            if (dist1 >= -0.01*cellWidth[1] && dist1 <= 1.01*cellWidth[1])
                            {
                                // edge crossed plane
                                if (pointInTriangle<2,0>(pos,p0,p1,p2))
                                {
                                    // edge crossed triangle
                                    Real dist2 = cellWidth[1] - dist1;
                                    if (normal[1]<0)
                                    {
                                        // p1 is in outside, p2 inside
                                        if (dist1 < (dists[ind]))
                                        {
                                            // nearest triangle
                                            dists[ind] = dist1;
                                            fmm_status[ind] = FMM_KNOWN_OUT;
                                        }
                                        if (dist2 < (dists[ind2]))
                                        {
                                            // nearest triangle
                                            dists[ind2] = dist2;
                                            fmm_status[ind2] = FMM_KNOWN_IN;
                                        }
                                    }
                                    else
                                    {
                                        // p1 is in inside, p2 outside
                                        if (dist1 < (dists[ind]))
                                        {
                                            // nearest triangle
                                            dists[ind] = dist1;
                                            fmm_status[ind] = FMM_KNOWN_IN;
                                        }
                                        if (dist2 < (dists[ind2]))
                                        {
                                            // nearest triangle
                                            dists[ind2] = dist2;
                                            fmm_status[ind2] = FMM_KNOWN_OUT;
                                        }
                                    }
                                }
                            }
                        }

                        // Z edge
                        if (rabs(normal[2]) > 1e-6)
                        {
                            Real dist1 = -dist / normal[2];
                            int ind2 = ind+nxny;
                            if (dist1 >= -0.01*cellWidth[2] && dist1 <= 1.01*cellWidth[2])
                            {
                                // edge crossed plane
                                if (pointInTriangle<0,1>(pos,p0,p1,p2))
                                {
                                    // edge crossed triangle
                                    Real dist2 = cellWidth[2] - dist1;
                                    if (normal[2]<0)
                                    {
                                        // p1 is in outside, p2 inside
                                        if (dist1 < (dists[ind]))
                                        {
                                            // nearest triangle
                                            dists[ind] = dist1;
                                            fmm_status[ind] = FMM_KNOWN_OUT;
                                        }
                                        if (dist2 < (dists[ind2]))
                                        {
                                            // nearest triangle
                                            dists[ind2] = dist2;
                                            fmm_status[ind2] = FMM_KNOWN_IN;
                                        }
                                    }
                                    else
                                    {
                                        // p1 is in inside, p2 outside
                                        if (dist1 < (dists[ind]))
                                        {
                                            // nearest triangle
                                            dists[ind] = dist1;
                                            fmm_status[ind] = FMM_KNOWN_IN;
                                        }
                                        if (dist2 < (dists[ind2]))
                                        {
                                            // nearest triangle
                                            dists[ind2] = dist2;
                                            fmm_status[ind2] = FMM_KNOWN_OUT;
                                        }
                                    }
                                }
                            }
                        }
                    }
        }
    }

    // Update known points neighbors
    std::cout << "FMM: Update known points neighbors."<<std::endl;

    for (int z=0, ind=0; z<nz; z++)
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx; x++, ind++)
            {
                if (fmm_status[ind] < FMM_FAR)
                {
                    int ind2;
                    Real dist1 = dists[ind];
                    Real dist2 = dist1+cellWidth[0];
                    // X-1
                    if (x>0)
                    {
                        ind2 = ind-1;
                        if (x>0 && fmm_status[ind2] >= FMM_FAR && (dists[ind2]) > dist2)
                        {
                            dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                    // X+1
                    if (x<nx-1)
                    {
                        ind2 = ind+1;
                        if (x>0 && fmm_status[ind2] >= FMM_FAR && (dists[ind2]) > dist2)
                        {
                            dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                    dist2 = dist1+cellWidth[1];
                    // Y-1
                    if (y>0)
                    {
                        ind2 = ind-nx;
                        if (x>0 && fmm_status[ind2] >= FMM_FAR && (dists[ind2]) > dist2)
                        {
                            dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                    // Y+1
                    if (y<ny-1)
                    {
                        ind2 = ind+nx;
                        if (x>0 && fmm_status[ind2] >= FMM_FAR && (dists[ind2]) > dist2)
                        {
                            dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                    dist2 = dist1+cellWidth[2];
                    // Z-1
                    if (z>0)
                    {
                        ind2 = ind-nxny;
                        if (x>0 && fmm_status[ind2] >= FMM_FAR && (dists[ind2]) > dist2)
                        {
                            dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                    // Z+1
                    if (z<nz-1)
                    {
                        ind2 = ind+nxny;
                        if (x>0 && fmm_status[ind2] >= FMM_FAR && (dists[ind2]) > dist2)
                        {
                            dists[ind2] = dist2;
                            fmm_push(ind2);
                        }
                    }
                }
            }

    // March through the heap
    std::cout << "FMM: March through the heap." << std::endl;
    while (fmm_heap_size > 0)
    {
        int ind = fmm_pop();
        int nbin = 0, nbout = 0;
        int x = ind%nx;
        int y = (ind/nx)%ny;
        int z = ind/nxny;

        int ind2;
        Real dist1 = dists[ind];
        Real dist2 = dist1+cellWidth[0];
        // X-1
        if (x>0)
        {
            ind2 = ind-1;
            if (fmm_status[ind2] < FMM_FAR)
            {
                if (fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((dists[ind2]) > dist2)
            {
                dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        // X+1
        if (x<nx-1)
        {
            ind2 = ind+1;
            if (fmm_status[ind2] < FMM_FAR)
            {
                if (fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((dists[ind2]) > dist2)
            {
                dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        dist2 = dist1+cellWidth[1];
        // Y-1
        if (y>0)
        {
            ind2 = ind-nx;
            if (fmm_status[ind2] < FMM_FAR)
            {
                if (fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((dists[ind2]) > dist2)
            {
                dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        // Y+1
        if (y<ny-1)
        {
            ind2 = ind+nx;
            if (fmm_status[ind2] < FMM_FAR)
            {
                if (fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((dists[ind2]) > dist2)
            {
                dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        dist2 = dist1+cellWidth[2];
        // Z-1
        if (z>0)
        {
            ind2 = ind-nxny;
            if (fmm_status[ind2] < FMM_FAR)
            {
                if (fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((dists[ind2]) > dist2)
            {
                dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        // Z+1
        if (z<nz-1)
        {
            ind2 = ind+nxny;
            if (fmm_status[ind2] < FMM_FAR)
            {
                if (fmm_status[ind2] == FMM_KNOWN_IN) ++nbin; else ++nbout;
            }
            else if ((dists[ind2]) > dist2)
            {
                dists[ind2] = dist2;
                fmm_push(ind2); // create or update the corresponding entry in the heap
            }
        }
        if (nbin && nbout)
        {
            std::cerr << "FMM WARNING: in/out conflict at cell "<<x<<" "<<y<<" "<<z<<" ( "<<nbin<<" in, "<<nbout<<" out), dist = "<<dists[ind]<<std::endl;
        }
        if (nbin > nbout)
            fmm_status[ind] = FMM_KNOWN_IN;
        else
            fmm_status[ind] = FMM_KNOWN_OUT;
    }

    // Finalize distances
    std::cout << "FMM: Finalize distances."<<std::endl;
    int nbin = 0;
    for (int z=0, ind=0; z<nz; z++)
        for (int y=0; y<ny; y++)
            for (int x=0; x<nx; x++, ind++)
            {
                if (fmm_status[ind] == FMM_KNOWN_IN)
                {
                    dists[ind] = -dists[ind];
                    ++nbin;
                }
                else if (fmm_status[ind] != FMM_KNOWN_OUT)
                {
                    //std::cerr << "FMM ERROR: cell "<<x<<" "<<y<<" "<<z<<" not computed. dist="<<dists[ind]<<std::endl;
                }
            }
    std::cout << "FMM: DONE. "<< nbin << " points inside ( " << (nbin*100)/size() <<" % )" << std::endl;
}

inline void DistanceGrid::fmm_swap(int entry1, int entry2)
{
    int ind1 = fmm_heap[entry1];
    int ind2 = fmm_heap[entry2];
    fmm_heap[entry1] = ind2;
    fmm_heap[entry2] = ind1;
    fmm_status[ind1] = entry2 + FMM_FRONT0;
    fmm_status[ind2] = entry1 + FMM_FRONT0;
}

int DistanceGrid::fmm_pop()
{
    int res = fmm_heap[0];
#ifdef FMM_VERBOSE
    std::cout << "fmm_pop -> <"<<(res%nx)<<','<<((res/nx)%ny)<<','<<(res/nxny)<<">="<<dists[res]<<std::endl;
#endif
    --fmm_heap_size;
    if (fmm_heap_size>0)
    {
        fmm_swap(0, fmm_heap_size);
        int i=0;
        Real phi = (dists[fmm_heap[i]]);
        while (i*2+1 < fmm_heap_size)
        {
            Real phi1 = (dists[fmm_heap[i*2+1]]);
            if (i*2+2 < fmm_heap_size)
            {
                Real phi2 = (dists[fmm_heap[i*2+2]]);
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
#ifdef FMM_VERBOSE
    std::cout << "fmm_heap = [";
    for (int i=0; i<fmm_heap_size; i++)
        std::cout << " <"<<(fmm_heap[i]%nx)<<','<<((fmm_heap[i]/nx)%ny)<<','<<(fmm_heap[i]/nxny)<<">="<<dists[fmm_heap[i]];
    std::cout << std::endl;
#endif
    //fmm_status[res] = FMM_KNOWN;
    return res;
}

void DistanceGrid::fmm_push(int index)
{
    Real phi = (dists[index]);
    int i;
    if (fmm_status[index] >= FMM_FRONT0)
    {
        i = fmm_status[index] - FMM_FRONT0;
#ifdef FMM_VERBOSE
        std::cout << "fmm update <"<<(index%nx)<<','<<((index/nx)%ny)<<','<<(index/nxny)<<">="<<dists[index]<<" from entry "<<i<<std::endl;
#endif
        while (i>0 && phi < (dists[fmm_heap[(i-1)/2]]))
        {
            fmm_swap(i,(i-1)/2);
            i = (i-1)/2;
        }
        while (i*2+1 < fmm_heap_size)
        {
            Real phi1 = (dists[fmm_heap[i*2+1]]);
            if (i*2+2 < fmm_heap_size)
            {
                Real phi2 = (dists[fmm_heap[i*2+2]]);
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
#ifdef FMM_VERBOSE
        std::cout << "fmm push <"<<(index%nx)<<','<<((index/nx)%ny)<<','<<(index/nxny)<<">="<<dists[index]<<std::endl;
#endif
        i = fmm_heap_size;
        ++fmm_heap_size;
        fmm_heap[i] = index;
        fmm_status[index] = i;
        while (i>0 && phi < (dists[fmm_heap[(i-1)/2]]))
        {
            fmm_swap(i,(i-1)/2);
            i = (i-1)/2;
        }
    }
#ifdef FMM_VERBOSE
    std::cout << "fmm_heap = [";
    for (int i=0; i<fmm_heap_size; i++)
        std::cout << " <"<<(fmm_heap[i]%nx)<<','<<((fmm_heap[i]/nx)%ny)<<','<<(fmm_heap[i]/nxny)<<">="<<dists[fmm_heap[i]];
    std::cout << std::endl;
#endif
}

DistanceGrid* DistanceGrid::loadShared(const std::string& filename, int nx, int ny, int nz, Coord pmin, Coord pmax)
{
    DistanceGridParams params;
    params.filename = filename;
    params.nx = nx;
    params.ny = ny;
    params.nz = nz;
    params.pmin = pmin;
    params.pmax = pmax;
    std::map<DistanceGridParams, DistanceGrid*>& shared = getShared();
    std::map<DistanceGridParams, DistanceGrid*>::iterator it = shared.find(params);
    if (it != shared.end())
        return it->second;
    else
    {
        return shared[params] = load(filename, nx, ny, nz, pmin, pmax);
    }
}

std::map<DistanceGrid::DistanceGridParams, DistanceGrid*>& DistanceGrid::getShared()
{
    static std::map<DistanceGridParams, DistanceGrid*> instance;
    return instance;
}

} // namespace collision

} // namespace component

} // namespace sofa
