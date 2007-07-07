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
    if (rigid!=NULL)
    {
        glPushMatrix();
        float m[16];
        (*rigid->getX())[index].writeOpenGlMatrix( m );
        glMultMatrixf(m);
    }

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
    for(unsigned int i=0; i<8; i++)
        corners[i] = grid->getBBCorner(i);
    //glEnable(GL_BLEND);
    //glDepthMask(0);
    if (isStatic())
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

    const float maxdist = (grid->getBBMax()-grid->getBBMin()).norm()*0.1f;

    /*
    glBegin(GL_POINTS);
    {
    for (int z=0, ind=0; z<grid->getNz(); z++)
    for (int y=0; y<grid->getNy(); y++)
    for (int x=0; x<grid->getNx(); x++, ind++)
    {
        DistanceGrid::Coord p = grid->coord(x,y,z);
        DistanceGrid::Real d = (*grid)[ind];
        if (rabs(d) > maxdist) continue;
            d /= maxdist;
        if (d<0)
    	glColor3d(1+d*0.25, 0, 1+d);
        else
    	glColor3d(0, 1-d*0.25, 1-d);
        glVertex3fv(p.ptr());
    }
    }
    */
    glColor3d(1, 1 ,1);
    glBegin(GL_POINTS);
    for (unsigned int i=0; i<grid->meshPts.size(); i++)
    {
        DistanceGrid::Coord p = grid->meshPts[i];
        glVertex3fv(p.ptr());
    }
    glEnd();
    glBegin(GL_LINES);
    for (unsigned int i=0; i<grid->meshPts.size(); i++)
    {
        DistanceGrid::Coord p = grid->meshPts[i];
        glColor3d(1, 1 ,1);
        DistanceGrid::Coord grad = grid->grad(p);
        grad.normalize();
        for (int j = -2; j <= 2; j++)
        {
            DistanceGrid::Coord p2 = p + grad * (j*maxdist/2);
            DistanceGrid::Real d = grid->eval(p2);
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

    if (rigid!=NULL)
    {
        glPopMatrix();
    }
}

DistanceGridCollisionModel::DistanceGridCollisionModel()
    : filename( dataField( &filename, "filename", "load distance grid from specified file"))
    , scale( dataField( &scale, 1.0, "scale", "scaling factor for input file"))
    , box( dataField( &box, "box", "Field bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , nx( dataField( &nx, 64, "nx", "number of values on X axis") )
    , ny( dataField( &ny, 64, "ny", "number of values on Y axis") )
    , nz( dataField( &nz, 64, "nz", "number of values on Z axis") )
    , dumpfilename( dataField( &dumpfilename, "dumpfilename","write distance grid to specified file"))
    , usePoints( dataField( &usePoints, true, "usePoints", "use mesh vertices for collision detection"))
{
    ffd = NULL;
    ffdGrid = NULL;
    rigid = NULL;
}

DistanceGridCollisionModel::~DistanceGridCollisionModel()
{
    for (unsigned int i=0; i<elems.size(); i++)
        if (elems[i]!=NULL) elems[i]->release();
}

void DistanceGridCollisionModel::init()
{
    std::cout << "> DistanceGridCollisionModel::init()"<<std::endl;
    this->core::CollisionModel::init();
    rigid = dynamic_cast< core::componentmodel::behavior::MechanicalState<RigidTypes>* > (getContext()->getMechanicalState());
    ffd = dynamic_cast< core::componentmodel::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());
    ffdGrid = dynamic_cast< topology::RegularGridTopology* > (getContext()->getTopology());

    DistanceGrid* grid = NULL;
    if (filename.getValue().empty())
    {
        std::cerr << "ERROR: DistanceGridCollisionModel requires an input filename.\n";
        return;
    }
    std::cout << "DistanceGridCollisionModel: creating "<<nx.getValue()<<"x"<<ny.getValue()<<"x"<<nz.getValue()<<" DistanceGrid from file "<<filename.getValue();
    if (scale.getValue()!=1.0) std::cout<<" scale="<<scale.getValue();
    if (box.getValue()[0][0]<box.getValue()[1][0]) std::cout<<" bbox=<"<<box.getValue()[0]<<">-<"<<box.getValue()[0]<<">";
    std::cout << std::endl;
    grid = DistanceGrid::loadShared(filename.getValue(), scale.getValue(), nx.getValue(),ny.getValue(),nz.getValue(),box.getValue()[0],box.getValue()[1]);

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
}

void DistanceGridCollisionModel::draw()
{
    if (!isActive()) return;
    if (getContext()->getShowCollisionModels())
    {
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
        glPointSize(1);
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (getPrevious()!=NULL && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

/// Create or update the bounding volume hierarchy.
void DistanceGridCollisionModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = this->createPrevious<CubeModel>();

    if (isStatic() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    //if (filename.getValue().empty())
    //    updateGrid();

    bool xform = false;
    Mat3x3d rotation;
    Vec3d translation;

    if (rigid)
    {
        xform = true;
        translation = (*rigid->getX())[0].getCenter();
        (*rigid->getX())[0].getOrientation().toMatrix(rotation);
    }
    else rotation.identity();

    cubeModel->resize(size);
    for (int i=0; i<size; i++)
    {
        //static_cast<DistanceGridCollisionElement*>(elems[i])->recalcBBox();
        Vector3 emin, emax;
        if (xform)
        {
            Vector3 corner = translation + rotation * elems[i]->getBBCorner(0);
            emin = corner;
            emax = emin;
            for (int j=1; j<8; j++)
            {
                corner = translation + rotation * elems[i]->getBBCorner(j);
                for(int c=0; c<3; c++)
                    if (corner[c] < emin[c]) emin[c] = corner[c];
                    else if (corner[c] > emax[c]) emax[c] = corner[c];
            }
        }
        else
        {
            emin = elems[i]->getBBMin();
            emax = elems[i]->getBBMax();
        }
        cubeModel->setParentOf(i, emin, emax); // define the bounding box of the current element
    }
    cubeModel->computeBoundingTree(maxDepth);
}

DistanceGrid::DistanceGrid(int nx, int ny, int nz, Coord pmin, Coord pmax)
    : nbRef(1), nx(nx), ny(ny), nz(nz), nxny(nx*ny), nxnynz(nx*ny*nz)
    , pmin(pmin), pmax(pmax)
    , cellWidth   ((pmax[0]-pmin[0])/(nx-1), (pmax[1]-pmin[1])/(ny-1),(pmax[2]-pmin[2])/(nz-1))
    , invCellWidth((nx-1)/(pmax[0]-pmin[0]), (ny-1)/(pmax[1]-pmin[1]),(nz-1)/(pmax[2]-pmin[2]))
    , cubeDim(0)
{
    dists.resize(nxnynz);
}

/// Add one reference to this grid. Note that loadShared already does this.
DistanceGrid* DistanceGrid::addRef()
{
    ++nbRef;
    return this;
}

/// Release one reference, deleting this grid if this is the last
bool DistanceGrid::release()
{
    if (--nbRef != 0)
        return false;
    delete this;
    return true;
}

DistanceGrid* DistanceGrid::load(const std::string& filename, double scale, int nx, int ny, int nz, Coord pmin, Coord pmax)
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
        DistanceGrid* grid = new DistanceGrid(nx, ny, nz, pmin, pmax);
        grid->calcCubeDistance(dim, np);
        std::cout << "Distance grid creation DONE."<<std::endl;
        return grid;
    }
    else if (filename.length()>4 && filename.substr(filename.length()-4) == ".raw")
    {
        DistanceGrid* grid = new DistanceGrid(nx, ny, nz, pmin, pmax);
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
        DistanceGrid* grid = new DistanceGrid(nx, ny, nz, pmin, pmax);
        std::cout << "Copying "<<vertices.size()<<" mesh vertices."<<std::endl;
        grid->meshPts.resize(vertices.size());
        for(unsigned int i=0; i<vertices.size(); i++)
            grid->meshPts[i] = vertices[i]*scale;
        std::cout << "Computing distance field."<<std::endl;
        grid->calcDistance(mesh, scale);
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

void DistanceGrid::computeBBox()
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
void DistanceGrid::calcCubeDistance(Real dim, int np)
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
void DistanceGrid::calcDistance(sofa::helper::io::Mesh* mesh, double scale)
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
            Real d = -(p0*normal);
            int nedges = 0;
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
                                    ++nedges;
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
                                    ++nedges;
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
                                    ++nedges;
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
            std::cout << "Triangle "<<pts[pt0]<<"-"<<pts[pt1]<<"-"<<pts[pt2]<<" crossed "<<nedges<<" edges within <"<<ix0<<" "<<iy0<<" "<<iz0<<">-<"<<ix1-1<<" "<<iy1-1<<" "<<iz1-1<<" "<<">."<<std::endl;
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

DistanceGrid* DistanceGrid::loadShared(const std::string& filename, double scale, int nx, int ny, int nz, Coord pmin, Coord pmax)
{
    DistanceGridParams params;
    params.filename = filename;
    params.scale = scale;
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
        return shared[params] = load(filename, scale, nx, ny, nz, pmin, pmax);
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
