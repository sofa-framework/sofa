/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#define SOFA_COMPONENT_COLLISION_DISTANCEGRIDCOLLISIONMODEL_CPP
#include <fstream>
#include <sstream>
#include <SofaDistanceGrid/config.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/Factory.inl>
#include <sofa/component/collision/geometry/CubeCollisionModel.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.inl>
#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>

#include "DistanceGridCollisionModel.h"

#if SOFADISTANCEGRID_HAVE_SOFA_GL == 1
#include <sofa/gl/gl.h>
#include <sofa/gl/template.h>
#endif // SOFADISTANCEGRID_HAVE_SOFA_GL == 1

namespace sofa
{

namespace component
{

namespace collision
{

void registerRigidDistanceGridCollisionModel(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Grid-based distance field.")
    .add< RigidDistanceGridCollisionModel >());
}
      
void registerFFDDistanceGridCollisionModel(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Grid-based deformable distance field.")
    .add< FFDDistanceGridCollisionModel >());
}

using namespace sofa::type;
using namespace defaulttype;
using namespace sofa::component::collision::geometry;
using namespace sofa::component::collision::response::mapper;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

RigidDistanceGridCollisionModel::RigidDistanceGridCollisionModel()
    : modified(true)
    , fileRigidDistanceGrid( initData( &fileRigidDistanceGrid, "filename", "Load distance grid from specified file"))
    , scale( initData( &scale, 1.0, "scale", "scaling factor for input file"))
    , translation( initData( &translation, "translation", "translation to apply to input file"))
    , rotation( initData( &rotation, "rotation", "rotation to apply to input file"))
    , sampling( initData( &sampling, 0.0, "sampling", "if not zero: sample the surface with points approximately separated by the given sampling distance (expressed in voxels if the value is negative)"))
    , box( initData( &box, "box", "Field bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , nx( initData( &nx, 64, "nx", "number of values on X axis") )
    , ny( initData( &ny, 64, "ny", "number of values on Y axis") )
    , nz( initData( &nz, 64, "nz", "number of values on Z axis") )
    , dumpfilename( initData( &dumpfilename, "dumpfilename","write distance grid to specified file"))
    , usePoints( initData( &usePoints, true, "usePoints", "use mesh vertices for collision detection"))
    , flipNormals( initData( &flipNormals, false, "flipNormals", "reverse surface direction, i.e. points are considered in collision if they move outside of the object instead of inside"))
    , showMeshPoints( initData( &showMeshPoints, true, "showMeshPoints", "Enable rendering of mesh points"))
    , showGridPoints( initData( &showGridPoints, false, "showGridPoints", "Enable rendering of grid points"))
    , showMinDist ( initData( &showMinDist, 0.0, "showMinDist", "Min distance to render gradients"))
    , showMaxDist ( initData( &showMaxDist, 0.0, "showMaxDist", "Max distance to render gradients"))
{
    addAlias(&fileRigidDistanceGrid,"fileRigidDistanceGrid");
}

RigidDistanceGridCollisionModel::~RigidDistanceGridCollisionModel()
{}

void RigidDistanceGridCollisionModel::init()
{
    Inherit1::init();
    Inherit2::init();

    if (!this->mstate)
    {
        msg_error() << "MechanicalState is empty. Either no MechanicalState object was found in"
            " current context or the one provided is not using a Rigid type template.";
        return;
    }

    std::shared_ptr<DistanceGrid> grid;
    if (fileRigidDistanceGrid.getValue().empty())
    {
        if (elems.size() == 0 || elems[0].grid == nullptr)
            msg_error() << "An input filename is required.";
        // else the grid has already been set
        return;
    }
    msg_info() << "Creating "<<nx.getValue()<<"x"<<ny.getValue()<<"x"<<nz.getValue()<<" DistanceGrid from file "<<fileRigidDistanceGrid.getValue();

    if (scale.getValue()!=1.0) msg_info()<<" scale="<<scale.getValue();
    if (sampling.getValue()!=0.0) msg_info()<<" sampling="<<sampling.getValue();
    if (box.getValue()[0][0]<box.getValue()[1][0]) msg_info()<<" bbox=<"<<box.getValue()[0]<<">-<"<<box.getValue()[0]<<">";

    grid = DistanceGrid::loadShared(fileRigidDistanceGrid.getFullPath(), scale.getValue(), sampling.getValue(), nx.getValue(),ny.getValue(),nz.getValue(),box.getValue()[0],box.getValue()[1]);
    if (grid->getNx() != this->nx.getValue())
        this->nx.setValue(grid->getNx());
    if (grid->getNy() != this->ny.getValue())
        this->ny.setValue(grid->getNy());
    if (grid->getNz() != this->nz.getValue())
        this->nz.setValue(grid->getNz());
    resize(1);
    elems[0].grid = grid;
    if (grid && !dumpfilename.getValue().empty())
    {
        msg_info() << "Dump grid to "<<dumpfilename.getValue();
        grid->save(dumpfilename.getFullPath());
    }
    updateState();
    msg_info() << "Initialisation done.";
}

void RigidDistanceGridCollisionModel::resize(sofa::Size s)
{
    this->core::CollisionModel::resize(s);
    elems.resize(s);
}

void RigidDistanceGridCollisionModel::setNewState(sofa::Index index, double dt, const std::shared_ptr<DistanceGrid> grid, const Matrix3& rotation, const Vec3& translation)
{
    elems[index].prevGrid = elems[index].grid;
    elems[index].grid = grid;
    elems[index].prevRotation = elems[index].rotation;
    elems[index].rotation = rotation;
    elems[index].prevTranslation = elems[index].translation;
    elems[index].translation = translation;
    if (!elems[index].isTransformed)
    {
        Matrix3 I; I.identity();
        if (!(rotation == I) || !(translation == Vec3()))
            elems[index].isTransformed = true;
    }
    elems[index].prevDt = dt;
    modified = true;
}

/// Update transformation matrices from current rigid state
void RigidDistanceGridCollisionModel::updateState()
{
    const Vec3& initTranslation = this->translation.getValue();
    const Vec3& initRotation = this->rotation.getValue();
    bool useInitTranslation = (initTranslation != DistanceGrid::Coord());
    bool useInitRotation = (initRotation != Vec3(0,0,0));

    for (sofa::Size i=0; i<size; i++)
    {
        if (this->mstate)
        {
            const RigidTypes::Coord& xform =(this->mstate->read(core::vec_id::read_access::position)->getValue())[i];
            elems[i].translation = xform.getCenter();
            xform.getOrientation().toMatrix(elems[i].rotation);
            if (useInitRotation)
                elems[i].rotation = getInitRotation();
            if (useInitTranslation)
                elems[i].translation += elems[i].rotation * initTranslation;
            elems[i].isTransformed = true;
        }
        else
        {
            if(useInitRotation)
            {
                elems[i].rotation = getInitRotation();
                elems[i].isTransformed = true;
            }
            if(useInitTranslation)
            {
                elems[i].translation = initTranslation;
            }

        }
    }
}

/// Create or update the bounding volume hierarchy.
void RigidDistanceGridCollisionModel::computeBoundingTree(int maxDepth)
{
    CubeCollisionModel* cubeModel = this->createPrevious<CubeCollisionModel>();

    if (!modified && !isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    updateGrid();
    updateState();

    const bool flipped = isFlipped();
    cubeModel->resize(size);
    for (sofa::Size i=0; i<size; i++)
    {
        Vec3 emin, emax;
        if (elems[i].isTransformed)
        {
            for (int j=0; j<8; j++)
            {
                Vec3 corner = elems[i].translation + elems[i].rotation * (flipped ? elems[i].grid->getCorner(j) : elems[i].grid->getBBCorner(j));
                if (j == 0)
                {
                    emin = corner;
                    emax = emin;
                }
                else
                {
                    for(int c=0; c<3; c++)
                        if (corner[c] < emin[c]) emin[c] = corner[c];
                        else if (corner[c] > emax[c]) emax[c] = corner[c];
                }
            }
        }
        else
        {
            emin = flipped ? elems[i].grid->getPMin() : elems[i].grid->getBBMin();
            emax = flipped ? elems[i].grid->getPMax() : elems[i].grid->getBBMax();
        }
        cubeModel->setParentOf(i, emin, emax); // define the bounding box of the current element
    }
    cubeModel->computeBoundingTree(maxDepth);
    modified = false;
}

void RigidDistanceGridCollisionModel::updateGrid()
{
}

void RigidDistanceGridCollisionModel::drawCollisionModel(const core::visual::VisualParams* vparams)
{
#if SOFADISTANCEGRID_HAVE_SOFA_GL == 1
    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDisable(GL_LIGHTING);
    glColor4fv(getColor4f());
    glPointSize(3);
    for (unsigned int i = 0; i < elems.size(); i++)
    {
        draw(vparams, i);
    }
    glPointSize(1);
    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif // SOFADISTANCEGRID_HAVE_SOFA_GL == 1
}

void RigidDistanceGridCollisionModel::draw(const core::visual::VisualParams* ,sofa::Index index)
{
#if SOFADISTANCEGRID_HAVE_SOFA_GL == 1
    const bool flipped = isFlipped();

    if (elems[index].isTransformed)
    {
        glPushMatrix();
        Matrix4 m;
        m.identity();
        m = elems[index].rotation;
        m.transpose();
        m[3] = Vec4(elems[index].translation,1.0);

        sofa::gl::glMultMatrix(m.ptr());
    }

    const std::shared_ptr<DistanceGrid> grid = getGrid(index);
    DistanceGrid::Coord corners[8];
    for(unsigned int i=0; i<8; i++)
        corners[i] = grid->getCorner(i);
    if (!isSimulated())
        glColor4f(0.25f, 0.25f, 0.25f, 0.1f);
    else
        glColor4f(0.5f, 0.5f, 0.5f, 0.1f);
    glBegin(GL_LINES);
    {
        sofa::gl::glVertexT(corners[0]); sofa::gl::glVertexT(corners[4]);
        sofa::gl::glVertexT(corners[1]); sofa::gl::glVertexT(corners[5]);
        sofa::gl::glVertexT(corners[2]); sofa::gl::glVertexT(corners[6]);
        sofa::gl::glVertexT(corners[3]); sofa::gl::glVertexT(corners[7]);
        sofa::gl::glVertexT(corners[0]); sofa::gl::glVertexT(corners[2]);
        sofa::gl::glVertexT(corners[1]); sofa::gl::glVertexT(corners[3]);
        sofa::gl::glVertexT(corners[4]); sofa::gl::glVertexT(corners[6]);
        sofa::gl::glVertexT(corners[5]); sofa::gl::glVertexT(corners[7]);
        sofa::gl::glVertexT(corners[0]); sofa::gl::glVertexT(corners[1]);
        sofa::gl::glVertexT(corners[2]); sofa::gl::glVertexT(corners[3]);
        sofa::gl::glVertexT(corners[4]); sofa::gl::glVertexT(corners[5]);
        sofa::gl::glVertexT(corners[6]); sofa::gl::glVertexT(corners[7]);
    }
    glEnd();
    glDisable(GL_BLEND);
    glDepthMask(1);
    for(unsigned int i=0; i<8; i++)
        corners[i] = grid->getBBCorner(i);

    if (!isSimulated())
        glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
    else
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_LINES);
    {
        sofa::gl::glVertexT(corners[0]); sofa::gl::glVertexT(corners[4]);
        sofa::gl::glVertexT(corners[1]); sofa::gl::glVertexT(corners[5]);
        sofa::gl::glVertexT(corners[2]); sofa::gl::glVertexT(corners[6]);
        sofa::gl::glVertexT(corners[3]); sofa::gl::glVertexT(corners[7]);
        sofa::gl::glVertexT(corners[0]); sofa::gl::glVertexT(corners[2]);
        sofa::gl::glVertexT(corners[1]); sofa::gl::glVertexT(corners[3]);
        sofa::gl::glVertexT(corners[4]); sofa::gl::glVertexT(corners[6]);
        sofa::gl::glVertexT(corners[5]); sofa::gl::glVertexT(corners[7]);
        sofa::gl::glVertexT(corners[0]); sofa::gl::glVertexT(corners[1]);
        sofa::gl::glVertexT(corners[2]); sofa::gl::glVertexT(corners[3]);
        sofa::gl::glVertexT(corners[4]); sofa::gl::glVertexT(corners[5]);
        sofa::gl::glVertexT(corners[6]); sofa::gl::glVertexT(corners[7]);
    }
    glEnd();

    const SReal mindist = (SReal)(this->showMinDist.isSet() ? this->showMinDist.getValue() :
        -(grid->getPMax()-grid->getPMin()).norm()*0.1);
    const SReal maxdist = (SReal)(this->showMaxDist.isSet() ? this->showMaxDist.getValue() :
         (grid->getPMax()-grid->getPMin()).norm()*0.025);

    if (this->showGridPoints.getValue())
    {
        int dnz = (grid->getNz() < 128) ? grid->getNz() : 128;
        int dny = (grid->getNy() < 128) ? grid->getNy() : 128;
        int dnx = (grid->getNx() < 128) ? grid->getNx() : 128;
        glBegin(GL_POINTS);
        if (dnx >= 2 && dny >= 2 && dnz >= 2)
        {
            for (int iz=0; iz<dnz; ++iz)
            {
                int z = (iz*(grid->getNz()-1))/(dnz-1);
                for (int iy=0; iy<dny; ++iy)
                {
                    int y = (iy*(grid->getNy()-1))/(dny-1);
                    for (int ix=0; ix<dnx; ++ix)
                    {
                        int x = (ix*(grid->getNx()-1))/(dnx-1);
                        DistanceGrid::Coord p = grid->coord(x,y,z);
                        SReal d = (*grid)[grid->index(x,y,z)];
                        if (flipped) d = -d;
                        if (d < mindist || d > maxdist) continue;
                        d /= maxdist;
                        if (d<0)
                            glColor3d(1+d*0.25, 0, 1+d);
                        else
                            continue; //glColor3d(0, 1-d*0.25, 1-d);
                        sofa::gl::glVertexT(p);
                    }
                }
            }
        }
        glEnd();
    }
    if (!grid->meshPts.empty() && this->showMeshPoints.getValue())
    {
        glColor3d(1, 1 ,1);
        glBegin(GL_POINTS);
        for (unsigned int i=0; i<grid->meshPts.size(); i++)
        {
            DistanceGrid::Coord p = grid->meshPts[i];
            sofa::gl::glVertexT(p);
        }
        glEnd();
        glBegin(GL_LINES);
        for (unsigned int i=0; i<grid->meshPts.size(); i++)
        {
            DistanceGrid::Coord p = grid->meshPts[i];
            glColor3d(1, 1 ,1);
            DistanceGrid::Coord grad = grid->grad(p);
            if (flipped) grad = -grad;
            grad.normalize();
            for (int j = -2; j <= 2; j++)
            {
                DistanceGrid::Coord p2 = p + grad * (j*maxdist/2);
                SReal d = grid->eval(p2);
                if (flipped) d = -d;
                //if (rabs(d) > maxdist) continue;
                d /= maxdist;
                if (d<0)
                    glColor3d(1+d*0.25, 0, 1+d);
                else
                    glColor3d(0, 1-d*0.25, 1-d);
                sofa::gl::glVertexT(p2);
                if (j>-2 && j < 2)
                    sofa::gl::glVertexT(p2);
            }
        }
        glEnd();
    }
    if (elems[index].isTransformed)
    {
        glPopMatrix();
    }
#endif // SOFADISTANCEGRID_HAVE_SOFA_GL == 1
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

FFDDistanceGridCollisionModel::FFDDistanceGridCollisionModel()
    : fileFFDDistanceGrid( initData( &fileFFDDistanceGrid, "filename", "Load distance grid from specified file"))
    , scale( initData( &scale, 1.0, "scale", "scaling factor for input file"))
    , sampling( initData( &sampling, 0.0, "sampling", "if not zero: sample the surface with points approximately separated by the given sampling distance (expressed in voxels if the value is negative)"))
    , box( initData( &box, "box", "Field bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , nx( initData( &nx, 64, "nx", "number of values on X axis") )
    , ny( initData( &ny, 64, "ny", "number of values on Y axis") )
    , nz( initData( &nz, 64, "nz", "number of values on Z axis") )
    , dumpfilename( initData( &dumpfilename, "dumpfilename","write distance grid to specified file"))
    , usePoints( initData( &usePoints, true, "usePoints", "use mesh vertices for collision detection"))
    , singleContact( initData( &singleContact, false, "singleContact", "keep only the deepest contact in each cell"))
    , l_ffdMesh(initLink("topology", "link to the topological mesh associated with this collision model"))
{
    addAlias(&fileFFDDistanceGrid,"fileFFDDistanceGrid");
    enum_type = FFDDISTANCE_GRIDE_TYPE;
}

FFDDistanceGridCollisionModel::~FFDDistanceGridCollisionModel()
{}

void FFDDistanceGridCollisionModel::init()
{
    Inherit1::init();
    Inherit2::init();
    if (l_ffdMesh.empty())
    {
        msg_warning() << "Link to Topology should be set to a RegularGridTopology or SparseGridTopology. First Topology found in context will be used";
        l_ffdMesh.set(this->getContext()->getMeshTopologyLink());
    }
    topology::container::grid::RegularGridTopology* ffdRGrid = dynamic_cast< topology::container::grid::RegularGridTopology* > (l_ffdMesh.get());
    topology::container::grid::SparseGridTopology* ffdSGrid = dynamic_cast< topology::container::grid::SparseGridTopology* > (l_ffdMesh.get());
    if (!this->mstate || (!ffdRGrid && !ffdSGrid))
    {
        msg_error() << "Requires a Vec3-based deformable model with associated RegularGridTopology or SparseGridTopology";
        return;
    }

    std::shared_ptr<DistanceGrid> grid;
    if (fileFFDDistanceGrid.getValue().empty())
    {
        msg_error() << "Requires an input filename";
        return;
    }
    msg_info() << "Creating "<<nx.getValue()<<"x"<<ny.getValue()<<"x"<<nz.getValue()<<" DistanceGrid from file "<<fileFFDDistanceGrid.getValue();
    if (scale.getValue()!=1.0) msg_info()<<" scale="<<scale.getValue();
    if (sampling.getValue()!=0.0) msg_info()<<" sampling="<<sampling.getValue();
    if (box.getValue()[0][0]<box.getValue()[1][0]) msg_info()<<" bbox=<"<<box.getValue()[0]<<">-<"<<box.getValue()[0]<<">";

    grid = DistanceGrid::loadShared(fileFFDDistanceGrid.getFullPath(), scale.getValue(), sampling.getValue(), nx.getValue(),ny.getValue(),nz.getValue(),box.getValue()[0],box.getValue()[1]);
    if (!dumpfilename.getValue().empty())
    {
        msg_info() << "Dump grid to "<<dumpfilename.getValue();
        grid->save(dumpfilename.getFullPath());
    }
    /// place points in ffd elements
    int nbp = grid->meshPts.size();
    elems.resize(l_ffdMesh->getNbHexahedra());
    msg_info() << "Placing "<<nbp<<" points in "<<l_ffdMesh->getNbHexahedra()<<" cubes.";

    for (int i=0; i<nbp; i++)
    {
        Vec3Types::Coord p0 = grid->meshPts[i];
        Vec3 bary;
        sofa::Index elem = (ffdRGrid ? ffdRGrid->findCube(p0,bary[0],bary[1],bary[2]) : ffdSGrid->findCube(p0,bary[0],bary[1],bary[2]));
        if (elem == sofa::InvalidID) continue;
        if (elem >= elems.size())
        {
            msg_error() << "point "<<i<<" "<<p0<<" in invalid cube "<<elem;
        }
        else
        {
            DeformedCube::Point p;
            p.index = i;
            p.bary = bary;
            elems[elem].points.push_back(p);
            GCoord n = grid->grad(p0);
            n.normalize();
            elems[elem].normals.push_back(n);
        }
    }
    /// fill other data and remove inactive elements

    msg_info() << "Initializing "<<l_ffdMesh->getNbHexahedra()<<" cubes.";
    sofa::Size c=0;
    for (sofa::Size e=0; e<l_ffdMesh->getNbHexahedra(); e++)
    {
        if (c != e)
            elems[c].points.swap(elems[e].points); // move the list of points to the new
        elems[c].elem = e;

        core::topology::BaseMeshTopology::Hexa cube = (ffdRGrid ? ffdRGrid->getHexaCopy(e) : ffdSGrid->getHexahedron(e));
        { int t = cube[2]; cube[2] = cube[3]; cube[3] = t; }
        { int t = cube[6]; cube[6] = cube[7]; cube[7] = t; }

        elems[c].initP0 = GCoord(l_ffdMesh->getPX(cube[0]), l_ffdMesh->getPY(cube[0]), l_ffdMesh->getPZ(cube[0]));
        elems[c].initDP = GCoord(l_ffdMesh->getPX(cube[7]), l_ffdMesh->getPY(cube[7]), l_ffdMesh->getPZ(cube[7]))-elems[c].initP0;
        elems[c].invDP[0] = 1/elems[c].initDP[0];
        elems[c].invDP[1] = 1/elems[c].initDP[1];
        elems[c].invDP[2] = 1/elems[c].initDP[2];
        elems[c].grid = grid;
        ++c;
    }
    resize(c);

    /// compute neighbors
    type::vector<std::set<int> > shells;
    shells.resize(l_ffdMesh->getNbPoints());
    for (unsigned i = 0; i < elems.size(); ++i)
    {
        int e = elems[i].elem;
        core::topology::BaseMeshTopology::Hexa cube = (ffdRGrid ? ffdRGrid->getHexaCopy(e) : ffdSGrid->getHexahedron(e));
        { int t = cube[2]; cube[2] = cube[3]; cube[3] = t; }
        { int t = cube[6]; cube[6] = cube[7]; cube[7] = t; }

        for (int j=0; j<8; ++j)
            shells[cube[j]].insert(i);
    }

    for (unsigned i = 0; i < elems.size(); ++i)
    {
        int e = elems[i].elem;
        core::topology::BaseMeshTopology::Hexa cube = (ffdRGrid ? ffdRGrid->getHexaCopy(e) : ffdSGrid->getHexahedron(e));
        { int t = cube[2]; cube[2] = cube[3]; cube[3] = t; }
        { int t = cube[6]; cube[6] = cube[7]; cube[7] = t; }

        for (int j=0; j<8; ++j)
            elems[i].neighbors.insert(shells[cube[j]].begin(), shells[cube[j]].end());
        elems[i].neighbors.erase(i);
    }

    msg_info() << c <<" active cubes.";
}

void FFDDistanceGridCollisionModel::resize(sofa::Size s)
{
    this->core::CollisionModel::resize(s);
    elems.resize(s);
}

bool FFDDistanceGridCollisionModel::canCollideWithElement(sofa::Index index, CollisionModel* model2, sofa::Index index2)
{
    if (model2 != this) return true;
    if (!this->bSelfCollision.getValue()) return true;

    if (index >= index2) return false;
    if (elems[index].neighbors.count(index2)) return false;
    return true;
}

/// Create or update the bounding volume hierarchy.
void FFDDistanceGridCollisionModel::computeBoundingTree(int maxDepth)
{
    CubeCollisionModel* cubeModel = this->createPrevious<CubeCollisionModel>();

    if (!isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    updateGrid();

    cubeModel->resize(size);
    for (sofa::Size i=0; i<size; i++)
    {
        Vec3 emin, emax;
        const DeformedCube& cube = getDeformCube(i);
        {
            emin = cube.corners[0];
            emax = emin;
            for (int j=1; j<8; j++)
            {
                Vec3 corner = cube.corners[j];
                for(int c=0; c<3; c++)
                    if (corner[c] < emin[c]) emin[c] = corner[c];
                    else if (corner[c] > emax[c]) emax[c] = corner[c];
            }
        }
        cubeModel->setParentOf(i, emin, emax); // define the bounding box of the current element
    }
    cubeModel->computeBoundingTree(maxDepth);
}

void FFDDistanceGridCollisionModel::updateGrid()
{
    for (sofa::Size index=0; index<size; index++)
    {
        DeformedCube& cube = getDeformCube( index );
        const sofa::type::vector<core::topology::BaseMeshTopology::Hexa>& cubeCorners = l_ffdMesh->getHexahedra();

        const Vec3Types::VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
        {
            int e = cube.elem;
            DistanceGrid::Coord center;
            core::topology::BaseMeshTopology::Hexa c = cubeCorners[e];
            { int t = c[2]; c[2] = c[3]; c[3] = t; }
            { int t = c[6]; c[6] = c[7]; c[7] = t; }

            for (int j=0; j<8; j++)
            {
                cube.corners[j] = x[c[j]];
                center += cube.corners[j];
            }
            cube.center = center * 0.125f;
            SReal radius2 = 0.0f;
            for (int j=0; j<8; j++)
            {
                SReal r2 = (cube.corners[j] - cube.center).norm2();
                if (r2 > radius2) radius2 = r2;
            }
            cube.radius = sofa::helper::rsqrt(radius2);
            cube.updateDeform();
            cube.pointsUpdated = false;
            cube.facesUpdated = false;
        }
    }
}

/// Update the deformation precomputed values
void FFDDistanceGridCollisionModel::DeformedCube::updateDeform()
{
    Dx = corners[C100]-corners[C000];   // Dx = -C000+C100
    Dy = corners[C010]-corners[C000];   // Dy = -C000+C010
    Dz = corners[C001]-corners[C000];   // Dx = -C000+C001
    Dxy = corners[C110]-corners[C010]-Dx;  // Dxy = C000-C100-C010+C110 = C110-C010-Dx
    Dxz = corners[C101]-corners[C001]-Dx;  // Dxz = C000-C100-C001+C101 = C101-C001-Dx
    Dyz = corners[C011]-corners[C001]-Dy;  // Dyz = C000-C010-C001+C011 = C011-C001-Dy
    Dxyz = corners[C111]-corners[C101]-corners[C011]+corners[C001]-Dxy; // Dxyz = - C000 + C100 + C010 - C110 + C001 - C101 - C011 + C111 = C001 - C101 - C011 + C111 - Dxy
}

/// Update the deformedPoints position if not done yet (i.e. if pointsUpdated==false)
void FFDDistanceGridCollisionModel::DeformedCube::updatePoints()
{
    if (!pointsUpdated)
    {
        deformedPoints.resize(points.size());
        deformedNormals.resize(points.size());
        for (unsigned int i=0; i<points.size(); i++)
        {
            deformedPoints[i] = deform(points[i].bary);
            deformedNormals[i] = deformDir(points[i].bary, normals[i]);
            deformedNormals[i].normalize();
        }
        pointsUpdated = true;
    }
}

/// Update the face planes position if not done yet (i.e. if facesUpdated==false)
void FFDDistanceGridCollisionModel::DeformedCube::updateFaces()
{
    if (!facesUpdated)
    {
        faces[FX0] = computePlane(C000,C010,C001,C011); faces[FX1] = computePlane(C100,C101,C110,C111);
        faces[FY0] = computePlane(C000,C001,C100,C101); faces[FY1] = computePlane(C010,C110,C011,C111);
        faces[FZ0] = computePlane(C000,C100,C010,C110); faces[FZ1] = computePlane(C001,C011,C101,C111);
        facesUpdated = true;
    }
}

FFDDistanceGridCollisionModel::DeformedCube::Plane FFDDistanceGridCollisionModel::DeformedCube::computePlane(int c00, int c10, int c01, int c11)
{
    GCoord C4 = (corners[c00]+corners[c10]+corners[c01]+corners[c11]); //*0.25f;
    GCoord N = (corners[c11]-corners[c00]).cross(corners[c01]-corners[c10]);
    N.normalize();
    return Plane(N,N*C4*(-0.25f));
}

void FFDDistanceGridCollisionModel::drawCollisionModel(const core::visual::VisualParams* vparams)
{
#if SOFADISTANCEGRID_HAVE_SOFA_GL == 1
    if (vparams->displayFlags().getShowWireFrame())
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    glDisable(GL_LIGHTING);
    glColor4fv(getColor4f());
    for (unsigned int i = 0; i < elems.size(); i++)
    {
        draw(vparams, i);
    }
    if (vparams->displayFlags().getShowWireFrame())
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
#endif // SOFADISTANCEGRID_HAVE_SOFA_GL == 1
}

void FFDDistanceGridCollisionModel::draw(const core::visual::VisualParams* vparams, sofa::Index index)
{
#if SOFADISTANCEGRID_HAVE_SOFA_GL == 1
    DeformedCube& cube = getDeformCube( index );
    float cscale;
    if (!isSimulated())
        cscale = 0.5f;
    else
        cscale = 1.0f;
    if (cube.pointsUpdated && cube.facesUpdated)
        glColor4f(1.0f*cscale, 0.0f*cscale, 0.0f*cscale, 1.0f);
    else if (cube.pointsUpdated)
        glColor4f(1.0f*cscale, 0.5f*cscale, 0.0f*cscale, 1.0f);
    else if (cube.facesUpdated)
        glColor4f(0.5f*cscale, 1.0f*cscale, 0.0f*cscale, 1.0f);
    else
        glColor4f(0.0f*cscale, 1.0f*cscale, 0.5f*cscale, 1.0f);
    glBegin(GL_LINES);
    {
        sofa::gl::glVertexT(cube.corners[0]); sofa::gl::glVertexT(cube.corners[4]);
        sofa::gl::glVertexT(cube.corners[1]); sofa::gl::glVertexT(cube.corners[5]);
        sofa::gl::glVertexT(cube.corners[2]); sofa::gl::glVertexT(cube.corners[6]);
        sofa::gl::glVertexT(cube.corners[3]); sofa::gl::glVertexT(cube.corners[7]);
        sofa::gl::glVertexT(cube.corners[0]); sofa::gl::glVertexT(cube.corners[2]);
        sofa::gl::glVertexT(cube.corners[1]); sofa::gl::glVertexT(cube.corners[3]);
        sofa::gl::glVertexT(cube.corners[4]); sofa::gl::glVertexT(cube.corners[6]);
        sofa::gl::glVertexT(cube.corners[5]); sofa::gl::glVertexT(cube.corners[7]);
        sofa::gl::glVertexT(cube.corners[0]); sofa::gl::glVertexT(cube.corners[1]);
        sofa::gl::glVertexT(cube.corners[2]); sofa::gl::glVertexT(cube.corners[3]);
        sofa::gl::glVertexT(cube.corners[4]); sofa::gl::glVertexT(cube.corners[5]);
        sofa::gl::glVertexT(cube.corners[6]); sofa::gl::glVertexT(cube.corners[7]);
    }
    glEnd();
    glLineWidth(2);
    glPointSize(5);
    {
        glBegin(GL_POINTS);
        {
            sofa::gl::glVertexT(cube.center);

        }
        glEnd();
    }
    glLineWidth(1);
    if (cube.pointsUpdated)
    {
        glPointSize(2);
        glColor4f(1.0f, 0.5f, 0.5f, 1.0f);
        glBegin(GL_POINTS);
        for (unsigned int j=0; j<cube.deformedPoints.size(); j++)
            sofa::gl::glVertexT(cube.deformedPoints[j]);
        glEnd();
        if (vparams->displayFlags().getShowNormals())
        {
            glBegin(GL_LINES);
            for (unsigned int j=0; j<cube.deformedNormals.size(); j++)
            {
                sofa::gl::glVertexT(cube.deformedPoints[j]);
                sofa::gl::glVertexT(cube.deformedPoints[j] + cube.deformedNormals[j]);
            }
            glEnd();
        }
    }
    glPointSize(1);
#endif // SOFADISTANCEGRID_HAVE_SOFA_GL == 1
}

ContactMapperCreator< response::mapper::ContactMapper<FFDDistanceGridCollisionModel> > FFDDistanceGridContactMapperClass("PenalityContactForceField", true);

template class SOFA_SOFADISTANCEGRID_API response::mapper::ContactMapper<FFDDistanceGridCollisionModel, sofa::defaulttype::Vec3Types>;


ContactMapperCreator< response::mapper::ContactMapper<RigidDistanceGridCollisionModel> > DistanceGridContactMapperClass("PenalityContactForceField", true);

template class SOFA_SOFADISTANCEGRID_API response::mapper::ContactMapper<RigidDistanceGridCollisionModel, sofa::defaulttype::Vec3Types>;

} // namespace collision

} // namespace component

} // namespace sofa

