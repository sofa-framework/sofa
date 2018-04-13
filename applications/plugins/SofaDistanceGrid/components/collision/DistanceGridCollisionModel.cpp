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
#define SOFA_COMPONENT_COLLISION_DISTANCEGRIDCOLLISIONMODEL_CPP
#include <fstream>
#include <sstream>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/Factory.inl>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include <SofaMeshCollision/RigidContactMapper.inl>

#include "DistanceGridCollisionModel.h"


namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(DistanceGridCollisionModel)

int RigidDistanceGridCollisionModelClass = core::RegisterObject("Grid-based distance field")
        .add< RigidDistanceGridCollisionModel >()
        .addAlias("DistanceGridCollisionModel")
        .addAlias("RigidDistanceGrid")
        .addAlias("DistanceGrid")
        ;

int FFDDistanceGridCollisionModelClass = core::RegisterObject("Grid-based deformable distance field")
        .add< FFDDistanceGridCollisionModel >()
        .addAlias("FFDDistanceGrid")
        ;

using namespace defaulttype;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

RigidDistanceGridCollisionModel::RigidDistanceGridCollisionModel()
    : modified(true)
    , fileRigidDistanceGrid( initData( &fileRigidDistanceGrid, "fileRigidDistanceGrid", "load distance grid from specified file"))
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
    rigid = NULL;
    addAlias(&fileRigidDistanceGrid,"filename");
}

RigidDistanceGridCollisionModel::~RigidDistanceGridCollisionModel()
{
    for (unsigned int i=0; i<elems.size(); i++)
    {
        if (elems[i].grid!=NULL) elems[i].grid->release();
        if (elems[i].prevGrid!=NULL) elems[i].prevGrid->release();
    }
}

void RigidDistanceGridCollisionModel::init()
{
    this->core::CollisionModel::init();
    rigid = dynamic_cast< core::behavior::MechanicalState<RigidTypes>* > (getContext()->getMechanicalState());

    DistanceGrid* grid = NULL;
    if (fileRigidDistanceGrid.getValue().empty())
    {
        if (elems.size()==0 || elems[0].grid==NULL)
            serr << "ERROR: RigidDistanceGridCollisionModel requires an input filename." << sendl;
        // else the grid has already been set
        return;
    }
    sout << "RigidDistanceGridCollisionModel: creating "<<nx.getValue()<<"x"<<ny.getValue()<<"x"<<nz.getValue()<<" DistanceGrid from file "<<fileRigidDistanceGrid.getValue();
    if (scale.getValue()!=1.0) sout<<" scale="<<scale.getValue();
    if (sampling.getValue()!=0.0) sout<<" sampling="<<sampling.getValue();
    if (box.getValue()[0][0]<box.getValue()[1][0]) sout<<" bbox=<"<<box.getValue()[0]<<">-<"<<box.getValue()[0]<<">";
    sout << sendl;
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
        sout << "RigidDistanceGridCollisionModel: dump grid to "<<dumpfilename.getValue()<<sendl;
        grid->save(dumpfilename.getFullPath());
    }
    updateState();
    sout << "< RigidDistanceGridCollisionModel::init()"<<sendl;
}

void RigidDistanceGridCollisionModel::resize(int s)
{
    this->core::CollisionModel::resize(s);
    elems.resize(s);
}

void RigidDistanceGridCollisionModel::setGrid(DistanceGrid* surf, int index)
{
    if (elems[index].grid == surf) return;
    if (elems[index].grid!=NULL) elems[index].grid->release();
    elems[index].grid = surf->addRef();
    modified = true;
}

void RigidDistanceGridCollisionModel::setNewState(int index, double dt, DistanceGrid* grid, const Matrix3& rotation, const Vector3& translation)
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

/// Update transformation matrices from current rigid state
void RigidDistanceGridCollisionModel::updateState()
{
    const Vector3& initTranslation = this->translation.getValue();
    const Vector3& initRotation = this->rotation.getValue();
    bool useInitTranslation = (initTranslation != DistanceGrid::Coord());
    bool useInitRotation = (initRotation != Vector3(0,0,0));

    for (int i=0; i<size; i++)
    {
        //static_cast<DistanceGridCollisionElement*>(elems[i])->recalcBBox();
        Vector3 emin, emax;
        if (rigid)
        {
            const RigidTypes::Coord& xform = (rigid->read(core::ConstVecCoordId::position())->getValue())[i];
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
    CubeModel* cubeModel = this->createPrevious<CubeModel>();

    if (!modified && !isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    updateGrid();
    updateState();

    const bool flipped = isFlipped();
    cubeModel->resize(size);
    for (int i=0; i<size; i++)
    {
        //static_cast<DistanceGridCollisionElement*>(elems[i])->recalcBBox();
        Vector3 emin, emax;
        if (elems[i].isTransformed)
        {
            for (int j=0; j<8; j++)
            {
                Vector3 corner = elems[i].translation + elems[i].rotation * (flipped ? elems[i].grid->getCorner(j) : elems[i].grid->getBBCorner(j));
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

void RigidDistanceGridCollisionModel::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
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
#endif /* SOFA_NO_OPENGL */
}

void RigidDistanceGridCollisionModel::draw(const core::visual::VisualParams* ,int index)
{
#ifndef SOFA_NO_OPENGL
    const bool flipped = isFlipped();

    if (elems[index].isTransformed)
    {
        glPushMatrix();
        // float m[16];
        // (*rigid->read(sofa::core::ConstVecCoordId::position())->getValue())[index].writeOpenGlMatrix( m );
        // glMultMatrixf(m);
        Matrix4 m;
        m.identity();
        m = elems[index].rotation;
        m.transpose();
        m[3] = Vector4(elems[index].translation,1.0);

        helper::gl::glMultMatrix(m.ptr());
    }

    DistanceGrid* grid = getGrid(index);
    DistanceGrid::Coord corners[8];
    for(unsigned int i=0; i<8; i++)
        corners[i] = grid->getCorner(i);
    //glEnable(GL_BLEND);
    //glDepthMask(0);
    if (!isSimulated())
        glColor4f(0.25f, 0.25f, 0.25f, 0.1f);
    else
        glColor4f(0.5f, 0.5f, 0.5f, 0.1f);
    glBegin(GL_LINES);
    {
        helper::gl::glVertexT(corners[0]); helper::gl::glVertexT(corners[4]);
        helper::gl::glVertexT(corners[1]); helper::gl::glVertexT(corners[5]);
        helper::gl::glVertexT(corners[2]); helper::gl::glVertexT(corners[6]);
        helper::gl::glVertexT(corners[3]); helper::gl::glVertexT(corners[7]);
        helper::gl::glVertexT(corners[0]); helper::gl::glVertexT(corners[2]);
        helper::gl::glVertexT(corners[1]); helper::gl::glVertexT(corners[3]);
        helper::gl::glVertexT(corners[4]); helper::gl::glVertexT(corners[6]);
        helper::gl::glVertexT(corners[5]); helper::gl::glVertexT(corners[7]);
        helper::gl::glVertexT(corners[0]); helper::gl::glVertexT(corners[1]);
        helper::gl::glVertexT(corners[2]); helper::gl::glVertexT(corners[3]);
        helper::gl::glVertexT(corners[4]); helper::gl::glVertexT(corners[5]);
        helper::gl::glVertexT(corners[6]); helper::gl::glVertexT(corners[7]);
    }
    glEnd();
    glDisable(GL_BLEND);
    glDepthMask(1);
    for(unsigned int i=0; i<8; i++)
        corners[i] = grid->getBBCorner(i);
    //glEnable(GL_BLEND);
    //glDepthMask(0);

    if (!isSimulated())
        glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
    else
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_LINES);
    {
        helper::gl::glVertexT(corners[0]); helper::gl::glVertexT(corners[4]);
        helper::gl::glVertexT(corners[1]); helper::gl::glVertexT(corners[5]);
        helper::gl::glVertexT(corners[2]); helper::gl::glVertexT(corners[6]);
        helper::gl::glVertexT(corners[3]); helper::gl::glVertexT(corners[7]);
        helper::gl::glVertexT(corners[0]); helper::gl::glVertexT(corners[2]);
        helper::gl::glVertexT(corners[1]); helper::gl::glVertexT(corners[3]);
        helper::gl::glVertexT(corners[4]); helper::gl::glVertexT(corners[6]);
        helper::gl::glVertexT(corners[5]); helper::gl::glVertexT(corners[7]);
        helper::gl::glVertexT(corners[0]); helper::gl::glVertexT(corners[1]);
        helper::gl::glVertexT(corners[2]); helper::gl::glVertexT(corners[3]);
        helper::gl::glVertexT(corners[4]); helper::gl::glVertexT(corners[5]);
        helper::gl::glVertexT(corners[6]); helper::gl::glVertexT(corners[7]);
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
                        helper::gl::glVertexT(p);
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
            helper::gl::glVertexT(p);
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
                helper::gl::glVertexT(p2);
                if (j>-2 && j < 2)
                    helper::gl::glVertexT(p2);
            }
        }
        glEnd();
    }
    if (elems[index].isTransformed)
    {
        glPopMatrix();
    }
#endif /* SOFA_NO_OPENGL */
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

FFDDistanceGridCollisionModel::FFDDistanceGridCollisionModel()
    : fileFFDDistanceGrid( initData( &fileFFDDistanceGrid, "fileFFDDistanceGrid", "load distance grid from specified file"))
    , scale( initData( &scale, 1.0, "scale", "scaling factor for input file"))
    , sampling( initData( &sampling, 0.0, "sampling", "if not zero: sample the surface with points approximately separated by the given sampling distance (expressed in voxels if the value is negative)"))
    , box( initData( &box, "box", "Field bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , nx( initData( &nx, 64, "nx", "number of values on X axis") )
    , ny( initData( &ny, 64, "ny", "number of values on Y axis") )
    , nz( initData( &nz, 64, "nz", "number of values on Z axis") )
    , dumpfilename( initData( &dumpfilename, "dumpfilename","write distance grid to specified file"))
    , usePoints( initData( &usePoints, true, "usePoints", "use mesh vertices for collision detection"))
    , singleContact( initData( &singleContact, false, "singleContact", "keep only the deepest contact in each cell"))
{
    ffd = NULL;
    ffdMesh = NULL;
    ffdRGrid = NULL;
    ffdSGrid = NULL;
    addAlias(&fileFFDDistanceGrid,"filename");
    enum_type = FFDDISTANCE_GRIDE_TYPE;
}

FFDDistanceGridCollisionModel::~FFDDistanceGridCollisionModel()
{
    if (elems.size()>0 && elems[0].grid!=NULL) elems[0].grid->release();
}

void FFDDistanceGridCollisionModel::init()
{
    this->core::CollisionModel::init();
    ffd = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());
    ffdMesh = /*dynamic_cast< topology::RegularGridTopology* >*/ (getContext()->getMeshTopology());
    ffdRGrid = dynamic_cast< topology::RegularGridTopology* > (ffdMesh);
    ffdSGrid = dynamic_cast< topology::SparseGridTopology* > (ffdMesh);
    if (!ffd || (!ffdRGrid && !ffdSGrid))
    {
        serr <<"FFDDistanceGridCollisionModel requires a Vec3-based deformable model with associated RegularGridTopology or SparseGridTopology" << sendl;
        return;
    }

    DistanceGrid* grid = NULL;
    if (fileFFDDistanceGrid.getValue().empty())
    {
        serr<<"ERROR: FFDDistanceGridCollisionModel requires an input filename" << sendl;
        return;
    }
    sout << "FFDDistanceGridCollisionModel: creating "<<nx.getValue()<<"x"<<ny.getValue()<<"x"<<nz.getValue()<<" DistanceGrid from file "<<fileFFDDistanceGrid.getValue();
    if (scale.getValue()!=1.0) sout<<" scale="<<scale.getValue();
    if (sampling.getValue()!=0.0) sout<<" sampling="<<sampling.getValue();
    if (box.getValue()[0][0]<box.getValue()[1][0]) sout<<" bbox=<"<<box.getValue()[0]<<">-<"<<box.getValue()[0]<<">";
    sout << sendl;
    grid = DistanceGrid::loadShared(fileFFDDistanceGrid.getFullPath(), scale.getValue(), sampling.getValue(), nx.getValue(),ny.getValue(),nz.getValue(),box.getValue()[0],box.getValue()[1]);
    if (!dumpfilename.getValue().empty())
    {
        sout << "FFDDistanceGridCollisionModel: dump grid to "<<dumpfilename.getValue()<<sendl;
        grid->save(dumpfilename.getFullPath());
    }
    /// place points in ffd elements
    int nbp = grid->meshPts.size();
#ifdef SOFA_NEW_HEXA
    elems.resize(ffdMesh->getNbHexahedra());
    sout << "FFDDistanceGridCollisionModel: placing "<<nbp<<" points in "<<ffdMesh->getNbHexahedra()<<" cubes."<<sendl;
#else
    elems.resize(ffdMesh->getNbHexahedra());
    sout << "FFDDistanceGridCollisionModel: placing "<<nbp<<" points in "<<ffdMesh->getNbCubes()<<" cubes."<<sendl;
#endif
    for (int i=0; i<nbp; i++)
    {
        Vec3Types::Coord p0 = grid->meshPts[i];
        Vector3 bary;
        int elem = (ffdRGrid ? ffdRGrid->findCube(p0,bary[0],bary[1],bary[2]) : ffdSGrid->findCube(p0,bary[0],bary[1],bary[2]));
        if (elem == -1) continue;
        if ((unsigned)elem >= elems.size())
        {
            serr << "ERROR (FFDDistanceGridCollisionModel): point "<<i<<" "<<p0<<" in invalid cube "<<elem<<sendl;
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

#ifdef SOFA_NEW_HEXA
    sout << "FFDDistanceGridCollisionModel: initializing "<<ffdMesh->getNbHexahedra()<<" cubes."<<sendl;
    int c=0;
    for (int e=0; e<ffdMesh->getNbHexahedra(); e++)
#else
    sout << "FFDDistanceGridCollisionModel: initializing "<<ffdMesh->getNbCubes()<<" cubes."<<sendl;
    int c=0;
    for (int e=0; e<ffdMesh->getNbCubes(); e++)
#endif
    {
        if (c != e)
            elems[c].points.swap(elems[e].points); // move the list of points to the new
        elems[c].elem = e;
#ifdef SOFA_NEW_HEXA
        core::topology::BaseMeshTopology::Hexa cube = (ffdRGrid ? ffdRGrid->getHexaCopy(e) : ffdSGrid->getHexahedron(e));
        { int t = cube[2]; cube[2] = cube[3]; cube[3] = t; }
        { int t = cube[6]; cube[6] = cube[7]; cube[7] = t; }
#else
        core::topology::BaseMeshTopology::Cube cube = (ffdRGrid ? ffdRGrid->getCubeCopy(e) : ffdSGrid->getCube(e));
#endif
        elems[c].initP0 = GCoord(ffdMesh->getPX(cube[0]), ffdMesh->getPY(cube[0]), ffdMesh->getPZ(cube[0]));
        elems[c].initDP = GCoord(ffdMesh->getPX(cube[7]), ffdMesh->getPY(cube[7]), ffdMesh->getPZ(cube[7]))-elems[c].initP0;
        elems[c].invDP[0] = 1/elems[c].initDP[0];
        elems[c].invDP[1] = 1/elems[c].initDP[1];
        elems[c].invDP[2] = 1/elems[c].initDP[2];
        elems[c].grid = grid;
        ++c;
    }
    resize(c);

    /// compute neighbors
    helper::vector<std::set<int> > shells;
    shells.resize(ffdMesh->getNbPoints());
    for (unsigned i = 0; i < elems.size(); ++i)
    {
        int e = elems[i].elem;
#ifdef SOFA_NEW_HEXA
        core::topology::BaseMeshTopology::Hexa cube = (ffdRGrid ? ffdRGrid->getHexaCopy(e) : ffdSGrid->getHexahedron(e));
        { int t = cube[2]; cube[2] = cube[3]; cube[3] = t; }
        { int t = cube[6]; cube[6] = cube[7]; cube[7] = t; }
#else
        core::topology::BaseMeshTopology::Cube cube = (ffdRGrid ? ffdRGrid->getCubeCopy(e) : ffdSGrid->getCube(e));
#endif
        for (int j=0; j<8; ++j)
            shells[cube[j]].insert(i);
    }

    for (unsigned i = 0; i < elems.size(); ++i)
    {
        int e = elems[i].elem;
#ifdef SOFA_NEW_HEXA
        core::topology::BaseMeshTopology::Hexa cube = (ffdRGrid ? ffdRGrid->getHexaCopy(e) : ffdSGrid->getHexahedron(e));
        { int t = cube[2]; cube[2] = cube[3]; cube[3] = t; }
        { int t = cube[6]; cube[6] = cube[7]; cube[7] = t; }
#else
        core::topology::BaseMeshTopology::Cube cube = (ffdRGrid ? ffdRGrid->getCubeCopy(e) : ffdSGrid->getCube(e));
#endif
        for (int j=0; j<8; ++j)
            elems[i].neighbors.insert(shells[cube[j]].begin(), shells[cube[j]].end());
        elems[i].neighbors.erase(i);
    }

    sout << "FFDDistanceGridCollisionModel: "<<c<<" active cubes."<<sendl;
}

void FFDDistanceGridCollisionModel::resize(int s)
{
    this->core::CollisionModel::resize(s);
    elems.resize(s);
}

bool FFDDistanceGridCollisionModel::canCollideWithElement(int index, CollisionModel* model2, int index2)
{
    if (model2 != this) return true;
    if (!this->bSelfCollision.getValue()) return true;

    if (index >= index2) return false;
    if (elems[index].neighbors.count(index2)) return false;
    return true;
}

void FFDDistanceGridCollisionModel::setGrid(DistanceGrid* surf, int index)
{
    elems[index].grid = surf;
}

/// Create or update the bounding volume hierarchy.
void FFDDistanceGridCollisionModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = this->createPrevious<CubeModel>();

    if (!isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    updateGrid();

    cubeModel->resize(size);
    for (int i=0; i<size; i++)
    {
        Vector3 emin, emax;
        const DeformedCube& cube = getDeformCube(i);
        {
            emin = cube.corners[0];
            emax = emin;
            for (int j=1; j<8; j++)
            {
                Vector3 corner = cube.corners[j];
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
    for (int index=0; index<size; index++)
    {
        DeformedCube& cube = getDeformCube( index );
#ifdef SOFA_NEW_HEXA
        const sofa::helper::vector<core::topology::BaseMeshTopology::Hexa>& cubeCorners = ffdMesh->getHexahedra();
#else
        const sofa::helper::vector<core::topology::BaseMeshTopology::Cube>& cubeCorners = ffdMesh->getCubes();
#endif
        const Vec3Types::VecCoord& x = ffd->read(core::ConstVecCoordId::position())->getValue();
        {
            int e = cube.elem;
            DistanceGrid::Coord center;
#ifdef SOFA_NEW_HEXA
            core::topology::BaseMeshTopology::Hexa c = cubeCorners[e];
            { int t = c[2]; c[2] = c[3]; c[3] = t; }
            { int t = c[6]; c[6] = c[7]; c[7] = t; }
#else
            core::topology::BaseMeshTopology::Cube c = cubeCorners[e];
#endif

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

void FFDDistanceGridCollisionModel::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!isActive()) return;
    if (vparams->displayFlags().getShowCollisionModels())
    {
        if (vparams->displayFlags().getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDisable(GL_LIGHTING);
        glColor4fv(getColor4f());
        for (unsigned int i=0; i<elems.size(); i++)
        {
            draw(vparams,i);
        }
        if (vparams->displayFlags().getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (getPrevious()!=NULL)
        getPrevious()->draw(vparams);
#endif /* SOFA_NO_OPENGL */
}

void FFDDistanceGridCollisionModel::draw(const core::visual::VisualParams* vparams,int index)
{
#ifndef SOFA_NO_OPENGL
    //DistanceGrid* grid = getGrid(index);
    DeformedCube& cube = getDeformCube( index );
    //glEnable(GL_BLEND);
    //glDepthMask(0);
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
        helper::gl::glVertexT(cube.corners[0]); helper::gl::glVertexT(cube.corners[4]);
        helper::gl::glVertexT(cube.corners[1]); helper::gl::glVertexT(cube.corners[5]);
        helper::gl::glVertexT(cube.corners[2]); helper::gl::glVertexT(cube.corners[6]);
        helper::gl::glVertexT(cube.corners[3]); helper::gl::glVertexT(cube.corners[7]);
        helper::gl::glVertexT(cube.corners[0]); helper::gl::glVertexT(cube.corners[2]);
        helper::gl::glVertexT(cube.corners[1]); helper::gl::glVertexT(cube.corners[3]);
        helper::gl::glVertexT(cube.corners[4]); helper::gl::glVertexT(cube.corners[6]);
        helper::gl::glVertexT(cube.corners[5]); helper::gl::glVertexT(cube.corners[7]);
        helper::gl::glVertexT(cube.corners[0]); helper::gl::glVertexT(cube.corners[1]);
        helper::gl::glVertexT(cube.corners[2]); helper::gl::glVertexT(cube.corners[3]);
        helper::gl::glVertexT(cube.corners[4]); helper::gl::glVertexT(cube.corners[5]);
        helper::gl::glVertexT(cube.corners[6]); helper::gl::glVertexT(cube.corners[7]);
    }
    glEnd();
    glLineWidth(2);
    glPointSize(5);
    {
        /*
            for (int j=0; j<3; j++)
            {
                glBegin(GL_LINE_STRIP);
                for (int r=0;r<=16;r++)
                {
                    SReal c = cube.radius*(SReal)cos(r*M_PI/8);
                    SReal s = cube.radius*(SReal)sin(r*M_PI/8);
                    sofa::defaulttype::Vec<3, SReal> p = cube.center;
                    p[j] += c;
                    p[(j+1)%3] += s;
                    helper::gl::glVertexT(p);
                }
                glEnd();
            }
        */
        glBegin(GL_POINTS);
        {
            helper::gl::glVertexT(cube.center);

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
            helper::gl::glVertexT(cube.deformedPoints[j]);
        glEnd();
        if (vparams->displayFlags().getShowNormals())
        {
            glBegin(GL_LINES);
            for (unsigned int j=0; j<cube.deformedNormals.size(); j++)
            {
                helper::gl::glVertexT(cube.deformedPoints[j]);
                helper::gl::glVertexT(cube.deformedPoints[j] + cube.deformedNormals[j]);
            }
            glEnd();
        }
    }
    glPointSize(1);
#endif /* SOFA_NO_OPENGL */
}

//template <class DataTypes>
//typename ContactMapper<RigidDistanceGridCollisionModel,DataTypes>::MMechanicalState* ContactMapper<RigidDistanceGridCollisionModel,DataTypes>::createMapping(const char* name)
//{
//	using sofa::component::mapping::IdentityMapping;
//
//    MMechanicalState* outmodel = Inherit::createMapping(name);
//    if (this->child!=NULL && this->mapping==NULL)
//    {
//        // add velocity visualization
///*        sofa::component::visualmodel::DrawV* visu = new sofa::component::visualmodel::DrawV;
//        this->child->addObject(visu);
//        visu->useAlpha.setValue(true);
//        visu->vscale.setValue(this->model->getContext()->getDt());
//        IdentityMapping< DataTypes, ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > * map = new IdentityMapping< DataTypes, ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > >( outmodel, visu );
//        this->child->addObject(map);
//        visu->init();
//        map->init(); */
//    }
//    return outmodel;
//}


ContactMapperCreator< ContactMapper<FFDDistanceGridCollisionModel> > FFDDistanceGridContactMapperClass("default", true);

template class SOFA_SOFADISTANCEGRID_API ContactMapper<FFDDistanceGridCollisionModel, sofa::defaulttype::Vec3Types>;


ContactMapperCreator< ContactMapper<RigidDistanceGridCollisionModel> > DistanceGridContactMapperClass("default", true);

template class SOFA_SOFADISTANCEGRID_API ContactMapper<RigidDistanceGridCollisionModel, sofa::defaulttype::Vec3Types>;

} // namespace collision

} // namespace component

} // namespace sofa

