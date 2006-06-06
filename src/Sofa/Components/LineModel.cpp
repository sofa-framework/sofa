#include "LineModel.h"
#include "CubeModel.h"
#include "Line.h"
#include "Sofa/Abstract/CollisionElement.h"
#include "Common/ObjectFactory.h"
#include <vector>
#include <GL/gl.h>

namespace Sofa
{
namespace Components
{

SOFA_DECL_CLASS(Line)

void create(LineModel*& obj, ObjectDescription* /*arg*/)
{
    obj = new LineModel;
}

Creator< ObjectFactory, LineModel > LineModelClass("Line");

LineModel::LineModel()
{
    mmodel = NULL;
    mesh = NULL;
    previous = NULL;
    next = NULL;
    static_ = false;
}

LineModel::~LineModel()
{
}

void LineModel::init()
{
    mmodel = dynamic_cast< Core::MechanicalModel<Vec3Types>* > (getContext()->getMechanicalModel());
    mesh = dynamic_cast< MeshTopology* > (getContext()->getTopology());

    if (mmodel==NULL)
    {
        std::cerr << "ERROR: LineModel requires a Vec3 Mechanical Model.\n";
        return;
    }

    if (mesh==NULL)
    {
        std::cerr << "ERROR: LineModel requires a Mesh Topology.\n";
        return;
    }

    elems.clear();
    const int npoints = mmodel->getX()->size();
    const int nlines = mesh->getNbLines();
    elems.reserve(nlines);
    //VecCoord& x = *mmodel->getX();
    //VecDeriv& v = *mmodel->getV();
    for (int i=0; i<nlines; i++)
    {
        MeshTopology::Line idx = mesh->getLine(i);
        if (idx[0] >= npoints || idx[1] >= npoints)
        {
            std::cerr << "ERROR: Out of range index in Line "<<i<<": "<<idx[0]<<" "<<idx[1]<<" ( total points="<<npoints<<")\n";
            continue;
        }
        Line *t = new Line(i, idx[0], idx[1], this);
        elems.push_back(t);
    }
}

void LineModel::draw()
{
    if (!isActive() || !getContext()->getShowCollisionModels()) return;
    glDisable(GL_LIGHTING);
    if (isStatic())
        glColor3f(0.5, 0.5, 0.5);
    else
        glColor3f(1.0, 0.0, 0.0);
    for (unsigned int i=0; i<elems.size(); i++)
    {
        static_cast<Line*>(elems[i])->draw();
    }
    if (getPrevious()!=NULL && dynamic_cast<Abstract::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<Abstract::VisualModel*>(getPrevious())->draw();
}

void LineModel::computeContinuousBoundingBox (double dt)
{
    CubeModel* cubeModel = dynamic_cast<CubeModel*>(getPrevious());

    if (cubeModel == NULL)
    {
        if (getPrevious() != NULL)
        {
            delete getPrevious();
            setPrevious(NULL);
        }
        cubeModel = new CubeModel();
        cubeModel->setContext(getContext());
        this->setPrevious(cubeModel);
        cubeModel->setNext(this);
    }
    else if (isStatic()) return; // No need to recompute BBox if immobile

    std::vector<Vector3> newVertices;

    std::vector<Vector3> &verts = *(mmodel->getX());
    std::vector<Vector3> &velocityVerts = *(mmodel->getV());

    int size = verts.size();
    newVertices.reserve(size);

    for (int i = 0; i < size; i++)
    {
        Vector3 newPos = verts[i];
        newPos += velocityVerts[i] * dt; //getContext()->getDt();
        newVertices.push_back(newPos);
    }
    Vector3 minBB, maxBB, minBBMoved, maxBBMoved;
    findBoundingBox(verts, minBB, maxBB);
    findBoundingBox(newVertices, minBBMoved, maxBBMoved);

    // get the min max vector with minBB, minBBMoved, maxBB, maxBBMoved
    for (int i = 0; i < 3; i++)
    {
        minBB[i] = (minBB[i] > minBBMoved[i]) ? minBBMoved[i] : minBB[i];
        maxBB[i] = (maxBB[i] > maxBBMoved[i]) ? maxBBMoved[i] : maxBB[i];
    }

    cubeModel->setCube(0,minBB, maxBB);

    for (unsigned int i=0; i<elems.size(); i++)
        static_cast<Line*>(elems[i])->recalcContinuousBBox(dt);
}


void LineModel::computeBoundingBox(void)
{
    CubeModel* cubeModel = dynamic_cast<CubeModel*>(getPrevious());

    if (cubeModel == NULL)
    {
        if (getPrevious() != NULL)
        {
            delete getPrevious();
            setPrevious(NULL);
        }
        cubeModel = new CubeModel();
        cubeModel->setContext(getContext());
        this->setPrevious(cubeModel);
        cubeModel->setNext(this);
    }
    else if (isStatic()) return; // No need to recompute BBox if immobile

    Vector3 minBB, maxBB;

    findBoundingBox(*(mmodel->getX()), minBB, maxBB);

    //std::cout << "BBox: <"<<minBB[0]<<','<<minBB[1]<<','<<minBB[2]<<">-<"<<maxBB[0]<<','<<maxBB[1]<<','<<maxBB[2]<<">\n";

    cubeModel->setCube(0,minBB, maxBB);

    for (unsigned int i=0; i<elems.size(); i++)
        static_cast<Line*>(elems[i])->recalcBBox();
}

void LineModel::findBoundingBox(const std::vector<Vector3> &verts, Vector3 &minBB, Vector3 &maxBB)
{
    //std::vector<Vector3*>::const_iterator it = points.begin();
    //std::vector<Vector3*>::const_iterator itEnd = points.end();
    //std::vector<Vector3>* verts = this->getX();

    std::vector<Vector3>::const_iterator it = verts.begin();
    std::vector<Vector3>::const_iterator itEnd = verts.end();

    if (it != itEnd)
    {
        minBB = *it;
        maxBB = *it;
        it++;
    }

    for (; it != itEnd; it++)
    {
        minBB.x() =  (minBB.x() > (*it).x()) ? (*it).x() : minBB.x();
        minBB.y() =  (minBB.y() > (*it).y()) ? (*it).y() : minBB.y();
        minBB.z() =  (minBB.z() > (*it).z()) ? (*it).z() : minBB.z();

        maxBB.x() =  (maxBB.x() < (*it).x()) ? (*it).x() : maxBB.x();
        maxBB.y() =  (maxBB.y() < (*it).y()) ? (*it).y() : maxBB.y();
        maxBB.z() =  (maxBB.z() < (*it).z()) ? (*it).z() : maxBB.z();
    }
}

} // namespace Components

} // namespace Sofa

