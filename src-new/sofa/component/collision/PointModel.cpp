#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/Point.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <vector>
#include <GL/gl.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Point)

void create(PointModel*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    obj = new PointModel;
    if (obj!=NULL)
    {
        obj->setStatic(atoi(arg->getAttribute("static","0"))!=0);
    }
}

Creator<simulation::tree::xml::ObjectFactory, PointModel > PointModelClass("Point");

PointModel::PointModel()
    : static_(false), mstate(NULL)
{
}

void PointModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
}

void PointModel::init()
{
    mstate = dynamic_cast< core::componentmodel::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());

    if (mstate==NULL)
    {
        std::cerr << "ERROR: PointModel requires a Vec3 Mechanical Model.\n";
        return;
    }

    const int npoints = mstate->getX()->size();
    resize(npoints);
}

void PointModel::draw(int index)
{
    Point t(this,index);
    glBegin(GL_POINTS);
    glVertex3dv(t.p().ptr());
    glEnd();
}

void PointModel::draw()
{
    if (0) //isActive() && getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glDisable(GL_LIGHTING);
        glPointSize(3);
        if (isStatic())
            glColor3f(0.5, 0.5, 0.5);
        else
            glColor3f(1.0, 0.0, 0.0);

        for (int i=0; i<size; i++)
        {
            draw(i);
        }

        glColor3f(1.0f, 1.0f, 1.0f);
        glDisable(GL_LIGHTING);
        glPointSize(1);
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    //if (isActive() && getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels() && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
    //	dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

void PointModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getX()->size();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
    }
    if (updated) cubeModel->resize(0);
    if (isStatic() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    cubeModel->resize(size);
    if (!empty())
    {
        //VecCoord& x = *mstate->getX();
        for (int i=0; i<size; i++)
        {
            Point p(this,i);
            const Vector3& pt = p.p();
            cubeModel->setParentOf(i, pt, pt);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void PointModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getX()->size();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
    }
    if (isStatic() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        //VecCoord& x = *mstate->getX();
        //VecDeriv& v = *mstate->getV();
        for (int i=0; i<size; i++)
        {
            Point p(this,i);
            const Vector3& pt = p.p();
            const Vector3 ptv = pt + p.v()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt[c];
                maxElem[c] = pt[c];
                if (ptv[c] > maxElem[c]) maxElem[c] = ptv[c];
                else if (ptv[c] < minElem[c]) minElem[c] = ptv[c];
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

