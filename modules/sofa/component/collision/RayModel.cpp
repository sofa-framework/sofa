#include <sofa/component/collision/RayModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <GL/glut.h>


namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Ray)

int RayModelClass = core::RegisterObject("TODO")
        .add< RayModel >()
        .addAlias("Ray")
        ;


using namespace sofa::defaulttype;

RayModel::RayModel(double length)
    : defaultLength(dataField(&defaultLength, length, "", "TODO"))
{
}

void RayModel::resize(int size)
{
    this->component::MechanicalObject<Vec3Types>::resize(size);
    this->core::CollisionModel::resize(size/2);
    if ((int)length.size() < size/2)
    {
        length.reserve(size/2);
        while ((int)length.size() < size/2)
            length.push_back(defaultLength.getValue());
    }
    else
    {
        length.resize(size/2);
    }
}

int RayModel::addRay(const Vector3& origin, const Vector3& direction, double length)
{
    int i = size;
    resize(2*(i+1));
    Ray r = getRay(i);
    r.origin() = origin;
    r.direction() = direction;
    r.l() = length;
    return i;
}

void RayModel::draw(int index)
{
    Ray r(this, index);
    const Vector3& p1 = r.origin();
    const Vector3 p2 = p1 + r.direction()*r.l();
    glBegin(GL_LINES);
    glVertex3dv(p1.ptr());
    glVertex3dv(p2.ptr());
    glEnd();
}

void RayModel::draw()
{
    if (isActive() && getContext()->getShowCollisionModels())
    {
        glDisable(GL_LIGHTING);
        if (isStatic())
            glColor3f(0.5, 0.5, 0.5);
        else
            glColor3f(1.0, 0.0, 0.0);
        for (int i=0; i<size; i++)
        {
            draw(i);
        }
    }
    if (isActive() && getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels() && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

void RayModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (isStatic() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Ray r(this, i);
            const Vector3& o = r.origin();
            const Vector3& d = r.direction();
            const double l = r.l();
            for (int c=0; c<3; c++)
            {
                if (d[c]<0)
                {
                    minElem[c] = o[c] + d[c]*l;
                    maxElem[c] = o[c];
                }
                else
                {
                    minElem[c] = o[c];
                    maxElem[c] = o[c] + d[c]*l;
                }
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void RayModel::applyTranslation(double dx, double dy, double dz)
{
    Vector3 d(dx,dy,dz);
    for (int i = 0; i < getNbRay(); i++)
        getRay(i).origin() += d;
}

} // namespace collision

} // namespace component

} // namespace sofa

