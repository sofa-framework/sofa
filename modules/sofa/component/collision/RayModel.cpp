/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/collision/RayModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/glut.h>


namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Ray)

int RayModelClass = core::RegisterObject("Collision model representing a ray in space, e.g. a mouse click")
        .add< RayModel >()
        .addAlias("Ray")
        ;


using namespace sofa::defaulttype;

RayModel::RayModel(Real length)
    : defaultLength(initData(&defaultLength, length, "", "TODO"))
{
    this->contactResponse.setValue("ray"); // use RayContact response class
}

void RayModel::resize(int size)
{
    this->component::container::MechanicalObject<Vec3Types>::resize(size);
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

int RayModel::addRay(const Vector3& origin, const Vector3& direction, Real length)
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
    helper::gl::glVertexT(p1);
    helper::gl::glVertexT(p2);
    glEnd();
}

void RayModel::draw()
{
    if (getContext()->getShowCollisionModels())
    {
        glDisable(GL_LIGHTING);
        glColor4fv(getColor4f());
        for (int i=0; i<size; i++)
        {
            draw(i);
        }
    }
    if (getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels())
        getPrevious()->draw();
}

void RayModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (!isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Ray r(this, i);
            const Vector3& o = r.origin();
            const Vector3& d = r.direction();
            const Real l = r.l();
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

