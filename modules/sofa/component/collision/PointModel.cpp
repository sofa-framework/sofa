/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/system/gl.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Point)

int PointModelClass = core::RegisterObject("Collision model which represents a set of points")
        .add< PointModel >()
        .addAlias("Point")
// .addAlias("PointModel")
        .addAlias("PointMesh")
        .addAlias("PointSet")
        ;

PointModel::PointModel()
    : mstate(NULL)
{
}

void PointModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
}

void PointModel::init()
{
    this->CollisionModel::init();
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
#ifdef SOFA_FLOAT
    glVertex3fv(t.p().ptr());
#else
    glVertex3dv(t.p().ptr());
#endif
    glEnd();
}

void PointModel::draw()
{
    if (getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glDisable(GL_LIGHTING);
        glPointSize(3);
        glColor4fv(getColor4f());

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
    if (getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels() && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

bool PointModel::canCollideWithElement(int index, CollisionModel* model2, int index2)
{
    if (!this->bSelfCollision.getValue()) return true;
    if (this->getContext() != model2->getContext()) return true;
    if (model2 == this)
    {
        //std::cout << "point self test "<<index<<" - "<<index2<<std::endl;
        return index < index2-2; // || index > index2+1;
    }
    else
        return model2->canCollideWithElement(index2, this, index);
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
    if (!isMoving() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

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
    if (!isMoving() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

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

