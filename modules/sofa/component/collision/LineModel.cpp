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
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/Line.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#if defined (__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Line)

int LineModelClass = core::RegisterObject("Collision model representing a line")
        .add< LineModel >()
        .addAlias("Line")
        ;


LineModel::LineModel()
    : meshRevision(-1), mstate(NULL), mesh(NULL)
{
}

void LineModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
    elems.resize(size);
}

void LineModel::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::componentmodel::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());
    mesh = dynamic_cast< MeshTopology* > (getContext()->getTopology());

    if (mstate==NULL)
    {
        std::cerr << "ERROR: LineModel requires a Vec3 Mechanical Model.\n";
        return;
    }

    if (mesh==NULL)
    {
        std::cerr << "ERROR: LineModel requires a Mesh Topology.\n";
        return;
    }
    updateFromTopology();
}

bool LineModel::updateFromTopology()
{
    int revision = mesh->getRevision();
    if (revision == meshRevision) return false;

    const int npoints = mstate->getX()->size();
    const int nlines = mesh->getNbLines();
    resize(nlines);
    int index = 0;
    //VecCoord& x = *mstate->getX();
    //VecDeriv& v = *mstate->getV();
    for (int i=0; i<nlines; i++)
    {
        MeshTopology::Line idx = mesh->getLine(i);
        if (idx[0] >= npoints || idx[1] >= npoints)
        {
            std::cerr << "ERROR: Out of range index in Line "<<i<<": "<<idx[0]<<" "<<idx[1]<<" ( total points="<<npoints<<")\n";
            continue;
        }
        elems[index].i1 = idx[0];
        elems[index].i2 = idx[1];
        ++index;
    }
    meshRevision = revision;
    return true;
}

void LineModel::draw(int index)
{
    Line t(this,index);
    glBegin(GL_LINES);
    glVertex3dv(t.p1().ptr());
    glVertex3dv(t.p2().ptr());
    glEnd();
}

void LineModel::draw()
{
    if (isActive() && getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glDisable(GL_LIGHTING);
        if (isStatic())
            glColor3f(0.5, 0.5, 0.5);
        else
            glColor3f(1.0, 0.0, 0.0);

        for (int i=0; i<size; i++)
        {
            if (elems[i].i1 < elems[i].i2) // only display non-edge lines
                draw(i);
        }

        glColor3f(1.0f, 1.0f, 1.0f);
        glDisable(GL_LIGHTING);
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (isActive() && getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels() && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

void LineModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    bool updated = updateFromTopology();
    if (updated) cubeModel->resize(0);
    if (isStatic() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Vector3 minElem, maxElem;
            Line l(this,i);
            const Vector3& pt1 = l.p1();
            const Vector3& pt2 = l.p2();

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
            }

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void LineModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    bool updated = updateFromTopology();
    if (updated) cubeModel->resize(0);
    if (isStatic() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Line t(this,i);
            const Vector3& pt1 = t.p1();
            const Vector3& pt2 = t.p2();
            const Vector3 pt1v = pt1 + t.v1()*dt;
            const Vector3 pt2v = pt2 + t.v2()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];

                if (pt1v[c] > maxElem[c]) maxElem[c] = pt1v[c];
                else if (pt1v[c] < minElem[c]) minElem[c] = pt1v[c];
                if (pt2v[c] > maxElem[c]) maxElem[c] = pt2v[c];
                else if (pt2v[c] < minElem[c]) minElem[c] = pt2v[c];
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

