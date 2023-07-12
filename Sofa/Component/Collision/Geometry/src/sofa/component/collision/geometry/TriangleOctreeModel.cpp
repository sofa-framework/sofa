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
#include <sofa/component/collision/geometry/TriangleModel.inl>
#include <sofa/component/collision/geometry/TriangleOctreeModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology
{

typedef core::topology::BaseMeshTopology::Triangle	Triangle;
} // namespace sofa::component::topology

namespace sofa::component::collision::geometry
{

int TriangleOctreeModelClass =	core::RegisterObject ("collision model using a triangular mesh mapped to an Octree").add <	TriangleOctreeModel > ().addAlias ("TriangleOctree");

TriangleOctreeModel::TriangleOctreeModel ()
{
}

void TriangleOctreeModel::draw (const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    TriangleCollisionModel<sofa::defaulttype::Vec3Types>::draw(vparams);
    if (isActive () && vparams->displayFlags().getShowCollisionModels ())
    {
        if (vparams->displayFlags().getShowWireFrame ())
            vparams->drawTool()->setPolygonMode(0, true);

        vparams->drawTool()->enableLighting();
        const float* getCol = getColor4f();
        const auto color = sofa::type::RGBAColor(getCol[0], getCol[1], getCol[2], getCol[3]);
        vparams->drawTool()->setMaterial(color);

        if(octreeRoot)
            octreeRoot->draw(vparams->drawTool());

        vparams->drawTool()->disableLighting();
        if (vparams->displayFlags().getShowWireFrame ())
            vparams->drawTool()->setPolygonMode(0, false);
    }


}

void TriangleOctreeModel::computeBoundingTree(int maxDepth)
{
    const type::vector<topology::Triangle>& tri = *m_triangles;
    if(octreeRoot)
    {
        delete octreeRoot;
        octreeRoot=nullptr;
    }

    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();
    updateFromTopology();

    if (!isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile
    const std::size_t size2=m_mstate->getSize();
    pNorms.resize(size2);
    for(sofa::Size i=0; i<size2; i++)
    {
        pNorms[i]=type::Vec3(0,0,0);
    }
    type::Vec3 minElem, maxElem;
    maxElem[0]=minElem[0]=m_mstate->read(core::ConstVecCoordId::position())->getValue()[0][0];
    maxElem[1]=minElem[1]=m_mstate->read(core::ConstVecCoordId::position())->getValue()[0][1];
    maxElem[2]=minElem[2]=m_mstate->read(core::ConstVecCoordId::position())->getValue()[0][2];

    cubeModel->resize(1);  // size = number of triangles
    for (std::size_t i=1; i<size; i++)
    {
        Triangle t(this,i);
        pNorms[tri[i][0]]+=t.n();
        pNorms[tri[i][1]]+=t.n();
        pNorms[tri[i][2]]+=t.n();
        const sofa::type::Vec3* pt[3];
        pt[0] = &t.p1();
        pt[1] = &t.p2();
        pt[2] = &t.p3();
        t.n() = cross(*pt[1]-*pt[0],*pt[2]-*pt[0]);
        t.n().normalize();

        for (int p=0; p<3; p++)
        {


            for(int c=0; c<3; c++)
            {
                if ((*pt[p])[c] > maxElem[c]) maxElem[c] = (*pt[p])[c];
                if ((*pt[p])[c] < minElem[c]) minElem[c] = (*pt[p])[c];

            }
        }

    }

    cubeModel->setParentOf(0, minElem, maxElem); // define the bounding box of the current triangle
    cubeModel->computeBoundingTree(maxDepth);
    for(sofa::Size i=0; i<size2; i++)
    {
        pNorms[i].normalize();
    }
}

void TriangleOctreeModel::computeContinuousBoundingTree(SReal/* dt*/, int maxDepth)
{
    computeBoundingTree(maxDepth);
}

void TriangleOctreeModel::buildOctree()
{
    this->octreeTriangles = &this->getTriangles();
    this->octreePos = &this->getX();
    TriangleOctreeRoot::buildOctree();
}

} // namespace sofa::component::collision::geometry
