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
#pragma once
#include <sofa/component/collision/geometry/PointModel.h>

#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/collision/geometry/CubeModel.h>

namespace sofa::component::collision::geometry
{

template<class DataTypes>
PointCollisionModel<DataTypes>::PointCollisionModel()
    : bothSide(initData(&bothSide, false, "bothSide", "activate collision on both side of the point model (when surface normals are defined on these points)") )
    , mstate(nullptr)
    , computeNormals( initData(&computeNormals, false, "computeNormals", "activate computation of normal vectors (required for some collision detection algorithms)") )
    , m_displayFreePosition(initData(&m_displayFreePosition, false, "displayFreePosition", "Display Collision Model Points free position(in green)") )
    , l_topology(initLink("topology", "link to the topology container"))
{
    enum_type = POINT_TYPE;
}

template<class DataTypes>
void PointCollisionModel<DataTypes>::resize(sofa::Size size)
{
    this->core::CollisionModel::resize(size);
}

template<class DataTypes>
void PointCollisionModel<DataTypes>::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());

    if (mstate==nullptr)
    {
        msg_error() << "PointModel requires a Vec3 Mechanical Model";
        return;
    }

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    const int npoints = mstate->getSize();
    resize(npoints);
    if (computeNormals.getValue()) updateNormals();
}


template<class DataTypes>
bool PointCollisionModel<DataTypes>::canCollideWithElement(sofa::Index index, CollisionModel* model2, sofa::Index index2)
{

    if (!this->bSelfCollision.getValue()) return true; // we need to perform this verification process only for the selfcollision case.
    if (this->getContext() != model2->getContext()) return true;

    if (model2 == this)
    {

        if (index<=index2) // to avoid to have two times the same auto-collision we only consider the case when index > index2
            return false;

        sofa::core::topology::BaseMeshTopology* topology = l_topology.get();

        // in the neighborhood, if we find a point in common, we cancel the collision
        const auto& verticesAroundVertex1 =topology->getVerticesAroundVertex(index);
        const auto& verticesAroundVertex2 =topology->getVerticesAroundVertex(index2);

        for (sofa::Index i1=0; i1<verticesAroundVertex1.size(); i1++)
        {
            const sofa::Index v1 = verticesAroundVertex1[i1];

            for (sofa::Index i2=0; i2<verticesAroundVertex2.size(); i2++)
            {

                if (v1 == verticesAroundVertex2[i2] || v1 == index2 || index == verticesAroundVertex2[i2])
                {
                    return false;
                }
            }
        }
        return true;
    }
    else
        return model2->canCollideWithElement(index2, this, index);
}

template<class DataTypes>
void PointCollisionModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();
    const auto npoints = mstate->getSize();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
    }
    if (updated) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    if (computeNormals.getValue()) updateNormals();

    cubeModel->resize(size);
    if (!empty())
    {
        //VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
        const SReal distance = this->proximity.getValue();
        for (sofa::Size i=0; i<size; i++)
        {
            TPoint<DataTypes> p(this,i);
            const type::Vec3& pt = p.p();
            cubeModel->setParentOf(i, pt - type::Vec3(distance,distance,distance), pt + type::Vec3(distance,distance,distance));
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template<class DataTypes>
void PointCollisionModel<DataTypes>::computeContinuousBoundingTree(SReal dt, int maxDepth)
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();
    const auto npoints = mstate->getSize();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
    }
    if (!isMoving() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    if (computeNormals.getValue()) updateNormals();

    type::Vec3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        //VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
        //VecDeriv& v = mstate->read(core::ConstVecDerivId::velocity())->getValue();
        const SReal distance = (SReal)this->proximity.getValue();
        for (sofa::Size i=0; i<size; i++)
        {
            TPoint<DataTypes> p(this,i);
            const type::Vec3& pt = p.p();
            const type::Vec3 ptv = pt + p.v()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt[c];
                maxElem[c] = pt[c];
                if (ptv[c] > maxElem[c]) maxElem[c] = ptv[c];
                else if (ptv[c] < minElem[c]) minElem[c] = ptv[c];
                minElem[c] -= distance;
                maxElem[c] += distance;
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template<class DataTypes>
void PointCollisionModel<DataTypes>::updateNormals()
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    auto n = x.size();
    normals.resize(n);
    for (sofa::Index i=0; i<n; ++i)
    {
        normals[i].clear();
    }
    core::topology::BaseMeshTopology* mesh = l_topology.get();
    if (mesh->getNbTetrahedra()+mesh->getNbHexahedra() > 0)
    {
        if (mesh->getNbTetrahedra()>0)
        {
            const core::topology::BaseMeshTopology::SeqTetrahedra &elems = mesh->getTetrahedra();
            for (sofa::Index i=0; i < elems.size(); ++i)
            {
                const core::topology::BaseMeshTopology::Tetra &e = elems[i];
                const Coord& p1 = x[e[0]];
                const Coord& p2 = x[e[1]];
                const Coord& p3 = x[e[2]];
                const Coord& p4 = x[e[3]];
                Coord& n1 = normals[e[0]];
                Coord& n2 = normals[e[1]];
                Coord& n3 = normals[e[2]];
                Coord& n4 = normals[e[3]];
                Coord n;
                n = cross(p3-p1,p2-p1); n.normalize();
                n1 += n;
                n2 += n;
                n3 += n;
                n = cross(p4-p1,p3-p1); n.normalize();
                n1 += n;
                n3 += n;
                n4 += n;
                n = cross(p2-p1,p4-p1); n.normalize();
                n1 += n;
                n4 += n;
                n2 += n;
                n = cross(p3-p2,p4-p2); n.normalize();
                n2 += n;
                n4 += n;
                n3 += n;
            }
        }
        /// @todo Hexahedra
    }
    else if (mesh->getNbTriangles()+mesh->getNbQuads() > 0)
    {
        if (mesh->getNbTriangles()>0)
        {
            const core::topology::BaseMeshTopology::SeqTriangles &elems = mesh->getTriangles();
            for (sofa::Index i=0; i < elems.size(); ++i)
            {
                const core::topology::BaseMeshTopology::Triangle &e = elems[i];
                const Coord& p1 = x[e[0]];
                const Coord& p2 = x[e[1]];
                const Coord& p3 = x[e[2]];
                Coord& n1 = normals[e[0]];
                Coord& n2 = normals[e[1]];
                Coord& n3 = normals[e[2]];
                Coord n;
                n = cross(p2-p1,p3-p1); n.normalize();
                n1 += n;
                n2 += n;
                n3 += n;
            }
        }
        if (mesh->getNbQuads()>0)
        {
            const core::topology::BaseMeshTopology::SeqQuads &elems = mesh->getQuads();
            for (sofa::Index i=0; i < elems.size(); ++i)
            {
                const core::topology::BaseMeshTopology::Quad &e = elems[i];
                const Coord& p1 = x[e[0]];
                const Coord& p2 = x[e[1]];
                const Coord& p3 = x[e[2]];
                const Coord& p4 = x[e[3]];
                Coord& n1 = normals[e[0]];
                Coord& n2 = normals[e[1]];
                Coord& n3 = normals[e[2]];
                Coord& n4 = normals[e[3]];
                Coord n;
                n = cross(p3-p1,p4-p2); n.normalize();
                n1 += n;
                n2 += n;
                n3 += n;
                n4 += n;
            }
        }
    }
    for (sofa::Index i=0; i<n; ++i)
    {
        const SReal l = normals[i].norm();
        if (l > 1.0e-3)
            normals[i] *= 1/l;
        else
            normals[i].clear();
    }
}

template<class DataTypes>
void PointCollisionModel<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if( !onlyVisible ) return;

    const auto npoints = mstate->getSize();
    if (npoints != size)
        return;

    static constexpr Real max_real = std::numeric_limits<Real>::max();
    static constexpr Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};

    for (sofa::Size i=0; i<size; i++)
    {
        Element e(this,i);
        const Coord& p = e.p();

        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = (Real)p[c];
            else if (p[c] < minBBox[c]) minBBox[c] = (Real)p[c];
        }
    }

    this->f_bbox.setValue(sofa::type::TBoundingBox<Real>(minBBox,maxBBox));
}



template<class DataTypes>
void PointCollisionModel<DataTypes>::draw(const core::visual::VisualParams*, sofa::Index index)
{
    SOFA_UNUSED(index);
    //TODO(fred roy 2018-06-21)...please implement.
}


template<class DataTypes>
void PointCollisionModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {
        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, true);

        // Check topological modifications
        const auto npoints = mstate->getSize();
        if (npoints != size)
            return;

        std::vector< type::Vec3 > pointsP;
        std::vector< type::Vec3 > pointsL;
        for (sofa::Size i = 0; i < size; i++)
        {
            TPoint<DataTypes> p(this, i);
            if (p.isActive())
            {
                pointsP.push_back(p.p());
                if (i < sofa::Size(normals.size()))
                {
                    pointsL.push_back(p.p());
                    pointsL.push_back(p.p() + normals[i] * 0.1f);
                }
            }
        }

        const auto c = getColor4f();
        vparams->drawTool()->drawPoints(pointsP, 3, sofa::type::RGBAColor(c[0], c[1], c[2], c[3]));
        vparams->drawTool()->drawLines(pointsL, 1, sofa::type::RGBAColor(c[0], c[1], c[2], c[3]));

        if (m_displayFreePosition.getValue())
        {
            std::vector< type::Vec3 > pointsPFree;

            for (sofa::Size i = 0; i < size; i++)
            {
                TPoint<DataTypes> p(this, i);
                if (p.isActive())
                {
                    pointsPFree.push_back(p.pFree());
                }
            }

            vparams->drawTool()->drawPoints(pointsPFree, 3, sofa::type::RGBAColor(0.0f, 1.0f, 0.2f, 1.0f));
        }

        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, false);
    }

    if (getPrevious() != nullptr && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
}


} //namespace sofa::component::collision::geometry
