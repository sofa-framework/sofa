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

#include <sofa/component/constraint/projective/ParabolicProjectiveConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::constraint::projective
{

template <class DataTypes>
ParabolicProjectiveConstraint<DataTypes>::ParabolicProjectiveConstraint(core::behavior::MechanicalState<DataTypes>* mstate)
    : core::behavior::ProjectiveConstraintSet<DataTypes>(mstate)
    , d_indices(initData(&d_indices, "indices", "Indices of the constrained points") )
    , d_P1(initData(&d_P1, "P1", "first point of the parabol") )
    , d_P2(initData(&d_P2, "P2", "second point of the parabol") )
    , d_P3(initData(&d_P3, "P3", "third point of the parabol") )
    , d_tBegin(initData(&d_tBegin, "BeginTime", "Begin Time of the motion") )
    , d_tEnd(initData(&d_tEnd, "EndTime", "End Time of the motion") )
    , l_topology(initLink("topology", "link to the topology container"))
{
    m_indices.setParent(&d_indices);
    m_P1.setParent(&d_P1);
    m_P2.setParent(&d_P2);
    m_P3.setParent(&d_P3);
    m_tBegin.setParent(&d_tBegin);
    m_tEnd.setParent(&d_tEnd);
}

template <class DataTypes>
ParabolicProjectiveConstraint<DataTypes>::~ParabolicProjectiveConstraint()
{
}

template <class DataTypes>
void  ParabolicProjectiveConstraint<DataTypes>::addConstraint(unsigned index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}


template <class DataTypes>
void ParabolicProjectiveConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (sofa::core::topology::BaseMeshTopology* _topology = l_topology.get())
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        // Initialize functions and parameters for topology data and handler
        d_indices.createTopologyHandler(_topology);
    }
    else
    {
        msg_info() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
    }

    Vec3R P1 = d_P1.getValue();
    Vec3R P2 = d_P2.getValue();
    Vec3R P3 = d_P3.getValue();

    //compute the projection to go in the parabol plan,
    //such as P1 is the origin, P1P3 vector is the x axis, and P1P2 is in the xy plan
    //by the way the computation of the parabol equation is much easier
    if(P1 != P2 && P1 != P3 && P2 != P3)
    {

        Vec3R P1P2 = P2 - P1;
        Vec3R P1P3 = P3 - P1;

        Vec3R ax = P1P3;
        Vec3R az = cross(P1P3, P1P2);
        Vec3R ay = cross(az, ax);
        ax.normalize();
        ay.normalize();
        az.normalize();

        
        type::Mat<3,3,Real> Mrot(ax, ay, az);
        type::Mat<3,3,Real> Mrot2;
        Mrot2.transpose(Mrot);
        m_projection.fromMatrix(Mrot2);
        m_projection.normalize();

        m_locP1 = Vec3R();
        m_locP2 =  m_projection.inverseRotate(P1P2);
        m_locP3 =  m_projection.inverseRotate(P1P3);
    }
}

template <class DataTypes>
void ParabolicProjectiveConstraint<DataTypes>::reinit()
{
    init();
}


template <class DataTypes>
template <class DataDeriv>
void ParabolicProjectiveConstraint<DataTypes>::projectResponseT(DataDeriv& dx,
    const std::function<void(DataDeriv&, const unsigned int)>& clear)
{
    Real t = (Real) this->getContext()->getTime();
    if (t >= d_tBegin.getValue() && t <= d_tEnd.getValue())
    {
        const SetIndexArray & indices = d_indices.getValue();
        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            clear(dx, *it);
    }
}

template <class DataTypes>
void ParabolicProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(res.wref(),[](VecDeriv& dx, const unsigned int index) { dx[index].clear(); });
}

template <class DataTypes>
void ParabolicProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> dx = vData;
    Real t = (Real) this->getContext()->getTime();
    Real dt = (Real) this->getContext()->getDt();

    if (t >= d_tBegin.getValue() && t <= d_tEnd.getValue()	)
    {
        Real relativeTime = (t - d_tBegin.getValue() ) / (d_tEnd.getValue() - d_tBegin.getValue());
        const SetIndexArray & indices = d_indices.getValue();

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            //compute velocity by doing v = dx/dt
            Real pxP = m_locP3.x()*relativeTime;
            Real pyP = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (pxP *pxP) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * pxP;
            relativeTime = (t + dt - d_tBegin.getValue() ) / (d_tEnd.getValue() - d_tBegin.getValue());
            Real pxN = m_locP3.x()*relativeTime;
            Real pyN = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (pxN *pxN) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * pxN;

            Vec3R locVel = Vec3R( (pxN-pxP)/dt, (pyN-pyP)/dt, 0.0);

            Vec3R worldVel = m_projection.rotate(locVel);

            dx[*it] = worldVel;
        }
    }
}

template <class DataTypes>
void ParabolicProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    Real t = (Real) this->getContext()->getTime();

    if (t >= d_tBegin.getValue() && t <= d_tEnd.getValue()	)
    {
        Real relativeTime = (t - d_tBegin.getValue() ) / (d_tEnd.getValue() - d_tBegin.getValue());
        const SetIndexArray & indices = d_indices.getValue();

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            //compute position from the equation of the parabol : Y = -y2/(x3*x2-x2�) * X� + (x3*y2)/(x3*x2-x2�) * X
            //with P1:(0,0,0), P2:(x2,y2,z2), P3:(x3,y3,z3) , projected in parabol plan
            Real px = m_locP3.x()*relativeTime;
            Real py = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (px *px) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * px;
            Vec3R locPos( px , py, 0.0);

            //projection to world coordinates
            Vec3R worldPos = d_P1.getValue() + m_projection.rotate(locPos);

            x[*it] = worldPos;
        }
    }
}

template <class DataTypes>
void ParabolicProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    projectResponseT<MatrixDeriv>(c.wref(), [](MatrixDeriv& res, const unsigned int index) { res.clearColBlock(index); });
}


template <class DataTypes>
void ParabolicProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    Real dt = (Real) this->getContext()->getDt();
    Real t = d_tEnd.getValue() - d_tBegin.getValue();
    Real nbStep = t/dt;

    vparams->drawTool()->disableLighting();
    constexpr sofa::type::RGBAColor color(1, 0.5, 0.5, 1);
    std::vector<sofa::type::Vec3> vertices;

    for (unsigned int i=0 ; i< nbStep ; i++)
    {
        //draw lines between each step of the parabolic trajectory
        //so, the smaller is dt, the finer is the parabol
        Real relativeTime = i/nbStep;
        Real px = m_locP3.x()*relativeTime;
        Real py = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (px *px) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * px;
        Vec3R locPos( px , py, 0.0);
        Vec3R worldPos = d_P1.getValue() + m_projection.rotate(locPos);

        vertices.push_back(sofa::type::Vec3(worldPos[0],worldPos[1],worldPos[2]));

        relativeTime = (i+1)/nbStep;
        px = m_locP3.x()*relativeTime;
        py = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (px *px) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * px;
        locPos = Vec3R( px , py, 0.0);
        worldPos = d_P1.getValue() + m_projection.rotate(locPos);


        vertices.push_back(sofa::type::Vec3(worldPos[0],worldPos[1],worldPos[2]));

    }
    vparams->drawTool()->drawLines(vertices, 1.0, color);
    vertices.clear();

    //draw points for the 3 control points
    const Vec3R& mp1 = d_P1.getValue();
    const Vec3R& mp2 = d_P2.getValue();
    const Vec3R& mp3 = d_P3.getValue();
    vertices.push_back(sofa::type::Vec3(mp1[0],mp1[1],mp1[2]));
    vertices.push_back(sofa::type::Vec3(mp2[0],mp2[1],mp2[2]));
    vertices.push_back(sofa::type::Vec3(mp3[0],mp3[1],mp3[2]));

    vparams->drawTool()->drawPoints(vertices, 5.0, color);



}

} // namespace sofa::component::constraint::projective
