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

#include <sofa/component/constraint/projective/HermiteSplineProjectiveConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::constraint::projective
{

template <class DataTypes>
HermiteSplineProjectiveConstraint<DataTypes>::HermiteSplineProjectiveConstraint(core::behavior::MechanicalState<DataTypes>* mstate)
    : core::behavior::ProjectiveConstraintSet<DataTypes>(mstate)
    , d_indices(initData(&d_indices, "indices", "Indices of the constrained points") )
    , d_tBegin(initData(&d_tBegin, "BeginTime", "Begin Time of the motion") )
    , d_tEnd(initData(&d_tEnd, "EndTime", "End Time of the motion") )
    , d_x0(initData(&d_x0, "X0", "first control point") )
    , d_dx0(initData(&d_dx0, "dX0", "first control tangente") )
    , d_x1(initData(&d_x1, "X1", "second control point") )
    , d_dx1(initData(&d_dx1, "dX1", "sceond control tangente") )
    , d_sx0(initData(&d_sx0, "SX0", "first interpolation vector") )
    , d_sx1(initData(&d_sx1, "SX1", "second interpolation vector") )
    , l_topology(initLink("topology", "link to the topology container"))
{
    m_indices.setParent(&d_indices);
    m_tBegin.setParent(&d_tBegin);
    m_tEnd.setParent(&d_tEnd);
    m_x0.setParent(&d_x0);
    m_dx0.setParent(&d_dx0);
    m_x1.setParent(&d_x1);
    m_dx1.setParent(&d_dx1);
    m_sx0.setParent(&d_sx0);
    m_sx1.setParent(&d_sx1);
}

template <class DataTypes>
HermiteSplineProjectiveConstraint<DataTypes>::~HermiteSplineProjectiveConstraint()
{
}

template <class DataTypes>
void HermiteSplineProjectiveConstraint<DataTypes>::clearConstraints()
{
    d_indices.beginEdit()->clear();
    d_indices.endEdit();
}

template <class DataTypes>
void  HermiteSplineProjectiveConstraint<DataTypes>::addConstraint(unsigned index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}


template <class DataTypes>
void HermiteSplineProjectiveConstraint<DataTypes>::init()
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
}

template <class DataTypes>
void HermiteSplineProjectiveConstraint<DataTypes>::reinit()
{
    init();
}


template <class DataTypes>
void HermiteSplineProjectiveConstraint<DataTypes>::computeHermiteCoefs( const Real u, Real &H00, Real &H10, Real &H01, Real &H11)
{
    //-- time interpolation --> acceleration is itself computed from hemite
    Real u2 = u*u;
    Real u3 = u*u*u;
    //Real uH00 = 2*u3 -3*u2 +1 ;		//hermite coefs
    Real uH10 = u3 -2*u2 +u;
    Real uH01 = -2*u3 + 3*u2;
    Real uH11 = u3 -u2;
    Vec2R pu = d_sx0.getValue() * uH10 + Vec2R(1, 1) * uH01 + d_sx1.getValue() * uH11;
    Real su = pu.y();

    Real su2 = su*su;
    Real su3 = su*su*su;
    H00 = 2*su3 -3*su2 +1 ;
    H10 = su3 -2*su2 +su;
    H01 = -2*su3 + 3*su2;
    H11 = su3 -su2;
}

template <class DataTypes>
void HermiteSplineProjectiveConstraint<DataTypes>::computeDerivateHermiteCoefs( const Real u, Real &dH00, Real &dH10, Real &dH01, Real &dH11)
{
    //-- time interpolation --> acceleration is itself computed from hemite
    Real u2 = u*u;
    Real u3 = u*u*u;
    Real uH10 = u3 -2*u2 +u;
    Real uH01 = -2*u3 + 3*u2;
    Real uH11 = u3 -u2;
    Vec2R pu = d_sx0.getValue() * uH10 + Vec2R(1, 1) * uH01 + d_sx1.getValue() * uH11;
    Real su = pu.y();

    Real su2 = su*su;
    dH00 = 6*su2 -6*su ;
    dH10 = 3*su2 -4*su +1;
    dH01 = -6*su2 + 6*su;
    dH11 = 3*su2 -2*su;
}


template <class DataTypes> template <class DataDeriv>
void HermiteSplineProjectiveConstraint<DataTypes>::projectResponseT(DataDeriv& dx,
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
void HermiteSplineProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(res.wref(), [](auto& dx, const unsigned int index) {dx[index].clear();});
}

template <class DataTypes>
void HermiteSplineProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> dx = vData;
    Real t = (Real) this->getContext()->getTime();

    if (t >= d_tBegin.getValue() && t <= d_tEnd.getValue()	)
    {
        Real DT = d_tEnd.getValue() - d_tBegin.getValue();
        const SetIndexArray & indices = d_indices.getValue();

        t -= d_tBegin.getValue();
        Real u = t/DT;

        Real dH00, dH10, dH01, dH11;
        computeDerivateHermiteCoefs( u, dH00, dH10, dH01, dH11);

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            dx[*it] = d_x0.getValue() * dH00 + d_dx0.getValue() * dH10 + d_x1.getValue() * dH01 + d_dx1.getValue() * dH11;
        }
    }
}

template <class DataTypes>
void HermiteSplineProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    Real t = (Real) this->getContext()->getTime();

    if (t >= d_tBegin.getValue() && t <= d_tEnd.getValue()	)
    {
        Real DT = d_tEnd.getValue() - d_tBegin.getValue();
        const SetIndexArray & indices = d_indices.getValue();

        t -= d_tBegin.getValue();
        Real u = t/DT;

        Real H00, H10, H01, H11;
        computeHermiteCoefs( u, H00, H10, H01, H11);

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it] = d_x0.getValue() * H00 + d_dx0.getValue() * H10 + d_x1.getValue() * H01 + d_dx1.getValue() * H11;
        }
    }
}

template <class DataTypes>
void HermiteSplineProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    projectResponseT<MatrixDeriv>(c.wref(), [](MatrixDeriv& res, const unsigned int index) { res.clearColBlock(index); });
}

template <class DataTypes>
void HermiteSplineProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    Real dt = (Real) this->getContext()->getDt();
    Real DT = d_tEnd.getValue() - d_tBegin.getValue();

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    std::vector<sofa::type::Vec3> vertices;
    constexpr sofa::type::RGBAColor color(1, 0.5, 0.5, 1);

    const Vec3R& mx0 = d_x0.getValue();
    const Vec3R& mx1 = d_x0.getValue();
    const Vec3R& mdx0 = d_dx0.getValue();
    const Vec3R& mdx1 = d_dx1.getValue();

    for (Real t=0.0 ; t< DT ; t+= dt)
    {
        Real u = t/DT;

        Real H00, H10, H01, H11;
        computeHermiteCoefs( u, H00, H10, H01, H11);

        Vec3R p = mx0*H00 + mdx0*H10 + mx1*H01 + mdx1*H11;

        sofa::type::Vec3 v(p[0], p[1],p[2]);
        vertices.push_back(v);
    }
    vparams->drawTool()->drawLineStrip(vertices, 2, color);

    vertices.clear();
    vertices.push_back(sofa::type::Vec3(mx0[0], mx0[1], mx0[2]));
    vertices.push_back(sofa::type::Vec3(mx1[0], mx1[1], mx1[2]));

    vparams->drawTool()->drawPoints(vertices, 5.0, sofa::type::RGBAColor::red());

    //display control tangents
    vertices.clear();
    vertices.push_back(mx0);
    vertices.push_back(mx0 + mdx0*0.1);
    vertices.push_back(mx1);
    vertices.push_back(mx1 + mdx1*0.1);

    vparams->drawTool()->drawLines(vertices, 1.0, sofa::type::RGBAColor::red());

}

} // namespace sofa::component::constraint::projective
