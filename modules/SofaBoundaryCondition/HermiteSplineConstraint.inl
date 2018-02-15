/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_HERMITESPLINECONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_HERMITESPLINECONSTRAINT_INL

#include <SofaBoundaryCondition/HermiteSplineConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <SofaBaseTopology/TopologySubsetData.inl>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{


template <class DataTypes>
HermiteSplineConstraint<DataTypes>::HermiteSplineConstraint()
    :core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , m_indices( initData(&m_indices,"indices","Indices of the constrained points") )
    , m_tBegin(initData(&m_tBegin,"BeginTime","Begin Time of the motion") )
    , m_tEnd(initData(&m_tEnd,"EndTime","End Time of the motion") )
    , m_x0(initData(&m_x0,"X0","first control point") )
    , m_dx0(initData(&m_dx0,"dX0","first control tangente") )
    , m_x1(initData(&m_x1,"X1","second control point") )
    , m_dx1(initData(&m_dx1,"dX1","second control tangente") )
    , m_sx0(initData(&m_sx0,"SX0","first interpolation vector") )
    , m_sx1(initData(&m_sx1,"SX1","second interpolation vector") )
{
}


template <class DataTypes>
HermiteSplineConstraint<DataTypes>::HermiteSplineConstraint(core::behavior::MechanicalState<DataTypes>* mstate)
    : core::behavior::ProjectiveConstraintSet<DataTypes>(mstate)
    , m_indices( initData(&m_indices,"indices","Indices of the constrained points") )
    , m_tBegin(initData(&m_tBegin,"BeginTime","Begin Time of the motion") )
    , m_tEnd(initData(&m_tEnd,"EndTime","End Time of the motion") )
    , m_x0(initData(&m_x0,"X0","first control point") )
    , m_dx0(initData(&m_dx0,"dX0","first control tangente") )
    , m_x1(initData(&m_x1,"X1","second control point") )
    , m_dx1(initData(&m_dx1,"dX1","sceond control tangente") )
    , m_sx0(initData(&m_sx0,"SX0","first interpolation vector") )
    , m_sx1(initData(&m_sx1,"SX1","second interpolation vector") )
{
}

template <class DataTypes>
HermiteSplineConstraint<DataTypes>::~HermiteSplineConstraint()
{
}

template <class DataTypes>
void HermiteSplineConstraint<DataTypes>::clearConstraints()
{
    m_indices.beginEdit()->clear();
    m_indices.endEdit();
}

template <class DataTypes>
void  HermiteSplineConstraint<DataTypes>::addConstraint(unsigned index)
{
    m_indices.beginEdit()->push_back(index);
    m_indices.endEdit();
}


template <class DataTypes>
void HermiteSplineConstraint<DataTypes>::init()
{
    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters for topology data and handler
    m_indices.createTopologicalEngine(topology);
    m_indices.registerTopologicalData();

    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();
}

template <class DataTypes>
void HermiteSplineConstraint<DataTypes>::reinit()
{
    init();
}


template <class DataTypes>
void HermiteSplineConstraint<DataTypes>::computeHermiteCoefs( const Real u, Real &H00, Real &H10, Real &H01, Real &H11)
{
    //-- time interpolation --> acceleration is itself computed from hemite
    Real u2 = u*u;
    Real u3 = u*u*u;
    //Real uH00 = 2*u3 -3*u2 +1 ;		//hermite coefs
    Real uH10 = u3 -2*u2 +u;
    Real uH01 = -2*u3 + 3*u2;
    Real uH11 = u3 -u2;
    Vec2R pu = m_sx0.getValue()*uH10 + Vec2R(1,1)*uH01 + m_sx1.getValue()*uH11;
    Real su = pu.y();

    Real su2 = su*su;
    Real su3 = su*su*su;
    H00 = 2*su3 -3*su2 +1 ;
    H10 = su3 -2*su2 +su;
    H01 = -2*su3 + 3*su2;
    H11 = su3 -su2;
}

template <class DataTypes>
void HermiteSplineConstraint<DataTypes>::computeDerivateHermiteCoefs( const Real u, Real &dH00, Real &dH10, Real &dH01, Real &dH11)
{
    //-- time interpolation --> acceleration is itself computed from hemite
    Real u2 = u*u;
    Real u3 = u*u*u;
    //Real uH00 = 2*u3 -3*u2 +1 ;		//hermite coefs
    Real uH10 = u3 -2*u2 +u;
    Real uH01 = -2*u3 + 3*u2;
    Real uH11 = u3 -u2;
    Vec2R pu = m_sx0.getValue()*uH10 + Vec2R(1,1)*uH01 + m_sx1.getValue()*uH11;
    Real su = pu.y();

    Real su2 = su*su;
    dH00 = 6*su2 -6*su ;
    dH10 = 3*su2 -4*su +1;
    dH01 = -6*su2 + 6*su;
    dH11 = 3*su2 -2*su;
}


template <class DataTypes> template <class DataDeriv>
void HermiteSplineConstraint<DataTypes>::projectResponseT(const core::MechanicalParams* /*mparams*/, DataDeriv& dx)
{
    Real t = (Real) this->getContext()->getTime();
    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue())
    {
        const SetIndexArray & indices = m_indices.getValue();
        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            dx[*it] = Deriv();
    }
}

template <class DataTypes>
void HermiteSplineConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT(mparams, res.wref());
}

template <class DataTypes>
void HermiteSplineConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> dx = vData;
    Real t = (Real) this->getContext()->getTime();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real DT = m_tEnd.getValue() - m_tBegin.getValue();
        const SetIndexArray & indices = m_indices.getValue();

        t -= m_tBegin.getValue();
        Real u = t/DT;

        Real dH00, dH10, dH01, dH11;
        computeDerivateHermiteCoefs( u, dH00, dH10, dH01, dH11);

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            dx[*it] = m_x0.getValue()*dH00 + m_dx0.getValue()*dH10 + m_x1.getValue()*dH01 + m_dx1.getValue()*dH11;
        }
    }
}

template <class DataTypes>
void HermiteSplineConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    Real t = (Real) this->getContext()->getTime();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real DT = m_tEnd.getValue() - m_tBegin.getValue();
        const SetIndexArray & indices = m_indices.getValue();

        t -= m_tBegin.getValue();
        Real u = t/DT;

        Real H00, H10, H01, H11;
        computeHermiteCoefs( u, H00, H10, H01, H11);

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it] = m_x0.getValue()*H00 + m_dx0.getValue()*H10 + m_x1.getValue()*H01 + m_dx1.getValue()*H11;
        }
    }
}

template <class DataTypes>
void HermiteSplineConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    while (rowIt != rowItEnd)
    {
        projectResponseT<MatrixDerivRowType>(mparams, rowIt.row());
        ++rowIt;
    }
}

template <class DataTypes>
void HermiteSplineConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    Real dt = (Real) this->getContext()->getDt();
    Real DT = m_tEnd.getValue() - m_tBegin.getValue();

    glDisable (GL_LIGHTING);
    glPointSize(2);
    glColor4f (1,0.5,0.5,1);

    glBegin (GL_LINE_STRIP);
    for (Real t=0.0 ; t< DT ; t+= dt)
    {
        Real u = t/DT;

        Real H00, H10, H01, H11;
        computeHermiteCoefs( u, H00, H10, H01, H11);

        Vec3R p = m_x0.getValue()*H00 + m_dx0.getValue()*H10 + m_x1.getValue()*H01 + m_dx1.getValue()*H11;
        helper::gl::glVertexT(p);
    }
    glEnd();


    glColor4f (1,0.0,0.0,1);
    glPointSize(5);
    //display control point
    glBegin(GL_POINTS);
    helper::gl::glVertexT(m_x0.getValue());
    helper::gl::glVertexT(m_x1.getValue());
    glEnd();
    //display control tangeantes
    glBegin(GL_LINES);
    helper::gl::glVertexT(m_x0.getValue());
    helper::gl::glVertexT(m_x0.getValue()+m_dx0.getValue()*0.1);
    helper::gl::glVertexT(m_x1.getValue());
    helper::gl::glVertexT(m_x1.getValue()+m_dx1.getValue()*0.1);
    glEnd();
#endif /* SOFA_NO_OPENGL */
}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif
