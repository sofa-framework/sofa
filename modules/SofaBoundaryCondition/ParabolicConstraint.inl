/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARABOLICCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARABOLICCONSTRAINT_INL

#include <SofaBoundaryCondition/ParabolicConstraint.h>
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
ParabolicConstraint<DataTypes>::ParabolicConstraint()
    :core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , m_indices( initData(&m_indices,"indices","Indices of the constrained points") )
    , m_P1(initData(&m_P1,"P1","first point of the parabol") )
    , m_P2(initData(&m_P2,"P2","second point of the parabol") )
    , m_P3(initData(&m_P3,"P3","third point of the parabol") )
    , m_tBegin(initData(&m_tBegin,"BeginTime","Begin Time of the motion") )
    , m_tEnd(initData(&m_tEnd,"EndTime","End Time of the motion") )
{
}


template <class DataTypes>
ParabolicConstraint<DataTypes>::ParabolicConstraint(core::behavior::MechanicalState<DataTypes>* mstate)
    : core::behavior::ProjectiveConstraintSet<DataTypes>(mstate)
    , m_indices( initData(&m_indices,"indices","Indices of the constrained points") )
    , m_P1(initData(&m_P1,"P1","first point of the parabol") )
    , m_P2(initData(&m_P2,"P2","second point of the parabol") )
    , m_P3(initData(&m_P3,"P3","third point of the parabol") )
    , m_tBegin(initData(&m_tBegin,"BeginTime","Begin Time of the motion") )
    , m_tEnd(initData(&m_tEnd,"EndTime","End Time of the motion") )
{
}

template <class DataTypes>
ParabolicConstraint<DataTypes>::~ParabolicConstraint()
{
}

template <class DataTypes>
void  ParabolicConstraint<DataTypes>::addConstraint(unsigned index)
{
    m_indices.beginEdit()->push_back(index);
    m_indices.endEdit();
}


template <class DataTypes>
void ParabolicConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters for topology data and handler
    m_indices.createTopologicalEngine(topology);
    m_indices.registerTopologicalData();

    Vec3R P1 = m_P1.getValue();
    Vec3R P2 = m_P2.getValue();
    Vec3R P3 = m_P3.getValue();

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

        
        defaulttype::Mat<3,3,Real> Mrot(ax, ay, az);
        defaulttype::Mat<3,3,Real> Mrot2;
        Mrot2.transpose(Mrot);
        m_projection.fromMatrix(Mrot2);
        m_projection.normalize();

        m_locP1 = Vec3R();
        m_locP2 =  m_projection.inverseRotate(P1P2);
        m_locP3 =  m_projection.inverseRotate(P1P3);
    }
}

template <class DataTypes>
void ParabolicConstraint<DataTypes>::reinit()
{
    init();
}


template <class DataTypes>
template <class DataDeriv>
void ParabolicConstraint<DataTypes>::projectResponseT(const core::MechanicalParams* /*mparams*/, DataDeriv& dx)
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
void ParabolicConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT(mparams, res.wref());
}

template <class DataTypes>
void ParabolicConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> dx = vData;
    Real t = (Real) this->getContext()->getTime();
    Real dt = (Real) this->getContext()->getDt();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real relativeTime = (t - m_tBegin.getValue() ) / (m_tEnd.getValue() - m_tBegin.getValue());
        const SetIndexArray & indices = m_indices.getValue();

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            //compute velocity by doing v = dx/dt
            Real pxP = m_locP3.x()*relativeTime;
            Real pyP = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (pxP *pxP) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * pxP;
            relativeTime = (t+dt - m_tBegin.getValue() ) / (m_tEnd.getValue() - m_tBegin.getValue());
            Real pxN = m_locP3.x()*relativeTime;
            Real pyN = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (pxN *pxN) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * pxN;

            Vec3R locVel = Vec3R( (pxN-pxP)/dt, (pyN-pyP)/dt, 0.0);

            Vec3R worldVel = m_projection.rotate(locVel);

            dx[*it] = worldVel;
        }
    }
}

template <class DataTypes>
void ParabolicConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    Real t = (Real) this->getContext()->getTime();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real relativeTime = (t - m_tBegin.getValue() ) / (m_tEnd.getValue() - m_tBegin.getValue());
        const SetIndexArray & indices = m_indices.getValue();

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            //compute position from the equation of the parabol : Y = -y2/(x3*x2-x2�) * X� + (x3*y2)/(x3*x2-x2�) * X
            //with P1:(0,0,0), P2:(x2,y2,z2), P3:(x3,y3,z3) , projected in parabol plan
            Real px = m_locP3.x()*relativeTime;
            Real py = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (px *px) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * px;
            Vec3R locPos( px , py, 0.0);

            //projection to world coordinates
            Vec3R worldPos = m_P1.getValue() + m_projection.rotate(locPos);

            x[*it] = worldPos;
        }
    }
}

template <class DataTypes>
void ParabolicConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
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
void ParabolicConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    Real dt = (Real) this->getContext()->getDt();
    Real t = m_tEnd.getValue() - m_tBegin.getValue();
    Real nbStep = t/dt;

    glDisable (GL_LIGHTING);
    glPointSize(5);
    glColor4f (1,0.5,0.5,1);

    glBegin (GL_LINES);
    for (unsigned int i=0 ; i< nbStep ; i++)
    {
        //draw lines between each step of the parabolic trajectory
        //so, the smaller is dt, the finer is the parabol
        Real relativeTime = i/nbStep;
        Real px = m_locP3.x()*relativeTime;
        Real py = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (px *px) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * px;
        Vec3R locPos( px , py, 0.0);
        Vec3R worldPos = m_P1.getValue() + m_projection.rotate(locPos);

        helper::gl::glVertexT(worldPos);

        relativeTime = (i+1)/nbStep;
        px = m_locP3.x()*relativeTime;
        py = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (px *px) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * px;
        locPos = Vec3R( px , py, 0.0);
        worldPos = m_P1.getValue() + m_projection.rotate(locPos);
        helper::gl::glVertexT(worldPos);
    }
    glEnd();

    //draw points for the 3 control points
    glBegin(GL_POINTS);
    helper::gl::glVertexT(m_P1.getValue());
    helper::gl::glVertexT(m_P2.getValue());
    helper::gl::glVertexT(m_P3.getValue());
    glEnd();
#endif /* SOFA_NO_OPENGL */
}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif
