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


#include <sofa/component/mechanicalload/SurfacePressureForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/RGBAColor.h>
#include <vector>
#include <set>
#include <iostream>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>


namespace sofa::component::mechanicalload
{

template <class DataTypes>
SurfacePressureForceField<DataTypes>::SurfacePressureForceField()
    : m_pressure(initData(&m_pressure, (Real)0.0, "pressure", "Pressure force per unit area"))
    , m_min(initData(&m_min, Coord(), "min", "Lower bond of the selection box"))
    , m_max(initData(&m_max, Coord(), "max", "Upper bond of the selection box"))
    , m_triangleIndices(initData(&m_triangleIndices, "triangleIndices", "Indices of affected triangles"))
    , m_quadIndices(initData(&m_quadIndices, "quadIndices", "Indices of affected quads"))
    , m_pulseMode(initData(&m_pulseMode, false, "pulseMode", "Cyclic pressure application"))
    , m_pressureLowerBound(initData(&m_pressureLowerBound, (Real)0.0, "pressureLowerBound", "Pressure lower bound force per unit area (active in pulse mode)"))
    , m_pressureSpeed(initData(&m_pressureSpeed, (Real)0.0, "pressureSpeed", "Continuous pressure application in Pascal per second. Only active in pulse mode"))
    , m_volumeConservationMode(initData(&m_volumeConservationMode, false, "volumeConservationMode", "Pressure variation follow the inverse of the volume variation"))
    , m_useTangentStiffness(initData(&m_useTangentStiffness, true, "useTangentStiffness", "Whether (non-symmetric) stiffness matrix should be used"))
    , m_defaultVolume(initData(&m_defaultVolume, (Real)-1.0, "defaultVolume", "Default Volume"))
    , m_mainDirection(initData(&m_mainDirection, Deriv(), "mainDirection", "Main direction for pressure application"))
    , m_drawForceScale(initData(&m_drawForceScale, (Real)0, "drawForceScale", "DEBUG: scale used to render force vectors"))
    , l_topology(initLink("topology", "link to the topology container"))
    , state(INCREASE)
    , m_topology(nullptr)
{}


template <class DataTypes>
SurfacePressureForceField<DataTypes>::~SurfacePressureForceField()
{}


template <class DataTypes>
void SurfacePressureForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    state = m_pressure.getValue() > 0 ? INCREASE : DECREASE;

    if (m_pulseMode.getValue() && (m_pressureSpeed.getValue() == 0.0))
    {
        msg_warning() << "Default pressure speed value has been set in SurfacePressureForceField";
        m_pressureSpeed.setValue((Real)fabs(m_pressure.getValue()));
    }

    m_pulseModePressure = 0.0;
}


template <class DataTypes>
void SurfacePressureForceField<DataTypes>::verifyDerivative(VecDeriv& v_plus, VecDeriv& v, VecVec3DerivValues& DVval, VecVec3DerivIndices& DVind,
                                                            const VecDeriv& Din)
{

    msg_info() <<" verifyDerivative : vplus.size()="<<v_plus.size()<<"  v.size()="<<v.size()
               <<"  DVval.size()="<<DVval.size()<<" DVind.size()="<<DVind.size()<<"  Din.size()="<<Din.size();

    std::stringstream s;
    for (unsigned int i = 0; i < v.size(); i++)
    {
        Deriv DV;
        DV.clear();
        s << " DVnum[" << i << "] =" << v_plus[i] - v[i];

        for (unsigned int j = 0; j < DVval[i].size(); j++)
        {
            DV += DVval[i][j] * Din[(DVind[i][j])];
        }
        s << " DVana[" << i << "] = " << DV << " DVval[i].size() = " << DVval[i].size() << msgendl;
    }

    msg_info() << s.str();
}


template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    m_f.resize(f.size());
    for (unsigned int i = 0; i < m_f.size(); i++)
    {
        m_f[i].clear(); // store forces for visualization
    }

    Real p = m_pulseMode.getValue() ? computePulseModePressure() : m_pressure.getValue();

    if (m_topology)
    {
        if (m_volumeConservationMode.getValue())
        {
            if (m_defaultVolume.getValue() == -1)
            {
                m_defaultVolume.setValue(computeMeshVolume(f, x));
            }
            else if (m_defaultVolume.getValue() != 0)
            {
                p *= m_defaultVolume.getValue() / computeMeshVolume(f, x);
            }
        }
        bool useStiffness = m_useTangentStiffness.getValue();
        // Triangles

        derivTriNormalValues.clear();
        derivTriNormalValues.resize(x.size());
        derivTriNormalIndices.clear();
        derivTriNormalIndices.resize(x.size());

        for (unsigned int i = 0; i < x.size(); i++)
        {
            derivTriNormalValues[i].clear();
            derivTriNormalIndices[i].clear();
        }

        if (m_triangleIndices.getValue().size() > 0)
        {
            for (unsigned int i = 0; i < m_triangleIndices.getValue().size(); i++)
            {
                addTriangleSurfacePressure(m_triangleIndices.getValue()[i], m_f, x, v, p, useStiffness);
            }
        }
        else if (m_topology->getNbTriangles() > 0)
        {
            for (unsigned int i = 0; i < (unsigned int)m_topology->getNbTriangles(); i++)
            {
                Triangle t = m_topology->getTriangle(i);
                if (isInPressuredBox(x[t[0]]) && isInPressuredBox(x[t[1]]) && isInPressuredBox(x[t[2]]))
                {
                    addTriangleSurfacePressure(i, m_f, x, v, p, useStiffness);
                }
            }
        }

        // Quads

        if (m_quadIndices.getValue().size() > 0)
        {
            for (unsigned int i = 0; i < m_quadIndices.getValue().size(); i++)
            {
                addQuadSurfacePressure(m_quadIndices.getValue()[i], m_f, x, v, p);
            }
        }
        else if (m_topology->getNbQuads() > 0)
        {
            for (unsigned int i = 0; i < (unsigned int)m_topology->getNbQuads(); i++)
            {
                Quad q = m_topology->getQuad(i);

                if (isInPressuredBox(x[q[0]]) && isInPressuredBox(x[q[1]]) && isInPressuredBox(x[q[2]]) && isInPressuredBox(x[q[3]]))
                {
                    addQuadSurfacePressure(i, m_f, x, v, p);
                }
            }
        }
    }

    for (unsigned int i = 0; i < m_f.size(); i++)
    {
        f[i] += m_f[i];
    }
    d_f.endEdit();
}

template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    Real kFactor = (Real)mparams->kFactor();
    if (m_useTangentStiffness.getValue())
    {
        VecDeriv& df = *(d_df.beginEdit());
        const VecDeriv& dx = d_dx.getValue();

        for (unsigned int i = 0; i < derivTriNormalIndices.size(); i++)
        {
            for (unsigned int j = 0; j < derivTriNormalIndices[i].size(); j++)
            {
                unsigned int v = derivTriNormalIndices[i][j];
                df[i] += (derivTriNormalValues[i][j] * dx[v]) * kFactor;
            }
        }
        d_df.endEdit();
    }
}


template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    const sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::linearalgebra::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const int N = Coord::total_size;
    if (m_useTangentStiffness.getValue())
    {
        for (unsigned int i = 0; i < derivTriNormalIndices.size(); i++)
        {
            for (unsigned int j = 0; j < derivTriNormalIndices[i].size(); j++)
            {
                unsigned int v = derivTriNormalIndices[i][j];
                Mat33 Kiv = derivTriNormalValues[i][j];

                for (unsigned int l = 0; l < 3; l++)
                {
                    for (unsigned int c = 0; c < 3; c++)
                    {
                        mat->add(offset + N * i + l, offset + N * v + c, kFact * Kiv[l][c]);
                    }
                }
            }
        }
    }
}

template <class DataTypes>
void SurfacePressureForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    if (m_useTangentStiffness.getValue())
    {
        static constexpr auto N = Deriv::total_size;

        auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                   .withRespectToPositionsIn(this->mstate);

        for (std::size_t i = 0; i < derivTriNormalIndices.size(); i++)
        {
            for (std::size_t j = 0; j < derivTriNormalIndices[i].size(); j++)
            {
                const unsigned int v = derivTriNormalIndices[i][j];
                const Mat33& Kiv = derivTriNormalValues[i][j];

                dfdx(N * i, N * v) += Kiv;
            }
        }
    }
}

template <class DataTypes>
void SurfacePressureForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template <class DataTypes>
SReal SurfacePressureForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& /* x */) const
{
    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}

template <class DataTypes>
void SurfacePressureForceField<DataTypes>::setPressure(const Real _pressure)
{
    this->m_pressure.setValue(_pressure);
}

template <class DataTypes>
typename SurfacePressureForceField<DataTypes>::Real SurfacePressureForceField<DataTypes>::computeMeshVolume(const VecDeriv& /*f*/, const VecCoord& x)
{
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Quad Quad;

    Real volume = 0;
    unsigned int nTriangles = 0;
    const VecIndex& triangleIndices = m_triangleIndices.getValue();
    if (!triangleIndices.empty())
    {
        nTriangles = triangleIndices.size();
    }
    else
    {
        nTriangles = m_topology->getNbTriangles();
    }

    unsigned int triangleIdx = 0;
    for (unsigned int i = 0; i < nTriangles; i++)
    {
        if (!triangleIndices.empty())
        {
            triangleIdx = triangleIndices[i];
        }
        else
        {
            triangleIdx = i;
        }
        Triangle t = m_topology->getTriangle(triangleIdx);
        const Coord a = x[t[0]];
        const Coord b = x[t[1]];
        const Coord c = x[t[2]];
        volume += dot(cross(a, b), c);
    }

    unsigned int nQuads = 0;
    const VecIndex& quadIndices = m_quadIndices.getValue();
    if (!quadIndices.empty())
    {
        nQuads = quadIndices.size();
    }
    else
    {
        nQuads = m_topology->getNbQuads();
    }

    unsigned int quadIdx = 0;
    for (unsigned int i = 0; i < nQuads; i++)
    {
        if (!quadIndices.empty())
        {
            quadIdx = quadIndices[i];
        }
        else
        {
            quadIdx = i;
        }
        Quad q = m_topology->getQuad(quadIdx);
        const Coord a = x[q[0]];
        const Coord b = x[q[1]];
        const Coord c = x[q[2]];
        const Coord d = x[q[3]];
        volume += dot(cross(a, b), c);
        volume += dot(cross(a, c), d);
    }

    // Divide by 6 when computing tetrahedron volume
    return volume / 6.0;
}


template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addTriangleSurfacePressure(unsigned int triId, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure, bool computeDerivatives)
{
    Triangle t = m_topology->getTriangle(triId);

    Deriv ab = x[t[1]] - x[t[0]];
    Deriv ac = x[t[2]] - x[t[0]];
    Deriv bc = x[t[2]] - x[t[1]];

    Deriv p = (ab.cross(ac)) * (pressure / static_cast<Real>(6.0));


    if (computeDerivatives)
    {
        Mat33 DcrossDA;
        DcrossDA[0][0]=0;       DcrossDA[0][1]=-bc[2];  DcrossDA[0][2]=bc[1];
        DcrossDA[1][0]=bc[2];   DcrossDA[1][1]=0;       DcrossDA[1][2]=-bc[0];
        DcrossDA[2][0]=-bc[1];  DcrossDA[2][1]=bc[0];   DcrossDA[2][2]=0;

        Mat33 DcrossDB;
        DcrossDB[0][0]=0;       DcrossDB[0][1]=ac[2];   DcrossDB[0][2]=-ac[1];
        DcrossDB[1][0]=-ac[2];  DcrossDB[1][1]=0;       DcrossDB[1][2]=ac[0];
        DcrossDB[2][0]=ac[1];  DcrossDB[2][1]=-ac[0];   DcrossDB[2][2]=0;


        Mat33 DcrossDC;
        DcrossDC[0][0]=0;       DcrossDC[0][1]=-ab[2];  DcrossDC[0][2]=ab[1];
        DcrossDC[1][0]=ab[2];   DcrossDC[1][1]=0;       DcrossDC[1][2]=-ab[0];
        DcrossDC[2][0]=-ab[1];  DcrossDC[2][1]=ab[0];   DcrossDC[2][2]=0;

        for (unsigned int j = 0; j < 3; j++)
        {
            derivTriNormalValues[t[j]].push_back(DcrossDA * (pressure / static_cast<Real>(6.0)));
            derivTriNormalValues[t[j]].push_back(DcrossDB * (pressure / static_cast<Real>(6.0)));
            derivTriNormalValues[t[j]].push_back(DcrossDC * (pressure / static_cast<Real>(6.0)));

            derivTriNormalIndices[t[j]].push_back(t[0]);
            derivTriNormalIndices[t[j]].push_back(t[1]);
            derivTriNormalIndices[t[j]].push_back(t[2]);
        }
    }


    if (m_mainDirection.getValue() != Deriv())
    {
        Deriv n = ab.cross(ac);
        n.normalize();
        Real scal = n * m_mainDirection.getValue();
        p *= fabs(scal);
    }

    f[t[0]] += p;
    f[t[1]] += p;
    f[t[2]] += p;
}


template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addQuadSurfacePressure(unsigned int quadId, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure)
{
    Quad q = m_topology->getQuad(quadId);

    Deriv ab = x[q[1]] - x[q[0]];
    Deriv ac = x[q[2]] - x[q[0]];
    Deriv ad = x[q[3]] - x[q[0]];

    Deriv p1 = (ab.cross(ac)) * (pressure / static_cast<Real>(6.0));
    Deriv p2 = (ac.cross(ad)) * (pressure / static_cast<Real>(6.0));

    Deriv p = p1 + p2;

    f[q[0]] += p;
    f[q[1]] += p1;
    f[q[2]] += p;
    f[q[3]] += p2;
}


template <class DataTypes>
bool SurfacePressureForceField<DataTypes>::isInPressuredBox(const Coord& x) const
{
    if ((m_max.getValue() == Coord()) && (m_min.getValue() == Coord()))
    {
        return true;
    }

    return ((x[0] >= m_min.getValue()[0])
        && (x[0] <= m_max.getValue()[0])
        && (x[1] >= m_min.getValue()[1])
        && (x[1] <= m_max.getValue()[1])
        && (x[2] >= m_min.getValue()[2])
        && (x[2] <= m_max.getValue()[2]));
}

template <class DataTypes>
typename SurfacePressureForceField<DataTypes>::Real SurfacePressureForceField<DataTypes>::computePulseModePressure()
{
    SReal dt = this->getContext()->getDt();

    if (state == INCREASE)
    {
        Real pUpperBound = (m_pressure.getValue() > 0) ? m_pressure.getValue() : m_pressureLowerBound.getValue();

        m_pulseModePressure += (Real)(m_pressureSpeed.getValue() * dt);

        if (m_pulseModePressure >= pUpperBound)
        {
            m_pulseModePressure = pUpperBound;
            state = DECREASE;
        }

        return m_pulseModePressure;
    }

    if (state == DECREASE)
    {
        Real pLowerBound = (m_pressure.getValue() > 0) ? m_pressureLowerBound.getValue() : m_pressure.getValue();

        m_pulseModePressure -= (Real)(m_pressureSpeed.getValue() * dt);

        if (m_pulseModePressure <= pLowerBound)
        {
            m_pulseModePressure = pLowerBound;
            state = INCREASE;
        }

        return m_pulseModePressure;
    }

    return 0.0;
}

template <class DataTypes>
void SurfacePressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields())
        return;
    if (!this->mstate)
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (vparams->displayFlags().getShowWireFrame())
    {
        vparams->drawTool()->setPolygonMode(0, true);
    }

    vparams->drawTool()->disableLighting();

    constexpr sofa::type::RGBAColor boxcolor(0.0f, 0.8f, 0.3f, 1.0f);

    vparams->drawTool()->setMaterial(boxcolor);
    vparams->drawTool()->drawBoundingBox(DataTypes::getCPos(m_min.getValue()), DataTypes::getCPos(m_max.getValue()));

    if (vparams->displayFlags().getShowWireFrame())
    {
        vparams->drawTool()->setPolygonMode(0, false);
    }


    helper::ReadAccessor<DataVecCoord> x = this->mstate->read(core::ConstVecCoordId::position());
    if (m_drawForceScale.getValue() && m_f.size() == x.size())
    {
        std::vector<type::Vec3> points;
        constexpr sofa::type::RGBAColor color(0, 1, 0.5, 1);

        for (unsigned int i = 0; i < x.size(); i++)
        {
            points.push_back(DataTypes::getCPos(x[i]));
            points.push_back(DataTypes::getCPos(x[i]) + DataTypes::getDPos(m_f[i]) * m_drawForceScale.getValue());
        }
        vparams->drawTool()->drawLines(points, 1, color);
    }
}

} // namespace sofa::component::mechanicalload
