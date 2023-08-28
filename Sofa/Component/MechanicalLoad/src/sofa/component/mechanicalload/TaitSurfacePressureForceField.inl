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

#include <sofa/component/mechanicalload/TaitSurfacePressureForceField.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/behavior/BlocMatrixWriter.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <vector>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>


namespace sofa::component::mechanicalload
{

template <class DataTypes>
TaitSurfacePressureForceField<DataTypes>::TaitSurfacePressureForceField():

    m_p0(initData(&m_p0, (Real)0.0, "p0", "IN: Rest pressure when V = V0")),
    m_B(initData(&m_B, (Real)0.0, "B", "IN: Bulk modulus (resistance to uniform compression)")),
    m_gamma(initData(&m_gamma, (Real)0.0, "gamma", "IN: Bulk modulus (resistance to uniform compression)")),
    m_injectedVolume(initData(&m_injectedVolume, (Real)0.0, "injectedVolume", "IN: Injected (or extracted) volume since the start of the simulation")),
    m_maxInjectionRate(initData(&m_maxInjectionRate, (Real)1000.0, "maxInjectionRate", "IN: Maximum injection rate (volume per second)")),

    m_initialVolume(initData(&m_initialVolume, (Real)0.0, "initialVolume", "OUT: Initial volume, as computed from the surface rest position")),
    m_currentInjectedVolume(initData(&m_currentInjectedVolume, (Real)0.0, "currentInjectedVolume", "OUT: Current injected (or extracted) volume (taking into account maxInjectionRate)")),
    m_v0(initData(&m_v0, (Real)0.0, "v0", "OUT: Rest volume (as computed from initialVolume + injectedVolume)")),
    m_currentVolume(initData(&m_currentVolume, (Real)0.0, "currentVolume", "OUT: Current volume, as computed from the last surface position")),
    m_currentPressure(initData(&m_currentPressure, (Real)0.0, "currentPressure", "OUT: Current pressure, as computed from the last surface position")),
    m_currentStiffness(initData(&m_currentStiffness, (Real)0.0, "currentStiffness", "OUT: dP/dV at current volume and pressure")),
    m_pressureTriangles(initData(&m_pressureTriangles, "pressureTriangles", "OUT: list of triangles where a pressure is applied (mesh triangles + tesselated quads)")),
    m_initialSurfaceArea(initData(&m_initialSurfaceArea, (Real)0.0, "initialSurfaceArea", "OUT: Initial surface area, as computed from the surface rest position")),
    m_currentSurfaceArea(initData(&m_currentSurfaceArea, (Real)0.0, "currentSurfaceArea", "OUT: Current surface area, as computed from the last surface position")),
    m_drawForceScale(initData(&m_drawForceScale, (Real)0.001, "drawForceScale", "DEBUG: scale used to render force vectors")),
    m_drawForceColor(initData(&m_drawForceColor, sofa::type::RGBAColor(0,1,1,1), "drawForceColor", "DEBUG: color used to render force vectors")),
    m_volumeAfterTC(initData(&m_volumeAfterTC, "volumeAfterTC", "OUT: Volume after a topology change")),
    m_surfaceAreaAfterTC(initData(&m_surfaceAreaAfterTC, (Real)0.0, "surfaceAreaAfterTC", "OUT: Surface area after a topology change")),
    l_topology(initLink("topology", "link to the topology container")),
    m_topology(nullptr),
    lastTopologyRevision(-1)
{
    m_p0.setGroup("Controls");
    m_B.setGroup("Controls");
    m_gamma.setGroup("Controls");
    m_injectedVolume.setGroup("Controls");
    m_maxInjectionRate.setGroup("Controls");
    m_initialVolume.setGroup("Results");
    m_initialVolume.setReadOnly(true);
    m_currentInjectedVolume.setGroup("Results");
    //m_currentInjectedVolume.setReadOnly(true);
    m_v0.setGroup("Results");
    m_v0.setReadOnly(true);
    m_currentVolume.setGroup("Results");
    m_currentVolume.setReadOnly(true);
    m_currentPressure.setGroup("Results");
    m_currentPressure.setReadOnly(true);
    m_currentStiffness.setGroup("Results");
    m_currentStiffness.setReadOnly(true);
    m_pressureTriangles.setDisplayed(false);
    m_pressureTriangles.setPersistent(false);
    m_pressureTriangles.setReadOnly(true);
    m_initialSurfaceArea.setGroup("Stats");
    m_initialSurfaceArea.setReadOnly(true);
    m_currentSurfaceArea.setGroup("Stats");
    m_currentSurfaceArea.setReadOnly(true);
    m_volumeAfterTC.setGroup("Results");
    m_volumeAfterTC.setReadOnly(true);
    m_surfaceAreaAfterTC.setGroup("Results");
    m_surfaceAreaAfterTC.setReadOnly(true);
    this->f_listening.setValue(true);
}

template <class DataTypes>
TaitSurfacePressureForceField<DataTypes>::~TaitSurfacePressureForceField()
{

}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::init()
{
    Inherit1::init();

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

    updateFromTopology();
    computeMeshVolumeAndArea(*m_currentVolume.beginEdit(), *m_currentSurfaceArea.beginEdit(), this->mstate->read(sofa::core::VecCoordId::position()));
    m_currentVolume.endEdit();
    m_currentSurfaceArea.endEdit();
    Real currentStiffness = 0;
    Real currentPressure = 0;
    computePressureAndStiffness(currentPressure, currentStiffness, m_currentVolume.getValue(), m_v0.getValue());
    m_currentPressure.setValue(currentPressure);
    m_currentStiffness.setValue(currentStiffness);
    computeStatistics(this->mstate->read(sofa::core::VecCoordId::position()));
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::storeResetState()
{
    Inherit1::storeResetState();
    reset_injectedVolume = m_injectedVolume.getValue();
    reset_currentInjectedVolume = m_currentInjectedVolume.getValue();
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::reset()
{
    Inherit1::reset();
    m_injectedVolume.setValue(reset_injectedVolume);
    m_currentInjectedVolume.setValue(reset_currentInjectedVolume);
    updateFromTopology();
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::handleEvent(core::objectmodel::Event *event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        SReal dt = (static_cast<sofa::simulation::AnimateBeginEvent *> (event))->getDt();
        Real inj = m_injectedVolume.getValue();
        Real curInj = m_currentInjectedVolume.getValue();
        if (inj != curInj)
        {
            Real maxInj = (Real)(m_maxInjectionRate.getValue()*dt);
            if (fabs(inj-curInj) <= maxInj)
                curInj = inj;
            else if (inj < curInj)
                curInj -= maxInj;
            else if (inj > curInj)
                curInj += maxInj;
            msg_info() << "Current Injected Volume = " << curInj;
            m_currentInjectedVolume.setValue(curInj);
            updateFromTopology();
        }
    }
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::updateFromTopology()
{
    if (m_topology && lastTopologyRevision != m_topology->getRevision())
    {
        if (lastTopologyRevision >= 0)
            msg_error() << "NEW TOPOLOGY v" << m_topology->getRevision();

        lastTopologyRevision = m_topology->getRevision();
        computePressureTriangles();

        computeMeshVolumeAndArea(*m_volumeAfterTC.beginEdit(), *m_surfaceAreaAfterTC.beginEdit(), this->mstate->read(core::ConstVecCoordId::restPosition()));
        m_volumeAfterTC.endEdit();
        m_surfaceAreaAfterTC.endEdit();
		if (lastTopologyRevision == 0)
		{
			m_initialVolume.setValue(m_volumeAfterTC.getValue());
			m_initialSurfaceArea.setValue(m_surfaceAreaAfterTC.getValue());
		}
    }

    m_v0.setValue(m_initialVolume.getValue() + m_currentInjectedVolume.getValue());
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::computePressureTriangles()
{
    const SeqTriangles& triangles = m_topology->getTriangles();
    const SeqQuads& quads = m_topology->getQuads();
    helper::WriteAccessor< Data< SeqTriangles > > pressureTriangles = m_pressureTriangles;
    pressureTriangles.resize(triangles.size()+2*quads.size());
    unsigned int index = 0;
    for (unsigned int i=0; i<triangles.size(); i++)
    {
        pressureTriangles[index++] = triangles[i];
    }
    for (unsigned int i=0; i<quads.size(); i++)
    {
        pressureTriangles[index++] = Triangle(quads[i][0],quads[i][1],quads[i][2]);
        pressureTriangles[index++] = Triangle(quads[i][2],quads[i][3],quads[i][0]);
    }
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /*d_v*/)
{
    updateFromTopology();

    helper::WriteAccessor<DataVecDeriv> f = d_f;
    helper::ReadAccessor<DataVecCoord> x = d_x;
    //helper::ReadAccessor<DataVecDeriv> v = d_v;
    //helper::ReadAccessor<DataVecCoord> x0 = this->mstate->read(core::ConstVecCoordId::restPosition());
    const helper::ReadAccessor< Data< SeqTriangles > > pressureTriangles = m_pressureTriangles;

    computeMeshVolumeAndArea(*m_currentVolume.beginEdit(), *m_currentSurfaceArea.beginEdit(), x);
    m_currentVolume.endEdit();
    m_currentSurfaceArea.endEdit();
    Real currentVolume = m_currentVolume.getValue();
    computeStatistics(x);
    Real currentStiffness = 0;
    Real currentPressure = 0;
	// apply volume correction after a topological change

    if (m_volumeAfterTC.isSet())
    {
	    currentVolume -= (m_volumeAfterTC.getValue() - m_initialVolume.getValue());
    }

    computePressureAndStiffness(currentPressure, currentStiffness, currentVolume, m_v0.getValue());
    m_currentPressure.setValue(currentPressure);
    m_currentStiffness.setValue(currentStiffness);

    // first compute gradV
    helper::WriteAccessor<VecDeriv> gradV = this->gradV;
    gradV.resize(x.size());
    for (unsigned int i=0; i<x.size(); ++i)
        gradV[i].clear();
    for (unsigned int i = 0; i < pressureTriangles.size(); i++)
    {
        Triangle t = pressureTriangles[i];
        Deriv n = cross(x[t[1]] - x[t[0]],x[t[2]] - x[t[0]]);
        gradV[t[0]] += n;
        gradV[t[1]] += n;
        gradV[t[2]] += n;
    }
    for (unsigned int i=0; i<x.size(); ++i)
        gradV[i] /= (Real)6;

    // Then F = P * gradV

    for (unsigned int i=0; i<x.size(); ++i)
        f[i] += gradV[i] * currentPressure;
    /*
        const Real fscale = currentPressure/(Real)6;
        for (unsigned int i = 0; i < pressureTriangles.size(); i++)
        {
            Triangle t = pressureTriangles[i];
            Deriv force = cross(x[t[1]] - x[t[0]],x[t[2]] - x[t[0]]) * fscale;
            f[t[0]] += force;
            f[t[1]] += force;
            f[t[2]] += force;
        }
    */
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv&  d_df , const DataVecDeriv&  d_dx )
{
    helper::WriteAccessor<DataVecDeriv> df = d_df;
    helper::ReadAccessor<DataVecDeriv> dx = d_dx;
    helper::ReadAccessor<DataVecCoord> x = mparams->readX(this->mstate.get());
    //helper::ReadAccessor<DataVecCoord> x0 = this->mstate->read(core::ConstVecCoordId::restPosition());
    const helper::ReadAccessor< Data< SeqTriangles > > pressureTriangles = m_pressureTriangles;
    helper::ReadAccessor<VecDeriv> gradV = this->gradV;

    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    //const Real currentVolume = m_currentVolume.getValue();
    const Real currentPressure = m_currentPressure.getValue();
    const Real currentStiffness = m_currentStiffness.getValue();

    // First compute dV
    Real dV = 0;
    for (unsigned int i = 0; i < gradV.size(); i++)
        dV += dot(gradV[i],dx[i]);

    // Then add first component of df : K * dV * gradV

    const Real dfscale1 = kFactor * currentStiffness * dV;
    for (unsigned int i=0; i<x.size(); ++i)
        df[i] += gradV[i] * dfscale1;

    // Then add second component of df = P * dgradV

    const Real dfscale2 = kFactor * currentPressure/(Real)6;

    for (unsigned int i = 0; i < pressureTriangles.size(); i++)
    {
        Triangle t = pressureTriangles[i];
        Deriv df_dn = ( cross(dx[t[1]] - dx[t[0]],x[t[2]] - x[t[0]]) +
                cross(x[t[1]] - x[t[0]],dx[t[2]] - dx[t[0]]) ) * dfscale2;
        df[t[0]] += df_dn;
        df[t[1]] += df_dn;
        df[t[2]] += df_dn;
    }
}

template<class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    core::behavior::BlocMatrixWriter<MatBloc> writer;
    writer.addKToMatrix(this, mparams, matrix->getMatrix(this->mstate));
}

template<class DataTypes>
SReal TaitSurfacePressureForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
{
    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}

/// Convert a vector cross-product to a to matrix multiplication, i.e. cross(a,b) = matCross(a)*b
template <typename T>
inline sofa::type::Mat<3,3,T> matCross( const sofa::type::Vec<3,T>& u )
{
    sofa::type::Mat<3,3,T> res(sofa::type::NOINIT);
    res[0][0] =  0   ; res[0][1] = -u[2]; res[0][2] =  u[1];
    res[1][0] =  u[2]; res[1][1] =  0   ; res[1][2] = -u[0];
    res[2][0] = -u[1]; res[2][1] =  u[0]; res[2][2] =  0   ;
    return res;
}

template<class DataTypes>
template<class MatrixWriter>
void TaitSurfacePressureForceField<DataTypes>::addKToMatrixT(const core::MechanicalParams* mparams, MatrixWriter mwriter)
{
    helper::ReadAccessor<DataVecCoord> x = mparams->readX(this->mstate.get());
    const helper::ReadAccessor< Data< SeqTriangles > > pressureTriangles = m_pressureTriangles;

    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    const Real currentPressure = m_currentPressure.getValue();
    const Real currentStiffness = m_currentStiffness.getValue();

    // First compute df = dP*N
    if (currentStiffness != 0)
    {
        const Real dfscale1 = kFactor * currentStiffness;

        for (unsigned int i = 0; i < x.size(); i++)
        {
            Deriv di = gradV[i] * dfscale1;
            for (unsigned int j = 0; j < x.size(); j++)
            {
                Deriv dj = gradV[j];
                MatBloc m = type::dyad(di,dj);
                mwriter.add(i,j,m);
            }
        }
    }


    // Then compute df = P*dN
    if (currentPressure != 0)
    {
        const Real dfscale2 = kFactor * currentPressure / (Real)6;
        for (unsigned int i = 0; i < pressureTriangles.size(); i++)
        {
            Triangle t = pressureTriangles[i];
            MatBloc mbc,mca,mab;
            mbc = matCross((x[t[2]]-x[t[1]])*dfscale2);
            mca = matCross((x[t[0]]-x[t[2]])*dfscale2);
            mab = matCross((x[t[1]]-x[t[0]])*dfscale2);

            // Full derivative matrix of triangle (ABC):
            // K(A,A) = mbc   K(A,B) = mca   K(A,C) = mab
            // K(B,A) = mbc   K(B,B) = mca   K(B,C) = mab
            // K(C,A) = mbc   K(C,B) = mca   K(C,C) = mab

            // -> the diagonal contributions become zero for closed meshes

            /*mwriter.add(t[0], t[0], mbc);*/ mwriter.add(t[0], t[1], mca); mwriter.add(t[0], t[2], mab);
            mwriter.add(t[1], t[0], mbc); /*mwriter.add(t[1], t[1], mca);*/ mwriter.add(t[1], t[2], mab);
            mwriter.add(t[2], t[0], mbc); mwriter.add(t[2], t[1], mca); /*mwriter.add(t[2],t[2], mab); */
        }
    }
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
{
    const auto mstateSize = this->mstate->getSize();
    const helper::ReadAccessor< Data< SeqTriangles > > pressureTriangles = m_pressureTriangles;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate.get())
                       .withRespectToPositionsIn(this->mstate.get());

    const Real currentPressure = m_currentPressure.getValue();
    const Real currentStiffness = m_currentStiffness.getValue();

    // First compute df = dP*N
    if (currentStiffness != 0)
    {
        for (unsigned int i = 0; i < mstateSize; i++)
        {
            Deriv di = gradV[i] * currentStiffness;
            for (unsigned int j = 0; j < mstateSize; j++)
            {
                Deriv dj = gradV[j];
                MatBloc m = type::dyad(di,dj);
                dfdx(i * Deriv::total_size, j * Deriv::total_size) += m;
            }
        }
    }

    helper::ReadAccessor<Data<VecCoord> > x = this->mstate->readPositions();

    // Then compute df = P*dN
    if (currentPressure != 0)
    {
        const Real dfscale2 = currentPressure / 6;
        for (unsigned int i = 0; i < pressureTriangles.size(); i++)
        {
            Triangle t = pressureTriangles[i];
            MatBloc mbc, mca, mab;
            mbc = matCross((x[t[2]] - x[t[1]]) * dfscale2);
            mca = matCross((x[t[0]] - x[t[2]]) * dfscale2);
            mab = matCross((x[t[1]] - x[t[0]]) * dfscale2);

            // Full derivative matrix of triangle (ABC):
            // K(A,A) = mbc   K(A,B) = mca   K(A,C) = mab
            // K(B,A) = mbc   K(B,B) = mca   K(B,C) = mab
            // K(C,A) = mbc   K(C,B) = mca   K(C,C) = mab

            // -> the diagonal contributions become zero for closed meshes

            dfdx(t[0] * Deriv::total_size, t[1] * Deriv::total_size) += mca;
            dfdx(t[0] * Deriv::total_size, t[2] * Deriv::total_size) += mab;
            dfdx(t[1] * Deriv::total_size, t[0] * Deriv::total_size) += mbc;
            dfdx(t[1] * Deriv::total_size, t[2] * Deriv::total_size) += mab;
            dfdx(t[2] * Deriv::total_size, t[0] * Deriv::total_size) += mbc;
            dfdx(t[2] * Deriv::total_size, t[1] * Deriv::total_size) += mca;
        }
    }
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::computeMeshVolumeAndArea(Real& volume, Real& area, const helper::ReadAccessor<DataVecCoord>& x)
{
    volume = 0;
    area = 0;
    if (!m_topology || x.empty()) return;

    Coord center;
    for (unsigned int i = 0; i < x.size(); i++)
    {
        center += x[i];
    }
    center /= x.size();

    Real volume6 = 0;
    Real area2 = 0;

    const helper::ReadAccessor< Data< SeqTriangles > > pressureTriangles = m_pressureTriangles;
    for (unsigned int i = 0; i < pressureTriangles.size(); i++)
    {
        Triangle t = pressureTriangles[i];
        Coord a = x[t[0]] - center;
        Coord b = x[t[1]] - center;
        Coord c = x[t[2]] - center;
        volume6 += dot(cross(a,b),c);
        area2 += cross(b-a,c-a).norm();
    }

    volume = volume6 / (Real)6.0;
    area = area2 / (Real)2.0;
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::computeStatistics(const helper::ReadAccessor<DataVecCoord>& /*x*/)
{
}

template<class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowInteractionForceFields()) return;
    if (!this->mstate) return;
    if (!this->m_topology) return;

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());

    helper::ReadAccessor<DataVecCoord> x = this->mstate->read(core::ConstVecCoordId::position());

    const helper::ReadAccessor< Data< SeqTriangles > > pressureTriangles = m_pressureTriangles;

    std::vector< sofa::type::Vec3i > indices;
    std::vector< type::Vec3 > normals;
    if (m_drawForceScale.getValue() != (Real)0.0)
    {
        std::vector< sofa::type::Vec3 > points;
        points.clear();
        const Real fscale = m_currentPressure.getValue()*m_drawForceScale.getValue();
        for (unsigned int i=0; i<pressureTriangles.size(); i++)
        {
            Triangle t = pressureTriangles[i];
            sofa::type::Vec3 a = x[t[0]];
            sofa::type::Vec3 b = x[t[1]];
            sofa::type::Vec3 c = x[t[2]];
            sofa::type::Vec3 n = cross(b-a,c-a) * fscale;
            sofa::type::Vec3 center = (a+b+c)/(Real)3;
            points.push_back(center);
            points.push_back(center+n);
        }
        vparams->drawTool()->drawLines(points, 1, m_drawForceColor.getValue());
    }

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,true);
}

template <class DataTypes>
void TaitSurfacePressureForceField<DataTypes>::computePressureAndStiffness(Real& pressure, Real& stiffness, Real currentVolume, Real v0)
{
    if (currentVolume > 10*v0 || 10*currentVolume < v0)
    {
        msg_error() << "TOO MUCH VOLUME VARIATION.";
        pressure = 0;
        stiffness = 0;
    }
    const Real B = m_B.getValue();
    const Real gamma = m_gamma.getValue();
    const Real p0 = m_p0.getValue();
    if (B == 0 || gamma == 0 || v0 == 0)
    {
        stiffness = 0;
        pressure = p0;
    }
    else if (gamma == 1)
    {
        stiffness = -B/v0;
        pressure = p0-B*(currentVolume/v0 - 1);
    }
    else
    {
        if (currentVolume == 0)
            stiffness = 0;
        else
            stiffness = -B*gamma*((Real)pow(currentVolume/v0,gamma))/currentVolume;
        pressure = p0-B*((Real)pow(currentVolume/v0,gamma) - 1);
    }
}

} // namespace sofa::component::mechanicalload
