/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_INL

#include <sofa/component/forcefield/SurfacePressureForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/gl/template.h>
#include <vector>
#include <set>
#include <iostream>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace core::topology;


template <class DataTypes>
SurfacePressureForceField<DataTypes>::SurfacePressureForceField():
    m_pressure(initData(&m_pressure, (Real)0.0, "pressure", "Pressure force per unit area")),
    m_min(initData(&m_min, Coord(), "min", "Lower bond of the selection box")),
    m_max(initData(&m_max, Coord(), "max", "Upper bond of the selection box")),
    m_pulseMode(initData(&m_pulseMode, false, "pulseMode", "Cyclic pressure application")),
    m_pressureLowerBound(initData(&m_pressureLowerBound, (Real)0.0, "pressureLowerBound", "Pressure lower bound force per unit area (active in pulse mode)")),
    m_pressureSpeed(initData(&m_pressureSpeed, (Real)0.0, "pressureSpeed", "Continuous pressure application in Pascal per second. Only active in pulse mode")),
    m_volumeConservationMode(initData(&m_volumeConservationMode, false, "volumeConservationMode", "Pressure variation follow the inverse of the volume variation")),
    m_defaultVolume(initData(&m_defaultVolume, (Real)-1.0, "defaultVolume", "Default Volume")),
    m_mainDirection(initData(&m_mainDirection, Deriv(), "mainDirection", "Main direction for pressure application"))
{

}



template <class DataTypes>
SurfacePressureForceField<DataTypes>::~SurfacePressureForceField()
{

}



template <class DataTypes>
void SurfacePressureForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();
    m_topology = this->getContext()->getMeshTopology();

    state = ( m_pressure.getValue() > 0 ) ? INCREASE : DECREASE;

    if (m_pulseMode.getValue() && (m_pressureSpeed.getValue() == 0.0))
    {
        serr<<"Default pressure speed value has been set in SurfacePressureForceField" << sendl;
        m_pressureSpeed.setValue((Real)fabs( m_pressure.getValue()));
    }

    m_pulseModePressure = 0.0;
}



template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addForce(DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v, const core::MechanicalParams* /* mparams */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    Real p = m_pulseMode.getValue() ? computePulseModePressure() : m_pressure.getValue();

    if (m_topology)
    {
        if (m_volumeConservationMode.getValue())
        {
            if (m_defaultVolume.getValue() == -1)
            {
                m_defaultVolume.setValue(computeMeshVolume(f,x));
            }
            else if (m_defaultVolume.getValue() != 0)
            {
                p *= m_defaultVolume.getValue() / computeMeshVolume(f,x);
            }
        }

        if (m_topology->getNbTriangles() > 0)
        {
            addTriangleSurfacePressure(f,x,v,p);
        }

        if (m_topology->getNbQuads() > 0)
        {
            addQuadSurfacePressure(f,x,v,p);
        }
    }

    d_f.endEdit();
}


template <class DataTypes>
typename SurfacePressureForceField<DataTypes>::Real SurfacePressureForceField<DataTypes>::computeMeshVolume(const VecDeriv& /*f*/, const VecCoord& x)
{
    typedef BaseMeshTopology::Triangle Triangle;
    typedef BaseMeshTopology::Quad Quad;

    Real volume = 0;
    int i = 0;

    for (i = 0; i < m_topology->getNbTriangles(); i++)
    {
        Triangle t = m_topology->getTriangle(i);
        Deriv ab = x[t[1]] - x[t[0]];
        Deriv ac = x[t[2]] - x[t[0]];
        volume += (ab.cross(ac))[2] * (x[t[0]][2] + x[t[1]][2] + x[t[2]][2]) / static_cast<Real>(6.0);
    }

    for (i = 0; i < m_topology->getNbQuads(); i++)
    {
        Quad q = m_topology->getQuad(i);

        Deriv ab = x[q[1]] - x[q[0]];
        Deriv ac = x[q[2]] - x[q[0]];
        Deriv ad = x[q[3]] - x[q[0]];

        volume += ab.cross(ac)[2] * (x[q[0]][2] + x[q[1]][2] + x[q[2]][2]) / static_cast<Real>(6.0);
        volume += ac.cross(ad)[2] * (x[q[0]][2] + x[q[2]][2] + x[q[3]][2]) / static_cast<Real>(6.0);
    }

    return volume;
}


template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addTriangleSurfacePressure(VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure)
{
    typedef BaseMeshTopology::Triangle Triangle;

    for (int i = 0; i < m_topology->getNbTriangles(); i++)
    {
        Triangle t = m_topology->getTriangle(i);

        if ( isInPressuredBox(x[t[0]]) && isInPressuredBox(x[t[1]]) && isInPressuredBox(x[t[2]]) )
        {

            Deriv ab = x[t[1]] - x[t[0]];
            Deriv ac = x[t[2]] - x[t[0]];

            Deriv p = (ab.cross(ac)) * (pressure / static_cast<Real>(6.0));

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
    }
}



template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addQuadSurfacePressure(VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure)
{
    typedef BaseMeshTopology::Quad Quad;

    for (int i = 0; i < m_topology->getNbQuads(); i++)
    {
        Quad q = m_topology->getQuad(i);

        if ( isInPressuredBox(x[q[0]]) && isInPressuredBox(x[q[1]]) && isInPressuredBox(x[q[2]]) && isInPressuredBox(x[q[3]]) )
        {
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

    }
}



template <class DataTypes>
bool SurfacePressureForceField<DataTypes>::isInPressuredBox(const Coord &x) const
{
    if ( (m_max == Coord()) && (m_min == Coord()) )
        return true;

    return ( (x[0] >= m_min.getValue()[0])
            && (x[0] <= m_max.getValue()[0])
            && (x[1] >= m_min.getValue()[1])
            && (x[1] <= m_max.getValue()[1])
            && (x[2] >= m_min.getValue()[2])
            && (x[2] <= m_max.getValue()[2]) );
}

template<class DataTypes>
const typename SurfacePressureForceField<DataTypes>::Real SurfacePressureForceField<DataTypes>::computePulseModePressure()
{
    double dt = this->getContext()->getDt();

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



template<class DataTypes>
void SurfacePressureForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    glDisable(GL_LIGHTING);

    glColor4f(0.f,0.8f,0.3f,1.f);

    glBegin(GL_LINE_LOOP);
    glVertex3d(m_min.getValue()[0],m_min.getValue()[1],m_min.getValue()[2]);
    glVertex3d(m_max.getValue()[0],m_min.getValue()[1],m_min.getValue()[2]);
    glVertex3d(m_max.getValue()[0],m_min.getValue()[1],m_max.getValue()[2]);
    glVertex3d(m_min.getValue()[0],m_min.getValue()[1],m_max.getValue()[2]);
    glEnd();

    glBegin(GL_LINE_LOOP);
    glVertex3d(m_min.getValue()[0],m_max.getValue()[1],m_min.getValue()[2]);
    glVertex3d(m_max.getValue()[0],m_max.getValue()[1],m_min.getValue()[2]);
    glVertex3d(m_max.getValue()[0],m_max.getValue()[1],m_max.getValue()[2]);
    glVertex3d(m_min.getValue()[0],m_max.getValue()[1],m_max.getValue()[2]);
    glEnd();

    glBegin(GL_LINES);
    glVertex3d(m_min.getValue()[0],m_min.getValue()[1],m_min.getValue()[2]);
    glVertex3d(m_min.getValue()[0],m_max.getValue()[1],m_min.getValue()[2]);

    glVertex3d(m_max.getValue()[0],m_min.getValue()[1],m_min.getValue()[2]);
    glVertex3d(m_max.getValue()[0],m_max.getValue()[1],m_min.getValue()[2]);

    glVertex3d(m_max.getValue()[0],m_min.getValue()[1],m_max.getValue()[2]);
    glVertex3d(m_max.getValue()[0],m_max.getValue()[1],m_max.getValue()[2]);

    glVertex3d(m_min.getValue()[0],m_min.getValue()[1],m_max.getValue()[2]);
    glVertex3d(m_min.getValue()[0],m_max.getValue()[1],m_max.getValue()[2]);
    glEnd();


    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_INL
