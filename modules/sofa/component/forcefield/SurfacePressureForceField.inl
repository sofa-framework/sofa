/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/core/visual/VisualParams.h>
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
void SurfacePressureForceField<DataTypes>::verifyDerivative(VecDeriv& v_plus, VecDeriv& v,  VecVec3DerivValues& DVval, VecVec3DerivIndices& DVind,
        const VecDeriv& Din)
{

    std::cout<<" enters verifyDerivative"<<std::endl;

    std::cout<<" verifyDerivative : vplus.size()="<<v_plus.size()<<"  v.size()="<<v.size()<<"  DVval.size()="<<DVval.size()<<" DVind.size()="<<DVind.size()<<"  Din.size()="<<Din.size()<<std::endl;


    for (unsigned int i=0; i<v.size(); i++)
    {

        Deriv DV;
        DV.clear();
        std::cout<<" DVnum["<<i<<"] ="<<v_plus[i]-v[i];

        for(unsigned int j=0; j<DVval[i].size(); j++)
        {
            DV+=DVval[i][j]*Din[ (DVind[i][j]) ];
        }
        std::cout<<" DVana["<<i<<"] = "<<DV<<" DVval[i].size() = "<<DVval[i].size()<<std::endl;

        /*
                for(unsigned int j=0; j<DVval[i].size(); j++)
                {
                    std::cout<<" M["<<DVind[i][j]<<"] ="<<DVval[i][j]<<std::endl;
                }
         */

    }

}


template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    VecCoord xPlus = x;
    VecDeriv fplus = f;




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
            addTriangleSurfacePressure(f,x,v,p, true);

            /*

                        VecDeriv Din;
                        Din.resize(x.size());
                       for (unsigned int i=0; i<Din.size(); i++)
                       {
                           Real i1,i2,i3;
                           i1=(Real)(i%3+1);
                           i2=-(Real)(i%2)+0.156;
                           i3=(Real)(i%5+2);
                           Din[i]=Deriv(0.0000123*i1,0.0000152*i2,0.00000981*i3);
                           xPlus[i]=x[i]+Din[i];
                       }
                       addTriangleSurfacePressure(fplus,xPlus,v,p, false);


                       verifyDerivative(fplus, f,  derivTriNormalValues,derivTriNormalIndices, Din);

            */


        }

        if (m_topology->getNbQuads() > 0)
        {
            addQuadSurfacePressure(f,x,v,p);
        }
    }

    d_f.endEdit();
}

template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv&  d_df , const DataVecDeriv&  d_dx )
{
    double kFactor     =   mparams->kFactor() ;
    VecDeriv& df       = *(d_df.beginEdit());
    const VecDeriv& dx =   d_dx.getValue()  ;


//    std::cout<<" addDForce computed on SurfacePressureForceField Size ="<<derivTriNormalIndices.size()<<std::endl;


    /*
    for (unsigned int i=0; i<derivTriNormalIndices.size(); i++)
    {
        Deriv DFtri;
        DFtri.clear();

        for (unsigned int j=0; j<derivTriNormalIndices[i].size(); j++)
        {
            unsigned int v = derivTriNormalIndices[i][j];
            DFtri += derivTriNormalValues[i][j] * dx[v];
        }

        DFtri*= kFactor;

        for (unsigned int j=0; j<derivTriNormalIndices[i].size(); j++)
        {
            unsigned int v = derivTriNormalIndices[i][j];
            df[v] += DFtri;
        }


    }
    */

    for (unsigned int i=0; i<derivTriNormalIndices.size(); i++)
    {

        for (unsigned int j=0; j<derivTriNormalIndices[i].size(); j++)
        {
            unsigned int v = derivTriNormalIndices[i][j];
            df[i] += (derivTriNormalValues[i][j] * dx[v])*kFactor;

        }

    }

    d_df.endEdit();


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
void SurfacePressureForceField<DataTypes>::addTriangleSurfacePressure(VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure, bool computeDerivatives)
{
    typedef BaseMeshTopology::Triangle Triangle;


    if(computeDerivatives)
    {
        derivTriNormalValues.clear();
        derivTriNormalValues.resize(x.size());
        derivTriNormalIndices.clear();
        derivTriNormalIndices.resize(x.size());

        for (unsigned int i=0; i<x.size(); i++)
        {
            derivTriNormalValues[i].clear();
            derivTriNormalIndices[i].clear();
        }

    }



    std::cout<<" addTriangleSurfacePressure x.size() = "<<x.size()<<std::endl;

    for (int i = 0; i < m_topology->getNbTriangles(); i++)
    {

        Triangle t = m_topology->getTriangle(i);





        if ( isInPressuredBox(x[t[0]]) && isInPressuredBox(x[t[1]]) && isInPressuredBox(x[t[2]]) )
        {


            Deriv ab = x[t[1]] - x[t[0]];
            Deriv ac = x[t[2]] - x[t[0]];
            Deriv bc = x[t[2]] - x[t[1]];

            Deriv p = (ab.cross(ac)) * (pressure / static_cast<Real>(6.0));


            if(computeDerivatives)
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

                for (unsigned int j=0; j<3; j++)
                {
                    derivTriNormalValues[t[j]].push_back( DcrossDA * (pressure / static_cast<Real>(6.0))  );
                    derivTriNormalValues[t[j]].push_back( DcrossDB * (pressure / static_cast<Real>(6.0))  );
                    derivTriNormalValues[t[j]].push_back( DcrossDC * (pressure / static_cast<Real>(6.0))  );

                    derivTriNormalIndices[t[j]].push_back( t[0] );
                    derivTriNormalIndices[t[j]].push_back( t[1] );
                    derivTriNormalIndices[t[j]].push_back( t[2] );
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



    }
}



template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addQuadSurfacePressure(VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure)
{
    typedef BaseMeshTopology::Quad Quad;

    std::cout<<" addQuadSurfacePressure "<<std::endl;

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
void SurfacePressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    if (vparams->displayFlags().getShowWireFrame())
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


    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_INL
