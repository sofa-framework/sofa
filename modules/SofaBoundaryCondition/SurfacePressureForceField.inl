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
#ifndef SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_INL

#include <SofaBoundaryCondition/SurfacePressureForceField.h>
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



template <class DataTypes>
SurfacePressureForceField<DataTypes>::SurfacePressureForceField():
    m_pressure(initData(&m_pressure, (Real)0.0, "pressure", "Pressure force per unit area")),
    m_min(initData(&m_min, Coord(), "min", "Lower bond of the selection box")),
    m_max(initData(&m_max, Coord(), "max", "Upper bond of the selection box")),
    m_triangleIndices(initData(&m_triangleIndices, "triangleIndices", "Indices of affected triangles")),
    m_quadIndices(initData(&m_quadIndices, "quadIndices", "Indices of affected quads")),
    m_pulseMode(initData(&m_pulseMode, false, "pulseMode", "Cyclic pressure application")),
    m_pressureLowerBound(initData(&m_pressureLowerBound, (Real)0.0, "pressureLowerBound", "Pressure lower bound force per unit area (active in pulse mode)")),
    m_pressureSpeed(initData(&m_pressureSpeed, (Real)0.0, "pressureSpeed", "Continuous pressure application in Pascal per second. Only active in pulse mode")),
    m_volumeConservationMode(initData(&m_volumeConservationMode, false, "volumeConservationMode", "Pressure variation follow the inverse of the volume variation")),
	m_useTangentStiffness(initData(&m_useTangentStiffness, true, "useTangentStiffness", "Whether (non-symmetric) stiffness matrix should be used")),
    m_defaultVolume(initData(&m_defaultVolume, (Real)-1.0, "defaultVolume", "Default Volume")),
    m_mainDirection(initData(&m_mainDirection, Deriv(), "mainDirection", "Main direction for pressure application")),
    m_drawForceScale(initData(&m_drawForceScale, (Real)0, "drawForceScale", "DEBUG: scale used to render force vectors"))
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
void SurfacePressureForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();
    /*
        VecCoord xPlus = x;
        VecDeriv fplus = f;
      */

//    const Data<VecCoord>* xRest= this->mstate->read(core::ConstVecCoordId::restPosition()) ;
//    const VecCoord &x0 = xRest->getValue();

    m_f.resize(f.size());  for(unsigned int i=0;i<m_f.size();i++) m_f[i].clear(); // store forces for visualization

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
		bool useStiffness=m_useTangentStiffness.getValue();
        // Triangles

        derivTriNormalValues.clear();
        derivTriNormalValues.resize(x.size());
        derivTriNormalIndices.clear();
        derivTriNormalIndices.resize(x.size());

        for (unsigned int i=0; i<x.size(); i++)
        {
            derivTriNormalValues[i].clear();
            derivTriNormalIndices[i].clear();
        }

        if (m_triangleIndices.getValue().size() > 0)
        {
            for (unsigned int i = 0; i < m_triangleIndices.getValue().size(); i++)
            {
                addTriangleSurfacePressure(m_triangleIndices.getValue()[i], m_f,x,v,p, useStiffness);
            }
        }
        else if (m_topology->getNbTriangles() > 0)
        {
            for (unsigned int i = 0; i < (unsigned int)m_topology->getNbTriangles(); i++)
            {
                Triangle t = m_topology->getTriangle(i);
                if ( isInPressuredBox(x[t[0]]) && isInPressuredBox(x[t[1]]) && isInPressuredBox(x[t[2]]) )
                {
                    addTriangleSurfacePressure(i, m_f,x,v,p, useStiffness);
                }
            }

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

        // Quads

        if (m_quadIndices.getValue().size() > 0)
        {
            for (unsigned int i = 0; i < m_quadIndices.getValue().size(); i++)
            {
                addQuadSurfacePressure(m_quadIndices.getValue()[i], m_f,x,v,p);
            }
        }
        else if (m_topology->getNbQuads() > 0)
        {
            for (unsigned int i = 0; i < (unsigned int)m_topology->getNbQuads(); i++)
            {
                Quad q = m_topology->getQuad(i);

                if ( isInPressuredBox(x[q[0]]) && isInPressuredBox(x[q[1]]) && isInPressuredBox(x[q[2]]) && isInPressuredBox(x[q[3]]) )
                {
                    addQuadSurfacePressure(i, m_f,x,v,p);
                }
            }
        }

    }

    for(unsigned int i=0;i<m_f.size();i++) f[i]+=m_f[i];
    d_f.endEdit();
}

template <class DataTypes>
void SurfacePressureForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv&  d_df , const DataVecDeriv&  d_dx )
{


	//    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
	Real kFactor = (Real)mparams->kFactor();
	if (m_useTangentStiffness.getValue()) {
		VecDeriv& df       = *(d_df.beginEdit());
		const VecDeriv& dx =   d_dx.getValue()  ;

		for (unsigned int i=0; i<derivTriNormalIndices.size(); i++)
		{

			for (unsigned int j=0; j<derivTriNormalIndices[i].size(); j++)
			{
				unsigned int v = derivTriNormalIndices[i][j];
				df[i] += (derivTriNormalValues[i][j] * dx[v])*kFactor;

			}

		} 
		//			for (unsigned int i=0;i<df.size();++i) {
		//			msg_info()<<"df["<<i<<"]= "<<df[i]<<std::endl;
		//		}
		d_df.endEdit();
	}


}


template<class DataTypes>
void SurfacePressureForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{


    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    Real kFact = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    return;

    const int N = Coord::total_size;
	if (m_useTangentStiffness.getValue()) {
		for (unsigned int i=0; i<derivTriNormalIndices.size(); i++)
		{

			for (unsigned int j=0; j<derivTriNormalIndices[i].size(); j++)
			{
				unsigned int v = derivTriNormalIndices[i][j];

				Mat33 Kiv = derivTriNormalValues[i][j];

				for (unsigned int l=0; l<3; l++)
				{
					for (unsigned int c=0; c<3; c++)
					{
						mat->add(offset + N * i + l, offset + N * v + c, kFact * Kiv[l][c]);


					}
				}
			}
		}
	}
}

template <class DataTypes>
typename SurfacePressureForceField<DataTypes>::Real SurfacePressureForceField<DataTypes>::computeMeshVolume(const VecDeriv& /*f*/, const VecCoord& x)
{
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Quad Quad;

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
void SurfacePressureForceField<DataTypes>::addTriangleSurfacePressure(unsigned int triId, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure, bool computeDerivatives)
{
    Triangle t = m_topology->getTriangle(triId);

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



template<class DataTypes>
void SurfacePressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
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


    helper::ReadAccessor<DataVecCoord> x = this->mstate->read(core::ConstVecCoordId::position());
    if (m_drawForceScale.getValue() && m_f.size()==x.size())
    {
        std::vector< defaulttype::Vector3 > points;
        const sofa::defaulttype::Vec<4,float> color(0,1,0.5,1);

        for (unsigned int i=0; i<x.size(); i++)
        {
            points.push_back(x[i]);
            points.push_back(x[i]+m_f[i]*m_drawForceScale.getValue());
        }
        vparams->drawTool()->drawLines(points, 1, color);
    }
#endif /* SOFA_NO_OPENGL */
}

#ifndef SOFA_FLOAT

template<>
void SurfacePressureForceField<defaulttype::Rigid3dTypes>::addDForce(const core::MechanicalParams* mparams , DataVecDeriv& d_df , const DataVecDeriv& d_dx)
{


	Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
	VecDeriv& df       = *(d_df.beginEdit());
	const VecDeriv& dx =   d_dx.getValue()  ;


	for (unsigned int i=0; i<derivTriNormalIndices.size(); i++)
	{

		for (unsigned int j=0; j<derivTriNormalIndices[i].size(); j++)
		{
			unsigned int v = derivTriNormalIndices[i][j];
			df[i].getVCenter() += (derivTriNormalValues[i][j] * dx[v].getVCenter())*kFactor;

		}

	}

	d_df.endEdit();


}

template <>
SurfacePressureForceField<defaulttype::Rigid3dTypes>::Real SurfacePressureForceField<defaulttype::Rigid3dTypes>::computeMeshVolume(const VecDeriv& /*f*/, const VecCoord& x)
{
	typedef core::topology::BaseMeshTopology::Triangle Triangle;
	typedef core::topology::BaseMeshTopology::Quad Quad;

	Real volume = 0;
	int i = 0;

	for (i = 0; i < m_topology->getNbTriangles(); i++)
	{
		Triangle t = m_topology->getTriangle(i);
		defaulttype::Rigid3dTypes::CPos ab = x[t[1]].getCenter() - x[t[0]].getCenter();
		defaulttype::Rigid3dTypes::CPos ac = x[t[2]].getCenter() - x[t[0]].getCenter();
		volume += (ab.cross(ac))[2] * (x[t[0]][2] + x[t[1]][2] + x[t[2]][2]) / static_cast<Real>(6.0);
	}

	for (i = 0; i < m_topology->getNbQuads(); i++)
	{
		Quad q = m_topology->getQuad(i);

		defaulttype::Rigid3dTypes::CPos ab = x[q[1]].getCenter() - x[q[0]].getCenter();
		defaulttype::Rigid3dTypes::CPos ac = x[q[2]].getCenter() - x[q[0]].getCenter();
		defaulttype::Rigid3dTypes::CPos ad = x[q[3]].getCenter() - x[q[0]].getCenter();

		volume += ab.cross(ac)[2] * (x[q[0]][2] + x[q[1]][2] + x[q[2]][2]) / static_cast<Real>(6.0);
		volume += ac.cross(ad)[2] * (x[q[0]][2] + x[q[2]][2] + x[q[3]][2]) / static_cast<Real>(6.0);
	}

	return volume;
}

template <>
void SurfacePressureForceField<defaulttype::Rigid3dTypes>::addTriangleSurfacePressure(unsigned int triId, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure, bool computeDerivatives)
{
	Triangle t = m_topology->getTriangle(triId);

	defaulttype::Rigid3dTypes::CPos ab = x[t[1]].getCenter() - x[t[0]].getCenter();
	defaulttype::Rigid3dTypes::CPos ac = x[t[2]].getCenter() - x[t[0]].getCenter();
	defaulttype::Rigid3dTypes::CPos bc = x[t[2]].getCenter() - x[t[1]].getCenter();

	defaulttype::Rigid3dTypes::CPos p = (ab.cross(ac)) * (pressure / static_cast<Real>(6.0));


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



	if (m_mainDirection.getValue().getVCenter() != defaulttype::Rigid3dTypes::CPos())
	{
		defaulttype::Rigid3dTypes::CPos n = ab.cross(ac);
		n.normalize();
		Real scal = n * m_mainDirection.getValue().getVCenter();
		p *= fabs(scal);
	}

	f[t[0]].getVCenter() += p;
	f[t[1]].getVCenter() += p;
	f[t[2]].getVCenter() += p;
}

template <>
void SurfacePressureForceField<defaulttype::Rigid3dTypes>::addQuadSurfacePressure(unsigned int quadId, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure)
{
	Quad q = m_topology->getQuad(quadId);

	defaulttype::Rigid3dTypes::CPos ab = x[q[1]].getCenter() - x[q[0]].getCenter();
	defaulttype::Rigid3dTypes::CPos ac = x[q[2]].getCenter() - x[q[0]].getCenter();
	defaulttype::Rigid3dTypes::CPos ad = x[q[3]].getCenter() - x[q[0]].getCenter();

	defaulttype::Rigid3dTypes::CPos p1 = (ab.cross(ac)) * (pressure / static_cast<Real>(6.0));
	defaulttype::Rigid3dTypes::CPos p2 = (ac.cross(ad)) * (pressure / static_cast<Real>(6.0));

	defaulttype::Rigid3dTypes::CPos p = p1 + p2;

	f[q[0]].getVCenter() += p;
	f[q[1]].getVCenter() += p1;
	f[q[2]].getVCenter() += p;
	f[q[3]].getVCenter() += p2;
}

template<>
void SurfacePressureForceField<defaulttype::Rigid3dTypes>::verifyDerivative(VecDeriv& /*v_plus*/, VecDeriv& /*v*/,  VecVec3DerivValues& /*DVval*/, VecVec3DerivIndices& /*DVind*/, const VecDeriv& /*Din*/)
{
}

template<>
void SurfacePressureForceField<defaulttype::Rigid3dTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
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


	helper::ReadAccessor<DataVecCoord> x = this->mstate->read(core::ConstVecCoordId::position());
	if (m_drawForceScale.getValue() && m_f.size()==x.size())
	{
		std::vector< defaulttype::Vector3 > points;
		const defaulttype::Vec<4,float> color(0,1,0.5,1);

		for (unsigned int i=0; i<x.size(); i++)
		{
			points.push_back(x[i].getCenter());
			points.push_back(x[i].getCenter()+m_f[i].getVCenter()*m_drawForceScale.getValue());
		}
		vparams->drawTool()->drawLines(points, 1, color);
	}
#endif /* SOFA_NO_OPENGL */
}

#endif

#ifndef SOFA_DOUBLE

template<>
void SurfacePressureForceField<defaulttype::Rigid3fTypes>::addDForce(const core::MechanicalParams* mparams , DataVecDeriv& d_df , const DataVecDeriv& d_dx)
{


	Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
	VecDeriv& df       = *(d_df.beginEdit());
	const VecDeriv& dx =   d_dx.getValue()  ;


	for (unsigned int i=0; i<derivTriNormalIndices.size(); i++)
	{

		for (unsigned int j=0; j<derivTriNormalIndices[i].size(); j++)
		{
			unsigned int v = derivTriNormalIndices[i][j];
			df[i].getVCenter() += (derivTriNormalValues[i][j] * dx[v].getVCenter())*kFactor;

		}

	}

	d_df.endEdit();


}

template <>
SurfacePressureForceField<defaulttype::Rigid3fTypes>::Real SurfacePressureForceField<defaulttype::Rigid3fTypes>::computeMeshVolume(const VecDeriv& /*f*/, const VecCoord& x)
{
	typedef core::topology::BaseMeshTopology::Triangle Triangle;
	typedef core::topology::BaseMeshTopology::Quad Quad;

	Real volume = 0;
	int i = 0;

	for (i = 0; i < m_topology->getNbTriangles(); i++)
	{
		Triangle t = m_topology->getTriangle(i);
		defaulttype::Rigid3fTypes::CPos ab = x[t[1]].getCenter() - x[t[0]].getCenter();
		defaulttype::Rigid3fTypes::CPos ac = x[t[2]].getCenter() - x[t[0]].getCenter();
		volume += (ab.cross(ac))[2] * (x[t[0]][2] + x[t[1]][2] + x[t[2]][2]) / static_cast<Real>(6.0);
	}

	for (i = 0; i < m_topology->getNbQuads(); i++)
	{
		Quad q = m_topology->getQuad(i);

		defaulttype::Rigid3fTypes::CPos ab = x[q[1]].getCenter() - x[q[0]].getCenter();
		defaulttype::Rigid3fTypes::CPos ac = x[q[2]].getCenter() - x[q[0]].getCenter();
		defaulttype::Rigid3fTypes::CPos ad = x[q[3]].getCenter() - x[q[0]].getCenter();

		volume += ab.cross(ac)[2] * (x[q[0]][2] + x[q[1]][2] + x[q[2]][2]) / static_cast<Real>(6.0);
		volume += ac.cross(ad)[2] * (x[q[0]][2] + x[q[2]][2] + x[q[3]][2]) / static_cast<Real>(6.0);
	}

	return volume;
}

template <>
void SurfacePressureForceField<defaulttype::Rigid3fTypes>::addTriangleSurfacePressure(unsigned int triId, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure, bool computeDerivatives)
{
	Triangle t = m_topology->getTriangle(triId);

	defaulttype::Rigid3fTypes::CPos ab = x[t[1]].getCenter() - x[t[0]].getCenter();
	defaulttype::Rigid3fTypes::CPos ac = x[t[2]].getCenter() - x[t[0]].getCenter();
	defaulttype::Rigid3fTypes::CPos bc = x[t[2]].getCenter() - x[t[1]].getCenter();

	defaulttype::Rigid3fTypes::CPos p = (ab.cross(ac)) * (pressure / static_cast<Real>(6.0));


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



	if (m_mainDirection.getValue().getVCenter() != defaulttype::Rigid3fTypes::CPos())
	{
		defaulttype::Rigid3fTypes::CPos n = ab.cross(ac);
		n.normalize();
		Real scal = n * m_mainDirection.getValue().getVCenter();
		p *= fabs(scal);
	}

	f[t[0]].getVCenter() += p;
	f[t[1]].getVCenter() += p;
	f[t[2]].getVCenter() += p;
}

template <>
void SurfacePressureForceField<defaulttype::Rigid3fTypes>::addQuadSurfacePressure(unsigned int quadId, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/, const Real& pressure)
{
	Quad q = m_topology->getQuad(quadId);

	defaulttype::Rigid3fTypes::CPos ab = x[q[1]].getCenter() - x[q[0]].getCenter();
	defaulttype::Rigid3fTypes::CPos ac = x[q[2]].getCenter() - x[q[0]].getCenter();
	defaulttype::Rigid3fTypes::CPos ad = x[q[3]].getCenter() - x[q[0]].getCenter();

	defaulttype::Rigid3fTypes::CPos p1 = (ab.cross(ac)) * (pressure / static_cast<Real>(6.0));
	defaulttype::Rigid3fTypes::CPos p2 = (ac.cross(ad)) * (pressure / static_cast<Real>(6.0));

	defaulttype::Rigid3fTypes::CPos p = p1 + p2;

	f[q[0]].getVCenter() += p;
	f[q[1]].getVCenter() += p1;
	f[q[2]].getVCenter() += p;
	f[q[3]].getVCenter() += p2;
}

template<>
void SurfacePressureForceField<defaulttype::Rigid3fTypes>::verifyDerivative(VecDeriv& /*v_plus*/, VecDeriv& /*v*/,  VecVec3DerivValues& /*DVval*/, VecVec3DerivIndices& /*DVind*/, const VecDeriv& /*Din*/)
{
}

template<>
void SurfacePressureForceField<defaulttype::Rigid3fTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
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


	helper::ReadAccessor<DataVecCoord> x = this->mstate->read(core::ConstVecCoordId::position());
	if (m_drawForceScale.getValue() && m_f.size()==x.size())
	{
		std::vector< defaulttype::Vector3 > points;
		const defaulttype::Vec<4,float> color(0,1,0.5,1);

		for (unsigned int i=0; i<x.size(); i++)
		{
			points.push_back(x[i].getCenter());
			points.push_back(x[i].getCenter()+m_f[i].getVCenter()*m_drawForceScale.getValue());
		}
		vparams->drawTool()->drawLines(points, 1, color);
	}
#endif /* SOFA_NO_OPENGL */
}

#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_INL
