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
#ifndef SOFA_COMPONENT_MAPPING_TUBULARMAPPING_INL
#define SOFA_COMPONENT_MAPPING_TUBULARMAPPING_INL

#include <sofa/component/mapping/TubularMapping.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/behavior/MechanicalMapping.inl>
#include <sofa/core/behavior/MechanicalState.h>


namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
void TubularMapping<BasicMapping>::init()
{
    if (!m_radius.isSet())
    {
        this->getContext()->get(radiusContainer);
        sout << "get Radius Container" << std::endl;
        if(!radiusContainer)
            serr << "TubularMapping : No Radius defined" << sendl;
    }
    else sout << "get Radius tout court" << std::endl;

    this->BasicMapping::init();

}



template <class BasicMapping>
void TubularMapping<BasicMapping>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    // Propagation of positions from the input DOFs to the output DOFs

    unsigned int N = m_nbPointsOnEachCircle.getValue();
    double rho = m_radius.getValue();
    int peak = m_peak.getValue();

    out.resize(in.size() * N);
    rotatedPoints.resize(in.size() * N);

    Vec Y0;
    Vec Z0;
    Y0[0] = (Real) (0.0); Y0[1] = (Real) (1.0); Y0[2] = (Real) (0.0);
    Z0[0] = (Real) (0.0); Z0[1] = (Real) (0.0); Z0[2] = (Real) (1.0);

    for (unsigned int i=0; i<in.size(); i++)
    {
        if(radiusContainer)
            rho = radiusContainer->getPointRadius(i);

        // allows for peak at the beginning or at the end of the Tubular Mapping

        Real radius_rho = (Real) rho;
        if(peak>0)
        {
            int test= (int)i;
            if (test<peak)
            {
                double attenuation = (double)test/ (double)peak;
                radius_rho = (Real) (attenuation*rho);
            }
        }
        else
        {
            int test= (int) in.size()-(i+1) ;

            if (test < -peak)
            {
                double attenuation = -(double)test/(double)peak;
                radius_rho = (Real) (attenuation *rho);
            }

        }

        Vec curPos = in[i].getCenter();

        Mat rotation;
        in[i].writeRotationMatrix(rotation);

        Vec Y;
        Vec Z;

        Y = rotation * Y0;
        Z = rotation * Z0;

        for(unsigned int j=0; j<N; ++j)
        {

            rotatedPoints[i*N+j] = (Y*cos((Real) (2.0*j*M_PI/N)) + Z*sin((Real) (2.0*j*M_PI/N)))*((Real) radius_rho);
            Vec x = curPos + rotatedPoints[i*N+j];
            //sout << "INFO_print : TubularMapping  DO move point - j = " << j << " , curPos = " << curPos <<  " , x = " << x << sendl;

            out[i*N+j] = x;
        }
    }
}



template <class BasicMapping>
void TubularMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{

    // Propagation of velocities from the input DOFs to the output DOFs

    if(out.size() != rotatedPoints.size())
    {
        rotatedPoints.resize(out.size());
    }

    unsigned int N = m_nbPointsOnEachCircle.getValue();

    out.resize(in.size() * N);
    Deriv v,omega;

    for (unsigned int i=0; i<in.size(); i++)
    {
        v = in[i].getVCenter();
        omega = in[i].getVOrientation();

        for(unsigned int j=0; j<N; ++j)
        {

            out[i*N+j] = v - cross(rotatedPoints[i*N+j],omega);
            //sout << "INFO_print : TubularMapping  DO moveJ point - j = " << j << " , curPos = " << v <<  " , x = " << out[i*N+j] << sendl;
        }
    }
}


template <class BasicMapping>
void TubularMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    // usefull for a Mechanical Mapping that propagates forces from the output DOFs to the input DOFs

    //sout << "INFO_print : pass HERE applyJT !!!" << sendl;

    if(in.size() != rotatedPoints.size())
    {
        rotatedPoints.resize(in.size());
    }

    unsigned int N = m_nbPointsOnEachCircle.getValue();

    Deriv v,omega;

    for (unsigned int i=0; i<out.size(); i++)
    {
        for(unsigned int j=0; j<N; j++)
        {

            Deriv f = in[i*N+j];
            v += f;
            omega += cross(rotatedPoints[i*N+j],f);
        }

        out[i].getVCenter() += v;
        out[i].getVOrientation() += omega;
        //sout << "INFO_print : TubularMapping  DO moveJT point - i = " << i << sendl;
    }

}

template <class BasicMapping>
void TubularMapping<BasicMapping>::applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in)
{
    // useful for a Mechanical Mapping that propagates forces from the output DOFs to the input DOFs
    unsigned int N = m_nbPointsOnEachCircle.getValue();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                // index of the node
                const unsigned int iIn = colIt.index();
                const Deriv f = (Deriv) colIt.val();
                Deriv v, omega;
                v+=f;
                omega += cross(rotatedPoints[iIn],f);
                unsigned int Iout = iIn/N;
                InDeriv result(v, omega);

                o.addCol(Iout, result);
            }
        }
    }
}

/*
template <class BasicMapping>
void TubularMapping<BasicMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in)
{
	// usefull for a Mechanical Mapping that propagates forces from the output DOFs to the input DOFs

	//sout << "INFO_print : pass HERE applyJT !!!" << sendl;

	//std::cerr<< "INFO_print : pass HERE applyJT : numConstraint= " <<in.size()<<std::endl;

	unsigned int N = m_nbPointsOnEachCircle.getValue();


	int outSize = out.size();
	out.resize(in.size() + outSize); // we can accumulate in "out" constraints from several mappings

	//std::cerr<<"Resize ok"<<std::endl;
	for(unsigned int i=0; i<in.size(); i++)
	{
		OutConstraintIterator itIn;
		std::pair< OutConstraintIterator, OutConstraintIterator > iter=in[i].data();

		//std::cerr<<"iter ok"<<std::endl;


		for (itIn=iter.first;itIn!=iter.second;itIn++)
		{
			const unsigned int iIn = itIn->first;// index of the node
			const Deriv f = (Deriv) itIn->second;
			Deriv v, omega;

			//std::cerr<<"calcul omega"<<std::endl;
			v+=f;
			omega += cross(rotatedPoints[iIn],f);

			//std::cerr<<"omega ok"<<std::endl;

			unsigned int Iout = iIn/N;

			//std::cerr<<"io="<<io<<"  Iout="<<Iout<<std::endl;

			InDeriv result(v, omega);

			out[outSize+i].add(Iout, result);


		}
	}

	//std::cerr<< " applyJT ended !!!" <<std::endl;

}
*/



} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_TUBULARMAPPING_INL
