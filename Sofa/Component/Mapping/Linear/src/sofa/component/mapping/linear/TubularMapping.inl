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

#include <sofa/component/mapping/linear/TubularMapping.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
TubularMapping<TIn, TOut>::TubularMapping ( )
    : Inherit ( )
    , m_nbPointsOnEachCircle( initData(&m_nbPointsOnEachCircle, "nbPointsOnEachCircle", "Discretization of created circles"))
    , m_radius( initData(&m_radius, "radius", "Radius of created circles"))
    , m_peak (initData(&m_peak, 0, "peak", "=0 no peak, =1 peak on the first segment =2 peak on the two first segment, =-1 peak on the last segment"))
{
}
template <class TIn, class TOut>
void TubularMapping<TIn, TOut>::init()
{
    if (!m_radius.isSet())
    {
        msg_error() << "No Radius defined";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    Inherit::init();

}

template <class TIn, class TOut>
void TubularMapping<TIn, TOut>::apply ( const core::MechanicalParams* /* mparams */, OutDataVecCoord& dOut, const InDataVecCoord& dIn)
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    // Propagation of positions from the input DOFs to the output DOFs
    const InVecCoord& in = dIn.getValue();
    OutVecCoord& out = *dOut.beginEdit();

    unsigned int N = m_nbPointsOnEachCircle.getValue();
    double rho = m_radius.getValue();
    const int peak = m_peak.getValue();

    out.resize(in.size() * N);
    rotatedPoints.resize(in.size() * N);

    Vec Y0;
    Vec Z0;
    Y0[0] = (Real) (0.0); Y0[1] = (Real) (1.0); Y0[2] = (Real) (0.0);
    Z0[0] = (Real) (0.0); Z0[1] = (Real) (0.0); Z0[2] = (Real) (1.0);

    for (unsigned int i=0; i<in.size(); i++)
    {
        // allows for peak at the beginning or at the end of the Tubular Mapping

        Real radius_rho = (Real) rho;
        if(peak>0)
        {
            const int test= (int)i;
            if (test<peak)
            {
                const double attenuation = (double)test/ (double)peak;
                radius_rho = (Real) (attenuation*rho);
            }
        }
        else
        {
            const int test= (int) in.size()-(i+1) ;

            if (test < -peak)
            {
                const double attenuation = -(double)test/(double)peak;
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
            out[i*N+j] = x;
        }
    }
    dOut.endEdit();
}



template <class TIn, class TOut>
void TubularMapping<TIn, TOut>::applyJ( const core::MechanicalParams* /* mparams */, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    // Propagation of velocities from the input DOFs to the output DOFs
    const InVecDeriv& in = dIn.getValue();
    OutVecDeriv& out = *dOut.beginEdit();

    unsigned int N = m_nbPointsOnEachCircle.getValue();

    out.resize(in.size() * N);
    OutDeriv v,omega;

    if(out.size() != rotatedPoints.size())
    {
        rotatedPoints.resize(out.size());
    }

    for (unsigned int i=0; i<in.size(); i++)
    {
        v = getVCenter(in[i]);
        omega = getVOrientation(in[i]);

        for(unsigned int j=0; j<N; ++j)
        {
            out[i*N+j] = v - cross(rotatedPoints[i*N+j],omega);
        }
    }

    dOut.endEdit();
}


template <class TIn, class TOut>
void TubularMapping<TIn, TOut>::applyJT( const core::MechanicalParams* /* mparams */, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    // useful for a Mechanical Mapping that propagates forces from the output DOFs to the input DOFs
    const OutVecDeriv& in = dIn.getValue();
    InVecDeriv& out = *dOut.beginEdit();

    if(in.size() != rotatedPoints.size())
    {
        rotatedPoints.resize(in.size());
    }

    const unsigned int N = m_nbPointsOnEachCircle.getValue();

    OutDeriv v,omega;

    for (unsigned int i=0; i<out.size(); i++)
    {
        for(unsigned int j=0; j<N; j++)
        {

            OutDeriv f = in[i*N+j];
            v += f;
            omega += cross(rotatedPoints[i*N+j],f);
        }

        getVCenter(out[i]) += v;
        getVOrientation(out[i]) += omega;
    }

    dOut.endEdit();
}

template <class TIn, class TOut>
void TubularMapping<TIn, TOut>::applyJT( const core::ConstraintParams * /*cparams*/, InDataMatrixDeriv& dOut, const OutDataMatrixDeriv& dIn)
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    // useful for a Mechanical Mapping that propagates forces from the output DOFs to the input DOFs
    const OutMatrixDeriv& in = dIn.getValue();
    InMatrixDeriv& out = *dOut.beginEdit();

    const unsigned int N = m_nbPointsOnEachCircle.getValue();

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
                const OutDeriv f = (OutDeriv) colIt.val();
                OutDeriv v, omega;
                v+=f;
                omega += cross(rotatedPoints[iIn],f);
                unsigned int Iout = iIn/N;
                InDeriv result(v, omega);

                o.addCol(Iout, result);
            }
        }
    }

    dOut.endEdit();
}

} // namespace sofa::component::mapping::linear
