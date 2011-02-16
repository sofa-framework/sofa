/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. CoIn, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "PendulumMapping.h"
#include <sofa/simulation/common/Simulation.h>


namespace sofa
{

namespace component
{

namespace mapping
{

using helper::ReadAccessor;
using helper::WriteAccessor;
using defaulttype::Vec;
using defaulttype::Vector3;


template <class In, class Out>
PendulumMapping<In,Out>::PendulumMapping(core::State<In>* from, core::State<Out>* to)
    : Inherit ( from, to )
    , f_length ( initData ( &f_length,"lengths","distances from the fixed point to the end of the pendulum" ) )
{
}


template <class In, class Out>
PendulumMapping<In,Out>::~PendulumMapping()
{
}

template <class In, class Out>
void PendulumMapping<In,Out>::init()
{
    ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    WriteAccessor<Data<VecInCoord> > in (*this->fromModel->write(core::VecCoordId::position()));
    WriteAccessor<Data<vector<OutReal> > > distances (f_length);
    if( distances.size() != out.size() ) // values not read from file
    {
        in.resize(out.size());
        distances.resize(out.size());
        gap.resize(out.size());
        for(unsigned i=0; i<out.size(); i++)
        {
            typename Out::Real x,y,z;
            Out::get( x,y,z, out[i] );
            in[i][0] = (InReal) atan2(y,x);
            distances[i] = sqrt(x*x+y*y);
        }
    }
}

template <class In, class Out>
void PendulumMapping<In,Out>::draw()
{
    if ( !this->getShow() ) return;

    ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    std::vector< Vector3 > points(out.size());

    for ( unsigned int i=0; i<out.size(); i++ )
    {
        points[i] = Out::getCPos(out[i]);
    }
    simulation::getSimulation()->DrawUtility().drawPoints ( points, 7, Vec<4,float> ( 1,1,0,1 ) );

    points.resize(2*out.size());
    for ( unsigned int i=0; i<out.size(); i++ )
    {
        points[2*i] =   Vector3(0,0,0);
        points[2*i+1] = Out::getCPos(out[i]);
    }
    simulation::getSimulation()->DrawUtility().drawLines ( points, 1, Vec<4,float> ( 0,1,0,1 ) );


}

template <class In, class Out>
void PendulumMapping<In,Out>::apply( VecOutCoord& childPos, const VecInCoord& parentPos)
{
    ReadAccessor<Data<vector<OutReal> > > distances (f_length);
    for(unsigned i=0; i<childPos.size(); i++)
    {
        gap[i] = Vec2(distances[i]*cos(parentPos[i][0]) ,distances[i]*sin(parentPos[i][0])  );
        Out::set( childPos[i], gap[i][0], gap[i][1], (OutReal) 0 );
    }
}

template <class In, class Out>
void PendulumMapping<In,Out>::applyJ( VecOutDeriv& childVel, const VecInDeriv& parentVel)
{
    for(unsigned i=0; i<childVel.size(); i++)
    {
        // velocity is orthogonal to the radius and proportional with the angular velocity
        Out::set( childVel[i], (OutReal)( -gap[i][1] * parentVel[i][0] ), (OutReal)( gap[i][0] * parentVel[i][0] ), (OutReal) 0 );
    }
}

template <class In, class Out>
void PendulumMapping<In,Out>::applyJT( VecInDeriv& parentForce, const VecOutDeriv& childForce)
{
    for(unsigned i=0; i<parentForce.size(); i++)
    {
        // convert force to torque
        parentForce[i][0] += -gap[i][1] * childForce[i][0]  + gap[i][0] * childForce[i][1] ;
    }
}

template <class In, class Out>
void PendulumMapping<In,Out>::applyJT( MatrixInDeriv& parentJacobians, const MatrixOutDeriv& childJacobians )
{
    for (typename Out::MatrixDeriv::RowConstIterator childJacobian = childJacobians.begin(); childJacobian != childJacobians.end(); ++childJacobian)
    {
        typename In::MatrixDeriv::RowIterator parentJacobian = parentJacobians.writeLine(childJacobian.index());

        for (typename Out::MatrixDeriv::ColConstIterator childParticle = childJacobian.begin(); childParticle != childJacobian.end(); ++childParticle)
        {
            unsigned int childIndex = childParticle.index();
            const OutDeriv& childJacobianVec = childParticle.val();

            parentJacobian.addCol(childIndex, InDeriv(-gap[childIndex][1] * childJacobianVec[0]  + gap[childIndex][0] * childJacobianVec[1]) ) ;

        }
    }
}

template <class In, class Out>
void PendulumMapping<In,Out>::applyDJT(const core::MechanicalParams* mparams /* PARAMS FIRST */, core::MultiVecDerivId parentForceChangeId, core::ConstMultiVecDerivId )
{

    ReadAccessor<Data<VecOutDeriv> > childForce (*mparams->readF(this->toModel));
    WriteAccessor<Data<VecInDeriv> > parentForce (*parentForceChangeId[this->fromModel].write());
    ReadAccessor<Data<VecInDeriv> > parentDx (*mparams->readDx(this->fromModel));
    InReal kfactor = mparams->kFactor();

//    serr<<"PendulumMapping2<In,Out>::applyDJT"<< sendl;
    for(unsigned i=0; i<parentForce.size(); i++)
    {
        parentForce[i][0] -= ( gap[i][0] * childForce[i][0] +  gap[i][1] * childForce[i][1] ) * parentDx[i][0] * kfactor;
//        serr<<"PendulumMapping2<In,Out>::applyDJT, gap[i] = "<< gap[i] << sendl;
//        serr<<"PendulumMapping2<In,Out>::applyDJT, childForce[i] = "<< childForce[i] << sendl;
//        serr<<"PendulumMapping2<In,Out>::applyDJT, parent displacement = "<< parentDx[i][0] << sendl;
//        serr<<"PendulumMapping2<In,Out>::applyDJT, parent force -= "<< ( gap[i][0] * childForce[i][0] +  gap[i][1] * childForce[i][1] ) * parentDx[i][0] << sendl;
    }
}


}	//mapping

}	//component

}	//sofa

