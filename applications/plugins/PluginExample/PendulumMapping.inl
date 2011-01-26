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


namespace sofa
{

namespace component
{

namespace mapping
{



template <class In, class Out>
PendulumMapping<In,Out>::PendulumMapping(core::State<In>* from, core::State<Out>* to)
    : Inherit ( from, to )
{
}


template <class In, class Out>
PendulumMapping<In,Out>::~PendulumMapping()
{
}

template <class In, class Out>
void PendulumMapping<In,Out>::init()
{
}

template <class In, class Out>
void PendulumMapping<In,Out>::draw()
{
}

template <class In, class Out>
void PendulumMapping<In,Out>::apply(typename Out::VecCoord& out, const typename In::VecCoord& in)
{
}

template <class In, class Out>
void PendulumMapping<In,Out>::applyJ(typename Out::VecDeriv& out, const typename In::VecDeriv& in)
{
}

template <class In, class Out>
void PendulumMapping<In,Out>::applyJT(typename In::VecDeriv& out, const typename Out::VecDeriv& in)
{
}

template <class In, class Out>
void PendulumMapping<In,Out>::applyJT(typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in)
{
}



}	//mapping

}	//component

}	//sofa

