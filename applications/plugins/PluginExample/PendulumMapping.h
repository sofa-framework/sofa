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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_COMPONENT_CONSTRAINT_PendulumMapping_H
#define SOFA_COMPONENT_CONSTRAINT_PendulumMapping_H


#include <sofa/core/Mapping.h>
#include <sofa/component/component.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/defaulttype/Vec.h>




namespace sofa
{

namespace component
{

namespace mapping
{
using helper::vector;
using defaulttype::Vec;

/** input: pendulum angle; output: coordinates of the endpoint of the pendulum
  */

template <class TIn, class TOut>
class PendulumMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS( SOFA_TEMPLATE2(PendulumMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut) );
    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename In::Real InReal;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::MatrixDeriv MatrixInDeriv;
    typedef typename Out::Real OutReal;
    typedef typename Out::VecCoord VecOutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecDeriv VecOutDeriv;
    typedef typename Out::MatrixDeriv MatrixOutDeriv;

    PendulumMapping(core::State<In>* from, core::State<Out>* to );
    ~PendulumMapping();

    Data<vector<OutReal> > f_length;

    virtual void init();
    virtual void draw();

    virtual void apply(VecOutCoord& out, const VecInCoord& in);
    virtual void applyJ( VecOutDeriv& out, const VecInDeriv& in);
    virtual void applyJT( VecInDeriv& out, const VecOutDeriv& in);
    virtual void applyJT( MatrixInDeriv& out, const MatrixOutDeriv& in);


protected:
    typedef Vec<2,OutReal> Vec2;
    vector<Vec2> gap;

private:

};


}

}

}



#endif
