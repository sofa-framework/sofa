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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FLEXIBLE_BaseJacobian_H
#define FLEXIBLE_BaseJacobian_H

#include <sofa/defaulttype/Mat.h>

namespace sofa
{

namespace defaulttype
{


/** Template class used to implement one jacobian block
*/
template<class TIn, class TOut>
class BaseJacobianBlock
{
public:
    typedef TIn In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Real Real;

    typedef TOut Out;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;

    typedef Mat<Out::deriv_total_size,In::deriv_total_size,Real> MatBlock;

    // Called in Apply
    virtual void addapply( OutCoord& result, const InCoord& data )=0;
    // Called in ApplyJ
    virtual void addmult( OutDeriv& result,const InDeriv& data )=0;
    // Called in ApplyJT
    virtual void addMultTranspose( InDeriv& result, const OutDeriv& data )=0;
    // Called in getJ
    virtual MatBlock getJ()=0;
};




} // namespace defaulttype
} // namespace sofa



#endif
