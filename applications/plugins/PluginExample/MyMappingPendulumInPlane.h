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
#ifndef PLUGINEXAMPLE_MYMAPPINGPENDULUMINPLANE_H
#define PLUGINEXAMPLE_MYMAPPINGPENDULUMINPLANE_H

#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/OptionsGroup.h>


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
class MyMappingPendulumInPlane: public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(MyMappingPendulumInPlane, TIn, TOut), SOFA_TEMPLATE2(core::Mapping, TIn, TOut));
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

    typedef Data<VecInCoord> InDataVecCoord;
    typedef Data<VecInDeriv> InDataVecDeriv;
    typedef Data<MatrixInDeriv> InDataMatrixDeriv;

    typedef Data<VecOutCoord> OutDataVecCoord;
    typedef Data<VecOutDeriv> OutDataVecDeriv;
    typedef Data<MatrixOutDeriv> OutDataMatrixDeriv;

protected:
    MyMappingPendulumInPlane();
    ~MyMappingPendulumInPlane();

public:
    Data<vector<OutReal> > f_length; ///< distances from the fixed point to the end of the pendulum

    virtual void init();
    virtual void draw(const core::visual::VisualParams*);

    virtual void apply(const core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in);
    virtual void applyJ(const core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in);
    virtual void applyJT(const core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in);
    virtual void applyJT(const core::ConstraintParams* mparams, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in);
    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForceChange, core::ConstMultiVecDerivId);

protected:
    typedef Vec<2, OutReal> Vec2;
    vector<Vec2> gap;
};

} // namespace mapping
} // namespace component
} // namespace sofa

#endif // PLUGINEXAMPLE_MYMAPPINGPENDULUMINPLANE_H
