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
#ifndef SOFA_COMPONENT_MAPPING_FrameBlendingMapping_H
#define SOFA_COMPONENT_MAPPING_FrameBlendingMapping_H

#include "initFrame.h"
#include <sofa/core/Mapping.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/helper/SVector.h>

#include <vector>

#include <sofa/component/component.h>
#include <sofa/helper/OptionsGroup.h>

#include "AffineTypes.h"
#include "DeformationGradientTypes.h"

namespace sofa
{

namespace component
{

namespace mapping
{

using helper::vector;

/** This class is not supposed to be instanciated. It is only used to contain what is common between its derived classes.
  */
template <class TIn, class TOut>
class FrameBlendingMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(FrameBlendingMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut) );

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef Out DataTypes;

    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;
    typedef typename Out::SpatialCoord SpatialCoord;


    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;



protected:
    Data<unsigned int> nbRefs; // Number of primitives influencing each point.
    Data<vector<unsigned int> > repartition; // indices of primitives influencing each point.
    Data<vector<Real> > weights;
    Data<vector<SpatialCoord > > weightGradients;

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;


public:
    Data<bool> showBlendedFrame;
    Data<bool> showDefTensors;
    Data<bool> showDefTensorsValues;
    Data<double> showDefTensorScale;
    Data<unsigned int> showFromIndex;
    Data<bool> showWeights;
    Data<double> showGammaCorrection;
    Data<bool> showWeightsValues;
    Data<bool> showReps;
    Data<int> showValuesNbDecimals;
    Data<double> showTextScaleFactor;
    Data<bool> showGradients;
    Data<double> showGradientsScaleFactor;


public:
    FrameBlendingMapping (core::State<In>* from, core::State<Out>* to );
    virtual ~FrameBlendingMapping();

    void init();

    virtual void apply(typename Out::VecCoord& out, const typename In::VecCoord& in);
    virtual void applyJ(typename Out::VecDeriv& out, const typename In::VecDeriv& in);
    virtual void applyJT(typename In::VecDeriv& out, const typename Out::VecDeriv& in);
    virtual void applyJT(typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in);

    void draw();


    //   virtual bool inverseSkinning( InCoord& /*X0*/, InCoord& /*X*/, const InCoord& /*Xtarget*/){}

protected:
//	void M33toV6(Vec6 &v,const Mat33& M) const;
//	void QtoR(Mat33& M, const Quat& q) const;
//	void ComputeL(Mat76& L, const Quat& q) const;
//	void ComputeQ(Mat37& Q, const Quat& q, const Vec3& p) const;
//	void ComputeMa(Mat33& M, const Quat& q) const;
//	void ComputeMb(Mat33& M, const Quat& q) const;
//	void ComputeMc(Mat33& M, const Quat& q) const;
//	void ComputeMw(Mat33& M, const Quat& q) const;

};


///////////////////////////////////////////////////////////////////////////////
//                           Affine Specialization                           //
///////////////////////////////////////////////////////////////////////////////

using sofa::defaulttype::Affine3dTypes;
using sofa::defaulttype::Affine3fTypes;

#if defined(WIN32) && !defined(SOFA_BUILD_FRAME)
#ifndef SOFA_FLOAT
extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, sofa::defaulttype::DeformationGradient332dTypes >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, sofa::defaulttype::DeformationGradient332fTypes >;
//extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, ExtVec3fTypes >;
#endif //SOFA_DOUBLE
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Vec3fTypes >;
//extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, Vec3dTypes >;
#endif //SOFA_DOUBLE
#endif //SOFA_FLOAT
#endif





} // namespace mapping

} // namespace component

} // namespace sofa

#endif
