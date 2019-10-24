/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MISC_PARTICLESINK_H
#define SOFA_COMPONENT_MISC_PARTICLESINK_H

#include <SofaSphFluid/config.h>

#include <sofa/helper/system/config.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/RGBAColor.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <SofaBaseTopology/TopologySubsetData.inl>
#include <SofaBaseTopology/PointSetTopologyModifier.h>
#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/defaulttype/RGBAColor.h>
#include <vector>
#include <iterator>
#include <iostream>
#include <ostream>
#include <algorithm>


namespace sofa
{

namespace component
{

namespace misc
{

template<class TDataTypes>
class ParticleSink : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParticleSink,TDataTypes), SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef helper::vector<Real> VecDensity;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalModel;
    typedef helper::vector<unsigned int> SetIndexArray;

    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;

    Data<Deriv> planeNormal; ///< plane normal
    Data<Real> planeD0; ///< plane d coef at which particles acceleration is constrained to 0
    Data<Real> planeD1; ///< plane d coef at which particles are removed
    Data<defaulttype::RGBAColor> color; ///< plane color. (default=[0.0,0.5,0.2,1.0])
    Data<bool> showPlane; ///< enable/disable drawing of plane

    sofa::component::topology::PointSubsetData< SetIndexArray > fixed; ///< indices of fixed particles
protected:
    ParticleSink();

    virtual ~ParticleSink();
    
public:
    void init() override;
    
    virtual void animateBegin(double /*dt*/, double time);
    
    void projectResponse(const sofa::core::MechanicalParams* mparams, DataVecDeriv& dx) override; ///< project dx to constrained space
   
    void projectVelocity(const sofa::core::MechanicalParams* mparams, DataVecDeriv& v) override; ///< project dx to constrained space (dx models a velocity) override
  
    void projectPosition(const sofa::core::MechanicalParams* mparams, DataVecCoord& xData) override; ///< project x to constrained space (x models a position) override
   
    void projectJacobianMatrix(const sofa::core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    void draw(const core::visual::VisualParams* vparams) override;
};

#if !defined(SOFA_COMPONENT_MISC_PARTICLESINK_CPP)
extern template class SOFA_SPH_FLUID_API ParticleSink<sofa::defaulttype::Vec3Types>;
extern template class SOFA_SPH_FLUID_API ParticleSink<sofa::defaulttype::Vec2Types>;
#endif

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MISC_PARTICLESINK_H

