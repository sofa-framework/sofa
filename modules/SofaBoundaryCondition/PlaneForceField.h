/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class PlaneForceFieldInternalData
{
public:
};

template<class DataTypes>
class PlaneForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PlaneForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));
    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

	typedef typename DataTypes::CPos CPos;
	typedef typename DataTypes::DPos DPos;

protected:
    sofa::helper::vector<unsigned int> contacts;

    PlaneForceFieldInternalData<DataTypes> data;

public:

    Data<DPos> planeNormal;
    Data<Real> planeD;
    Data<Real> stiffness;
    Data<Real> damping;
	Data<Real> maxForce;
    Data<defaulttype::Vec3f> color;
    Data<bool> bDraw;
    Data<Real> drawSize;


    /// optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)
    Data< defaulttype::Vec<2,int> > localRange;

    /// option bilateral : if true, the force field is applied on both side of the plane
   Data<bool> bilateral;
protected:
    PlaneForceField()
        : planeNormal(initData(&planeNormal, "normal", "plane normal"))
        , planeD(initData(&planeD, (Real)0, "d", "plane d coef"))
        , stiffness(initData(&stiffness, (Real)500, "stiffness", "force stiffness"))
        , damping(initData(&damping, (Real)5, "damping", "force damping"))
        , maxForce(initData(&maxForce, (Real)0, "maxForce", "if non-null, the max force that can be applied to the object"))
		, color(initData(&color, defaulttype::Vec3f(0.0f,.5f,.2f), "color", "plane color"))
        , bDraw(initData(&bDraw, false, "draw", "enable/disable drawing of plane"))
        , drawSize(initData(&drawSize, (Real)10.0f, "drawSize", "plane display size if draw is enabled"))
        , localRange( initData(&localRange, defaulttype::Vec<2,int>(-1,-1), "localRange", "optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)" ) )
        , bilateral( initData(&bilateral, false, "bilateral", "if true the plane force field is applied on both sides"))
    {
		Deriv n;
		DataTypes::set(n, 0, 1, 0);
        planeNormal.setValue(DataTypes::getDPos(n));
    }
public:
    void setPlane(const Deriv& normal, Real d);

    void setMState(  core::behavior::MechanicalState<DataTypes>* mstate ) { this->mstate = mstate; }

    void setStiffness(Real stiff)
    {
        stiffness.setValue( stiff );
    }

    void setDamping(Real damp)
    {
        damping.setValue( damp );
    }

    void rotate( Deriv axe, Real angle ); // around the origin (0,0,0)


    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx);

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    virtual void updateStiffness( const VecCoord& x );

    virtual void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix );

    void draw(const core::visual::VisualParams* vparams);
    void drawPlane(const core::visual::VisualParams*, float size=0.0f);
    void computeBBox(const core::ExecParams *, bool onlyVisible=false);

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<defaulttype::Vec2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<defaulttype::Vec1dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<defaulttype::Vec6dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<defaulttype::Vec2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<defaulttype::Vec1fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<defaulttype::Vec6fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
