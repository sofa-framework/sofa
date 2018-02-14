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
#ifndef FRAME_FRAMEFIXEDCONSTRAINT_H
#define FRAME_FRAMEFIXEDCONSTRAINT_H

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include "AffineTypes.h"
#include "initFrame.h"

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using helper::vector;

using namespace sofa::defaulttype;
/** Attach given particles to their initial positions.
*/
template <class DataTypes>
class FrameFixedConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FrameFixedConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
    typedef Data<typename DataTypes::MatrixDeriv> DataMatrixDeriv;
    static const unsigned dimensions = DataTypes::Deriv::total_size;
    typedef Vec<dimensions, int> VecAllowed;

//protected:
//
//    template <class DataDeriv>
//    void projectResponseT(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataDeriv& dx);

public:
    Data<vector<unsigned> > f_index;   ///< Indices of the constrained frames
    Data<vector<VecAllowed > > f_allowed;  ///< Allowed displacements of the constrained frames
    Data<double> _drawSize; ///< 0 -> point based rendering, >0 -> radius of spheres

    FrameFixedConstraint();

    virtual ~FrameFixedConstraint();


    // -- Constraint interface
    void init();

    void projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& resData);
    void projectVelocity(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& vData);
    void projectPosition(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecCoord& xData);
    void projectJacobianMatrix(const core::MechanicalParams* /* PARAMS FIRST */, DataMatrixDeriv& ) {}

    void applyConstraint(defaulttype::BaseMatrix *, unsigned int /*offset*/) {}
    void applyConstraint(defaulttype::BaseVector *, unsigned int /*offset*/) {}

    // Handle topological changes
//        virtual void handleTopologyChange();

    virtual void draw(const core::visual::VisualParams* vparams);


protected :

//        sofa::core::topology::BaseMeshTopology* topology;

//        // Define TestNewPointFunction
//    static bool FCTestNewPointFunction(int, void*, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& );
//
//        // Define RemovalFunction
//        static void FCRemovalFunction ( int , void*);

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FRAME_FRAMEFIXEDCONSTRAINT_CPP)
extern template class SOFA_FRAME_API FrameFixedConstraint<Affine3dTypes>;
extern template class SOFA_FRAME_API FrameFixedConstraint<Quadratic3dTypes>;
extern template class SOFA_FRAME_API FrameFixedConstraint<Affine3fTypes>;
extern template class SOFA_FRAME_API FrameFixedConstraint<Quadratic3fTypes>;
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
