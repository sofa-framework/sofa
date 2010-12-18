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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
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
//        typedef typename DataTypes::Coord Coord;
//        typedef typename DataTypes::Deriv Deriv;
    static const unsigned dimensions = DataTypes::Deriv::total_size;
    typedef Vec<dimensions, int> VecAllowed;

protected:

    template <class DataDeriv>
    void projectResponseT(DataDeriv& dx, const core::MechanicalParams* mparams);

public:
    Data<vector<unsigned> > f_index;   ///< Indices of the constrained frames
    Data<vector<VecAllowed > > f_allowed;  ///< Allowed displacements of the constrained frames
    Data<double> _drawSize;

    FrameFixedConstraint();

    virtual ~FrameFixedConstraint();


    // -- Constraint interface
    void init();

    void projectResponse(DataVecDeriv& resData, const core::MechanicalParams* mparams);
    void projectVelocity(DataVecDeriv& vData, const core::MechanicalParams* mparams);
    void projectPosition(DataVecCoord& xData, const core::MechanicalParams* mparams);
    void projectJacobianMatrix(DataMatrixDeriv& , const core::MechanicalParams* ) {}

    void applyConstraint(defaulttype::BaseMatrix *, unsigned int /*offset*/) {}
    void applyConstraint(defaulttype::BaseVector *, unsigned int /*offset*/) {}

    // Handle topological changes
//        virtual void handleTopologyChange();

    virtual void draw();


protected :

//        sofa::core::topology::BaseMeshTopology* topology;

//        // Define TestNewPointFunction
//    static bool FCTestNewPointFunction(int, void*, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& );
//
//        // Define RemovalFunction
//        static void FCRemovalFunction ( int , void*);

};

#if defined(WIN32) && !defined(FRAME_FRAMEFIXEDCONSTRAINT_CPP)
extern template class SOFA_FRAME_API FrameFixedConstraint<Affine3dTypes>;
extern template class SOFA_FRAME_API FrameFixedConstraint<Quadratic3dTypes>;
extern template class SOFA_FRAME_API FrameFixedConstraint<Affine3fTypes>;
extern template class SOFA_FRAME_API FrameFixedConstraint<Quadratic3fTypes>;
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
