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
#ifndef FLEXIBLE_RigidConstraint_H
#define FLEXIBLE_RigidConstraint_H

#include "../initFlexible.h"
#include "AffineTypes.h"
#include "QuadraticTypes.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.inl>

#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using helper::vector;

using namespace sofa::defaulttype;
/** Make non-rigid frames rigid.
*/
template <class DataTypes>
class RigidConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RigidConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
    typedef Data<typename DataTypes::MatrixDeriv> DataMatrixDeriv;

    Data<vector<unsigned> > f_index;   ///< Indices of the constrained frames
    Data<double> _drawSize;

    // -- Constraint interface
    virtual void init() { Inherit1::init();    }

    virtual void projectResponse(const core::MechanicalParams* /*mparams*/, DataVecDeriv& resData)
    {
        helper::WriteAccessor<DataVecDeriv> res = resData;
        const vector<unsigned> & indices = f_index.getValue();
        for(unsigned i=0; i<indices.size(); i++) res[indices[i]].setRigid();
    }

    virtual void projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData)
    {
        helper::WriteAccessor<DataVecDeriv> res = vData;
        const vector<unsigned> & indices = f_index.getValue();
        for(unsigned i=0; i<indices.size(); i++)            res[indices[i]].setRigid();
    }

    virtual void projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
    {
        helper::WriteAccessor<DataVecCoord> res = xData;
        const vector<unsigned> & indices = f_index.getValue();
        for(unsigned i=0; i<indices.size(); i++)            res[indices[i]].setRigid();
    }

    virtual void projectJacobianMatrix(const core::MechanicalParams* /* PARAMS FIRST */, DataMatrixDeriv& ) {}

    virtual void applyConstraint(defaulttype::BaseMatrix *, unsigned int /*offset*/) {}
    virtual void applyConstraint(defaulttype::BaseVector *, unsigned int /*offset*/) {}

protected:
    RigidConstraint()
        : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
        , f_index( initData(&f_index,"indices","Indices of the constrained frames") )
        , _drawSize( initData(&_drawSize,0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    {
        // default to index 0
        helper::WriteAccessor<Data<vector<unsigned> > > index( f_index);
        index.push_back(0);
    }

    virtual ~RigidConstraint()  {    }

    virtual void draw(const core::visual::VisualParams* vparams)
    {
        if (!vparams->displayFlags().getShowBehaviorModels()) return;
        if (!this->isActive()) return;
        const VecCoord& x = *this->mstate->getX();
        const vector<unsigned> & indices = f_index.getValue();
        std::vector< Vector3 > points;
        for (vector<unsigned>::const_iterator it = indices.begin(); it != indices.end(); ++it) points.push_back(DataTypes::getCPos(x[*it]));
        if( _drawSize.getValue() == 0)  vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.0,0.5,1)); // old classical drawing by points
        else  vparams->drawTool()->drawSpheres(points, (float)_drawSize.getValue(), Vec<4,float>(1.0f,0.0f,0.35f,1.0f)); // new drawing by spheres
    }


};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_RigidConstraint_CPP)
extern template class SOFA_Flexible_API RigidConstraint<Affine3dTypes>;
extern template class SOFA_Flexible_API RigidConstraint<Quadratic3dTypes>;
extern template class SOFA_Flexible_API RigidConstraint<Affine3fTypes>;
extern template class SOFA_Flexible_API RigidConstraint<Quadratic3fTypes>;
#endif

}

}

}

#endif
