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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_H

#include <sofa/component/component.h>

#include <sofa/core/behavior/PairInteractionConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

#include <deque>

namespace sofa
{

namespace component
{

namespace constraintset
{

class BilateralConstraintResolution : public core::behavior::ConstraintResolution
{
public:
    BilateralConstraintResolution(std::deque<double>* vec=NULL) : _vec(vec) {}
    virtual void resolution(int line, double** w, double* d, double* force)
    {
        force[line] -= d[line] / w[line][line];
    }

    virtual void init(int line, double** /*w*/, double* force)
    {
        if(_vec && _vec->size()) { force[line] = _vec->front(); _vec->pop_front(); }
    }

    virtual void initForce(int line, double* force)
    {
        if(_vec && _vec->size()) { force[line] = _vec->front(); _vec->pop_front(); }
    }

    void store(int line, double* force, bool /*convergence*/)
    {
        if(_vec) _vec->push_back(force[line]);
    }

protected:
    std::deque<double>* _vec;
};

class BilateralConstraintResolution3Dof : public core::behavior::ConstraintResolution
{
public:

    BilateralConstraintResolution3Dof(std::vector<double>* vec=NULL) : _vec(vec) { nbLines=3; }
    virtual void init(int line, double** w, double *force)
    {
        sofa::defaulttype::Mat<3,3,double> temp;
        temp[0][0] = w[line][line];
        temp[0][1] = w[line][line+1];
        temp[0][2] = w[line][line+2];
        temp[1][0] = w[line+1][line];
        temp[1][1] = w[line+1][line+1];
        temp[1][2] = w[line+1][line+2];
        temp[2][0] = w[line+2][line];
        temp[2][1] = w[line+2][line+1];
        temp[2][2] = w[line+2][line+2];

        invertMatrix(invW, temp);

        if(_vec && _vec->size()>=3)
        {
            force[line  ] = (*_vec)[0];
            force[line+1] = (*_vec)[1];
            force[line+2] = (*_vec)[2];
        }
    }

    virtual void initForce(int line, double* force)
    {
        if(_vec && _vec->size()>=3)
        {
            force[line  ] =  (*_vec)[0];
            force[line+1] =  (*_vec)[1];
            force[line+2] =  (*_vec)[2];
        }
    }

    virtual void resolution(int line, double** /*w*/, double* d, double* force)
    {
        for(int i=0; i<3; i++)
        {
            //	force[line+i] = 0;
            for(int j=0; j<3; j++)
                force[line+i] -= d[line+j] * invW[i][j];
        }
    }

    void store(int line, double* force, bool /*convergence*/)
    {
        if(_vec)
        {
            _vec->clear();
            _vec->resize(3);
            (*_vec)[0] = force[line];
            (*_vec)[1] = force[line+1];
            (*_vec)[2] = force[line+2];
        }
    }

protected:
    sofa::defaulttype::Mat<3,3,double> invW;
    std::vector<double>* _vec;
};

template<class DataTypes>
class BilateralInteractionConstraint : public core::behavior::PairInteractionConstraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BilateralInteractionConstraint,DataTypes), SOFA_TEMPLATE(sofa::core::behavior::PairInteractionConstraint,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename core::behavior::PairInteractionConstraint<DataTypes> Inherit;

    typedef core::objectmodel::Data<VecCoord>		DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>		DataVecDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

protected:

    std::vector<Deriv> dfree;
    defaulttype::Quaternion q;

    std::vector<unsigned int> cid;

    Data<helper::vector<int> > m1;
    Data<helper::vector<int> > m2;

    std::vector<double> prevForces;
public:

    BilateralInteractionConstraint(MechanicalState* object1, MechanicalState* object2)
        : Inherit(object1, object2)
        , m1(initData(&m1, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, "second_point","index of the constraint on the second model"))
    {
    }

    BilateralInteractionConstraint(MechanicalState* object)
        : Inherit(object, object)
        , m1(initData(&m1, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, "second_point","index of the constraint on the second model"))
    {
    }

    BilateralInteractionConstraint()
        : m1(initData(&m1, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, "second_point","index of the constraint on the second model"))
    {
    }

    virtual ~BilateralInteractionConstraint()
    {
    }

    virtual void init();

    virtual void reinit()
    {
        init();
    }

    virtual void reset()
    {
        init();
    }

    void buildConstraintMatrix(const core::ConstraintParams* cParams /* PARAMS FIRST =core::ConstraintParams::defaultInstance()*/, DataMatrixDeriv &c1, DataMatrixDeriv &c2, unsigned int &cIndex
            , const DataVecCoord &x1, const DataVecCoord &x2);

    void getConstraintViolation(const core::ConstraintParams* cParams /* PARAMS FIRST =core::ConstraintParams::defaultInstance()*/, defaulttype::BaseVector *v, const DataVecCoord &x1, const DataVecCoord &x2
            , const DataVecDeriv &v1, const DataVecDeriv &v2);

    virtual void getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset);

    void draw();
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_CONSTRAINTSET_API BilateralInteractionConstraint< defaulttype::Vec3dTypes >;
extern template class SOFA_COMPONENT_CONSTRAINTSET_API BilateralInteractionConstraint< defaulttype::Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_CONSTRAINTSET_API BilateralInteractionConstraint< defaulttype::Vec3fTypes >;
extern template class SOFA_COMPONENT_CONSTRAINTSET_API BilateralInteractionConstraint< defaulttype::Rigid3fTypes >;
#endif
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_H
