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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_H

#include <sofa/SofaGeneral.h>

#include <sofa/core/behavior/PairInteractionConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

#include <deque>

namespace sofa
{

namespace component
{

namespace constraintset
{

inline double sign(double &toto)
{
    if (toto<0.0)
        return -1.0;
    return 1.0;
}
inline float sign(float &toto)
{
    if (toto<0.0f)
        return -1.0f;
    return 1.0f;
}

class BilateralConstraintResolution : public core::behavior::ConstraintResolution
{
public:
    BilateralConstraintResolution(double* initF=NULL) : _f(initF) {}
    virtual void resolution(int line, double** w, double* d, double* force)
    {
        force[line] -= d[line] / w[line][line];
    }

    virtual void init(int line, double** /*w*/, double* force)
    {
        if(_f) { force[line] = *_f; }
    }

    virtual void initForce(int line, double* force)
    {
        if(_f) { force[line] = *_f; }
    }

    void store(int line, double* force, bool /*convergence*/)
    {
        if(_f) *_f = force[line];
    }

protected:
    double* _f;
};

class BilateralConstraintResolution3Dof : public core::behavior::ConstraintResolution
{
public:

    BilateralConstraintResolution3Dof(sofa::defaulttype::Vec3d* vec=NULL) : _f(vec) { nbLines=3; }
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

        if(_f)
        {
            for(int i=0; i<3; i++)
                force[line+i] = (*_f)[i];
        }
    }

    virtual void initForce(int line, double* force)
    {
        if(_f)
        {
            for(int i=0; i<3; i++)
                force[line+i] = (*_f)[i];
        }
    }

    virtual void resolution(int line, double** /*w*/, double* d, double* force)
    {
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<3; j++)
                force[line+i] -= d[line+j] * invW[i][j];
        }
    }

    void store(int line, double* force, bool /*convergence*/)
    {
        if(_f)
        {
            for(int i=0; i<3; i++)
                (*_f)[i] = force[line+i];
        }
    }

protected:
    sofa::defaulttype::Mat<3,3,double> invW;
    sofa::defaulttype::Vec3d* _f;
};

template <int N>
class BilateralConstraintResolutionNDof : public core::behavior::ConstraintResolution
{
public:
    BilateralConstraintResolutionNDof(sofa::defaulttype::Vec<N, double>* vec=NULL) : _f(vec) { nbLines=N; }
    virtual void init(int line, double** w, double *force)
    {
        sofa::defaulttype::Mat<N,N,double> temp;
        for(int i=0; i<N; i++)
            for(int j=0; j<N; j++)
                temp[i][j] = w[line+i][line+j];

        invertMatrix(invW, temp);

        if(_f)
        {
            for(int i=0; i<N; i++)
                force[line+i] = (*_f)[i];
        }
    }

    virtual void initForce(int line, double* force)
    {
        if(_f)
        {
            for(int i=0; i<N; i++)
                force[line+i] = (*_f)[i];
        }
    }

    virtual void resolution(int line, double** /*w*/, double* d, double* force)
    {
        for(int i=0; i<N; i++)
        {
            for(int j=0; j<N; j++)
                force[line+i] -= d[line+j] * invW[i][j];
        }
    }

    void store(int line, double* force, bool /*convergence*/)
    {
        if(_f)
        {
            for(int i=0; i<N; i++)
                (*_f)[i] = force[line+i];
        }
    }

protected:
    sofa::defaulttype::Mat<N,N,double> invW;
    sofa::defaulttype::Vec<N, double>* _f;
};

template<class DataTypes>
class BilateralInteractionConstraint : public core::behavior::PairInteractionConstraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BilateralInteractionConstraint,DataTypes), SOFA_TEMPLATE(sofa::core::behavior::PairInteractionConstraint,DataTypes));

    typedef typename core::behavior::PairInteractionConstraint<DataTypes> Inherit;

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef core::behavior::BaseConstraint::PersistentID PersistentID;

    typedef core::objectmodel::Data<VecCoord>		DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>		DataVecDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

protected:

    std::vector<Deriv> dfree;
    defaulttype::Quaternion q;

    std::vector<unsigned int> cid;

    Data<helper::vector<int> > m1;
    Data<helper::vector<int> > m2;
    Data<VecDeriv> restVector;
    Data<int> activateAtIteration;
    Data<bool> merge;
    Data<bool> derivative;
    std::vector<defaulttype::Vec3d> prevForces;


    // grouped square constraints
    bool squareXYZ[3];
    Deriv dfree_square_total;


    bool activated;
    int iteration;


    BilateralInteractionConstraint(MechanicalState* object1, MechanicalState* object2)
        : Inherit(object1, object2)
        , m1(initData(&m1, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, "second_point","index of the constraint on the second model"))
        , restVector(initData(&restVector, "rest_vector","Relative position to maintain between attached points (optional)"))
        , activateAtIteration( initData(&activateAtIteration, 0, "activateAtIteration", "activate constraint at specified interation (0=disable)"))
        , merge(initData(&merge,false, "merge", "TEST: merge the bilateral constraints in a unique constraint"))
        , derivative(initData(&derivative,false, "derivative", "TEST: derivative"))
        , activated(true), iteration(0)
    {
        this->f_listening.setValue(true);
    }

    BilateralInteractionConstraint(MechanicalState* object)
        : Inherit(object, object)
        , m1(initData(&m1, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, "second_point","index of the constraint on the second model"))
        , restVector(initData(&restVector, "rest_vector","Relative position to maintain between attached points (optional)"))
        , activateAtIteration( initData(&activateAtIteration, 0, "activateAtIteration", "activate constraint at specified interation (0 = always enabled, -1=disabled)"))
        , merge(initData(&merge,false, "merge", "TEST: merge the bilateral constraints in a unique constraint"))
        , derivative(initData(&derivative,false, "derivative", "TEST: derivative"))
        , activated(true), iteration(0)
    {
        this->f_listening.setValue(true);
    }

    BilateralInteractionConstraint()
        : m1(initData(&m1, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, "second_point","index of the constraint on the second model"))
        , restVector(initData(&restVector, "rest_vector","Relative position to maintain between attached points (optional)"))
        , activateAtIteration( initData(&activateAtIteration, 0, "activateAtIteration", "activate constraint at specified interation (0 = always enabled, -1=disabled)"))
        , merge(initData(&merge,false, "merge", "TEST: merge the bilateral constraints in a unique constraint"))
        , derivative(initData(&derivative,false, "derivative", "TEST: derivative"))
        , activated(true), iteration(0)
    {
        this->f_listening.setValue(true);
    }

    virtual ~BilateralInteractionConstraint()
    {
    }
public:
    virtual void init();

    virtual void reinit();

    virtual void reset()
    {
        init();
    }

    void buildConstraintMatrix(const core::ConstraintParams* cParams /* PARAMS FIRST =core::ConstraintParams::defaultInstance()*/, DataMatrixDeriv &c1, DataMatrixDeriv &c2, unsigned int &cIndex
            , const DataVecCoord &x1, const DataVecCoord &x2);

    void getConstraintViolation(const core::ConstraintParams* cParams /* PARAMS FIRST =core::ConstraintParams::defaultInstance()*/, defaulttype::BaseVector *v, const DataVecCoord &x1, const DataVecCoord &x2
            , const DataVecDeriv &v1, const DataVecDeriv &v2);


    void getVelocityViolation(defaulttype::BaseVector *v, const DataVecCoord &x1, const DataVecCoord &x2, const DataVecDeriv &v1, const DataVecDeriv &v2);

    virtual void getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset);

    void handleEvent(sofa::core::objectmodel::Event *event);

    void draw(const core::visual::VisualParams* vparams);

    // Contact handling methods
public:
    void clear(int reserve = 0)
    {
        helper::WriteAccessor<Data<helper::vector<int> > > wm1 = this->m1;
        helper::WriteAccessor<Data<helper::vector<int> > > wm2 = this->m2;
        helper::WriteAccessor<Data<VecDeriv > > wrest = this->restVector;
        wm1.clear();
        wm2.clear();
        wrest.clear();
        if (reserve)
        {
            wm1.reserve(reserve);
            wm2.reserve(reserve);
            wrest.reserve(reserve);
        }
    }

    virtual void addContact(Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, Coord Pfree, Coord Qfree, long id=0, PersistentID localid=0);

    void addContact(Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, long id=0, PersistentID localid=0)
    {
        addContact(norm, P, Q, contactDistance, m1, m2,
                this->getMState2()->read(core::ConstVecCoordId::freePosition())->getValue()[m2],
                this->getMState1()->read(core::ConstVecCoordId::freePosition())->getValue()[m1],
                id, localid);
    }

    void addContact(Deriv norm, Real contactDistance, int m1, int m2, long id=0, PersistentID localid=0)
    {
        addContact(norm,
                this->getMState2()->read(core::ConstVecCoordId::position())->getValue()[m2],
                this->getMState1()->read(core::ConstVecCoordId::position())->getValue()[m1],
                contactDistance, m1, m2,
                this->getMState2()->read(core::ConstVecCoordId::freePosition())->getValue()[m2],
                this->getMState1()->read(core::ConstVecCoordId::freePosition())->getValue()[m1],
                id, localid);
    }


};

#ifndef SOFA_FLOAT
template<>
void BilateralInteractionConstraint<defaulttype::Rigid3dTypes>::buildConstraintMatrix(const core::ConstraintParams *cParams /* PARAMS FIRST */, DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &cIndex
        , const DataVecCoord &x1, const DataVecCoord &x2);

template<>
void BilateralInteractionConstraint<defaulttype::Rigid3dTypes>::getConstraintViolation(const core::ConstraintParams *cParams /* PARAMS FIRST */, defaulttype::BaseVector *v, const DataVecCoord &x1_d, const DataVecCoord &x2_d
        , const DataVecDeriv &v1_d, const DataVecDeriv &v2_d);

template<>
void BilateralInteractionConstraint<defaulttype::Rigid3dTypes>::addContact(Deriv /*norm*/, Coord P, Coord Q, Real /*contactDistance*/, int m1, int m2, Coord /*Pfree*/, Coord /*Qfree*/, long /*id*/, PersistentID /*localid*/);

#endif

#ifndef SOFA_DOUBLE
template<>
void BilateralInteractionConstraint<defaulttype::Rigid3fTypes>::buildConstraintMatrix(const core::ConstraintParams *cParams /* PARAMS FIRST */, DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &cIndex
        , const DataVecCoord &x1_d, const DataVecCoord &x2_d);

template<>
void BilateralInteractionConstraint<defaulttype::Rigid3fTypes>::getConstraintViolation(const core::ConstraintParams *cParams /* PARAMS FIRST */, defaulttype::BaseVector *v, const DataVecCoord &x1_d, const DataVecCoord &x2_d
        , const DataVecDeriv &v1_d, const DataVecDeriv &v2_d);

template<>
void BilateralInteractionConstraint<defaulttype::Rigid3fTypes>::addContact(Deriv /*norm*/, Coord P, Coord Q, Real /*contactDistance*/, int m1, int m2, Coord /*Pfree*/, Coord /*Qfree*/, long /*id*/, PersistentID /*localid*/);
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_CONSTRAINT)
#ifndef SOFA_FLOAT
extern template class SOFA_CONSTRAINT_API BilateralInteractionConstraint< defaulttype::Vec3dTypes >;
extern template class SOFA_CONSTRAINT_API BilateralInteractionConstraint< defaulttype::Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_CONSTRAINT_API BilateralInteractionConstraint< defaulttype::Vec3fTypes >;
extern template class SOFA_CONSTRAINT_API BilateralInteractionConstraint< defaulttype::Rigid3fTypes >;
#endif
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_H
