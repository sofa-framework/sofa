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
#ifndef SOFA_COMPONENT_CONSTRAINT_STOPPERCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_STOPPERCONSTRAINT_H

#include <sofa/core/behavior/InteractionConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <iostream>
#include <sofa/core/behavior/OdeSolver.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace constraint
{

#ifdef SOFA_DEV
/*
class StopperConstraintResolution : public core::behavior::ConstraintResolution
{
public:

	virtual void resolution(int line, double** w, double* d, double* force)
	{
		force[line] += 0.0; //d[line] / w[line][line];
	}
};
*/

class StopperConstraintResolution1Dof : public core::behavior::ConstraintResolution
{
protected:
    double _invW, _w, _min, _max ;

public:

    StopperConstraintResolution1Dof(const double &min, const double &max) { nbLines=1; _min=min; _max=max; }

    virtual void init(int line, double** w, double *force)
    {
        _w = w[line][line];
        _invW = 1.0/_w;
        force[line  ] = 0.0;
    }



    virtual void resolution(int line, double** /*w*/, double* d, double* force)
    {
        double dfree = d[line] - _w * force[line];

        if (dfree > _max)
            force[line] = (_max - dfree) * _invW;
        else if (dfree < _min)
            force[line] = (_min - dfree) * _invW;
        else
            force[line] = 0;




    }


};
#endif
template<class DataTypes>
class StopperConstraint : public core::behavior::BaseConstraint
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(StopperConstraint,DataTypes), core::behavior::BaseConstraint);

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecConst VecConst;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename defaulttype::SparseConstraint<Deriv> SparseConstraint;
    typedef typename SparseConstraint::const_data_iterator ConstraintIterator;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;

protected:
    MechanicalState* object;
    bool yetIntegrated;

    Coord dfree;
    unsigned int cid;

    Data<int> index;
    Data<std::string> pathObject;
    Data<double> min;
    Data<double> max;



    sofa::core::behavior::OdeSolver* ode_integrator;

public:

    StopperConstraint(MechanicalState* object)
        : object(object), yetIntegrated(false)
        , index(initData(&index, 0, "index","index of the stop constraint"))
        , pathObject(initData(&pathObject,  "object","path of object in interaction"))
        , min(initData(&min,-100.0, "min", "minimum value accepted"))
        , max(initData(&max, 100.0, "max", "maximum value accepted"))
    {
    }


    StopperConstraint()
        : object(object), yetIntegrated(false)
        , index(initData(&index, 0, "index","index of the stop constraint"))
        , pathObject(initData(&pathObject,  "object","path of object in interaction"))
        , min(initData(&min,-100.0, "min", "minimum value accepted"))
        , max(initData(&max, 100.0, "max", "maximum value accepted"))
    {

    }

    virtual ~StopperConstraint()
    {
    }

    MechanicalState* getObject() { return object; }
    core::behavior::BaseMechanicalState* getMechModel() { return object; }

    virtual void init();

    virtual void applyConstraint(unsigned int & /*constraintId*/);

    virtual void getConstraintValue(defaulttype::BaseVector *, bool /* freeMotion */ = true );

    virtual void getConstraintId(long* id, unsigned int &offset);

    int getIndex()
    {
        return index.getValue();
    }

    double getMin()
    {
        return min.getValue();
    }

    double getMax()
    {
        return max.getValue();
    }

#ifdef SOFA_DEV
    virtual void getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset);
#endif
    // Previous Constraint Interface
    virtual void projectResponse() {}
    virtual void projectJacobianMatrix() {}
    virtual void projectVelocity() {}
    virtual void projectPosition() {}
    virtual void projectFreeVelocity() {}
    virtual void projectFreePosition() {}

    /// Pre-construction check method called by ObjectFactory.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("object"))
        {
            if (dynamic_cast<MechanicalState*>(arg->findObject(arg->getAttribute("object",".."))) == NULL)
                return false;
        }
        else
        {
            if (dynamic_cast<MechanicalState*>(context->getMechanicalState()) == NULL)
                return false;
        }
        return core::behavior::BaseConstraint::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::behavior::BaseConstraint::create(obj, context, arg);
        if (arg && (arg->getAttribute("object") ))
        {
            obj->object = dynamic_cast<MechanicalState*>(arg->findObject(arg->getAttribute("object","..")));

        }
        else if (context)
        {
            obj->object =dynamic_cast<MechanicalState*>(context->getMechanicalState());


        }
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const StopperConstraint<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
    void draw();

    /// this constraint is holonomic
    bool isHolonomic() {return true;}
};
} // namespace constraint

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_H
