/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/core/config.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/SingleStateAccessor.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::core::behavior
{

/**
 *  \brief Component responsible for mass-related computations (gravity, acceleration).
 *
 *  Mass can be defined either as a scalar, vector, or a full mass-matrix.
 *  It is responsible for converting forces to accelerations (for explicit integrators),
 *  or displacements to forces (for implicit integrators).
 *
 *  It is also a ForceField, computing gravity-related forces.
 */
template<class DataTypes>
class Mass : public BaseMass, public SingleStateAccessor<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(Mass,DataTypes), BaseMass);

    typedef typename DataTypes::VecCoord    VecCoord;
    typedef typename DataTypes::VecDeriv    VecDeriv;
    typedef typename DataTypes::Real        Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef typename DataTypes::Coord       Coord;
    typedef typename DataTypes::Deriv       Deriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;

protected:
    Mass(MechanicalState<DataTypes> *mm = nullptr);

    ~Mass() override;
public:

    /// @name Vector operations
    /// @{
    ///                         $ f += factor M dx $
    ///
    /// This method retrieves the force and dx vector and call the internal
    /// addMDx(const MechanicalParams*, DataVecDeriv&, const DataVecDeriv&, SReal) method implemented by the component.
    void addMDx(const MechanicalParams* mparams, MultiVecDerivId fid, SReal factor) override;

    virtual void addMDx(const MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

    ///                            $ dx = M^-1 f $
    ///
    /// This method retrieves the force and dx vector and call the internal
    /// accFromF(VecDeriv&,const VecDeriv&) method implemented by the component.
    void accFromF(const MechanicalParams* mparams, MultiVecDerivId aid) override;

    virtual void accFromF(const MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f);

    ///                         $ f = M g $
    ///
    /// This method computes the external force due to the gravitational acceleration
    /// addGravitationalForce(const MechanicalParams*, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) method implemented by the component.
    virtual void addGravitationalForce( const MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v, const Deriv& gravity) = 0;

    ///                         $ e = M g x $
    ///
    /// This method retrieves the positions vector and call the internal
    /// getGravitationalPotentialEnergy(const MechanicalParams*, const VecCoord&) method implemented by the component.
    virtual SReal getGravitationalPotentialEnergy( const MechanicalParams* mparams, const DataVecCoord& x, const Deriv& gravity) const = 0;

    ///                         $ e = 1/2  v^t M v $
    ///
    /// This method retrieves the velocity vector and call the internal
    /// getKineticEnergy(const MechanicalParams*, const DataVecDeriv&) method implemented by the component.
    SReal getKineticEnergy( const MechanicalParams* mparams) const override;
    virtual SReal getKineticEnergy( const MechanicalParams* mparams, const DataVecDeriv& v) const;

    ///    $ m = ( Mv, cross(x,Mv)+Iw ) $
    /// linearMomentum = Mv, angularMomentum_particle = cross(x,linearMomentum), angularMomentum_body = cross(x,linearMomentum)+Iw
    ///
    /// This method retrieves the positions and velocity vectors and call the internal
    /// getMomentum(const MechanicalParams*, const VecCoord&, const VecDeriv&) method implemented by the component.
    type::Vector6 getMomentum( const MechanicalParams* mparams ) const override;
    virtual type::Vector6 getMomentum( const MechanicalParams* , const DataVecCoord& , const DataVecDeriv&  ) const;



    /// @}

    /// @name Matrix operations
    /// @{

    void addMToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    virtual void addMToMatrix(sofa::linearalgebra::BaseMatrix * matrix, SReal mFact, unsigned int &offset);

    /// @}

    /// initialization to export kinetic and potential energy to gnuplot files format
    void initGnuplot(const std::string path) override;

    /// export kinetic and potential energy state at "time" to a gnuplot file
    void exportGnuplot(const MechanicalParams* mparams, SReal time) override;

    /// recover the mass of an element
    SReal getElementMass(sofa::Index) const override;
    void getElementMass(sofa::Index index, linearalgebra::BaseMatrix *m) const override;

protected:
    /// stream to export Kinematic, Potential and Mechanical Energy to gnuplot files
    std::ofstream* m_gnuplotFileEnergy;

public:
    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        const std::string attributeName {"mstate"};
        std::string mstateLink = arg->getAttribute(attributeName,"");
        if (mstateLink.empty())
        {
            if (dynamic_cast<MechanicalState<DataTypes>*>(context->getMechanicalState()) == nullptr)
            {
                arg->logError("Since the attribute '" + attributeName + "' has not been specified, a mechanical state "
                    "with the datatype '" + DataTypes::Name() + "' has been searched in the current context, but not found.");
                return false;
            }
        }
        else
        {
            MechanicalState<DataTypes>* mstate = nullptr;
            context->findLinkDest(mstate, mstateLink, nullptr);
            if (!mstate)
            {
                arg->logError("Data attribute '" + attributeName + "' does not point to a valid mechanical state of datatype '" + std::string(DataTypes::Name()) + "'.");
                return false;
            }
        }
        return BaseObject::canCreate(obj, context, arg);
    }

    /// Automatic creation of a local GravityForceField if the worldGravity in the root node is defined
    template<class T>
    static typename T::SPtr create(T*, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>();
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);

        const sofa::core::objectmodel::BaseContext::Vec3& gravity = context->getRootContext()->getGravity();
        SReal gravityNorm = gravity.norm();
        if(gravityNorm!=0.0)
        {
            // SOFA_ATTRIBUTE_DISABLED("v22.12 (PR#2988)", "v23.12", "Transition removing gravity and introducing GravityForceField")
            // to remove after v23.06 ...
            bool savePLog = context->f_printLog.getValue();
            context->f_printLog.setValue(true);
            msg_info(context) << "A gravity seem to apply using the worldGravity in the root node (PR#2988)." << msgendl
                               << "A GravityForceField is automatically added in the node \"" << context->getName() << "\".";
            context->f_printLog.setValue(savePLog);
            // until here


            const std::string templated = context->getMechanicalState()->getTemplateName();
            std::string gravity_string = "@"+ context->getRootContext()->getPathName()+".worldGravity";

            sofa::core::objectmodel::BaseObjectDescription desc("GravityForceField","GravityForceField");
            desc.setAttribute("template", templated);
            desc.setAttribute("worldGravity", gravity_string);

            /// Create the object
            BaseObject::SPtr objGravityFF = sofa::core::ObjectFactory::getInstance()->createObject(context, &desc);
            if (objGravityFF==nullptr)
            {
                std::stringstream msg;
                msg << "Component '" << desc.getName() << "' of type '" << desc.getAttribute("type","") << "' failed:" << msgendl ;
                for (std::vector< std::string >::const_iterator it = desc.getErrors().begin(); it != desc.getErrors().end(); ++it)
                    msg << " " << *it << msgendl ;
                msg_error(context) << msg.str() ;
                return nullptr;
            }
        }

        return obj;
    }
};


#if  !defined(SOFA_CORE_BEHAVIOR_MASS_CPP)
extern template class SOFA_CORE_API Mass<defaulttype::Vec3Types>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec2Types>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec1Types>;
extern template class SOFA_CORE_API Mass<defaulttype::Vec6Types>;
extern template class SOFA_CORE_API Mass<defaulttype::Rigid3Types>;
extern template class SOFA_CORE_API Mass<defaulttype::Rigid2Types>;


#endif

} // namespace sofa::core::behavior
