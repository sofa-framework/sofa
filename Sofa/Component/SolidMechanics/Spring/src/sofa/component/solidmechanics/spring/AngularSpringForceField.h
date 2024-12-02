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
#include <sofa/component/solidmechanics/spring/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/type/vector.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>

#include <sofa/core/objectmodel/RenamedData.h>


namespace sofa::core::behavior
{

template< class T > class MechanicalState;

} // namespace sofa::core::behavior

namespace sofa::component::solidmechanics::spring
{

/**
* @brief This class describes a simple elastic springs ForceField between DOFs positions and rest positions.
*
* Springs are applied to given degrees of freedom between their current positions and their rest shape positions.
* An external MechanicalState reference can also be passed to the ForceField as rest shape position.
*/
template<class DataTypes>
class AngularSpringForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(AngularSpringForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef type::vector< sofa::Index > VecIndex;
    typedef type::vector< Real >	 VecReal;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    sofa::core::objectmodel::RenamedData< type::vector< sofa::Index > > indices;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    sofa::core::objectmodel::RenamedData<VecReal> angularStiffness;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    sofa::core::objectmodel::RenamedData<VecReal> angularLimit;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    sofa::core::objectmodel::RenamedData<bool> drawSpring;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    sofa::core::objectmodel::RenamedData<type::RGBAColor> springColor;

    Data< type::vector< sofa::Index > > d_indices; ///< index of nodes controlled by the angular springs
    Data< VecReal > d_angularStiffness; ///< angular stiffness for the controlled nodes
    Data<VecReal> d_angularLimit; ///< angular limit (max; min) values where the force applies
    Data< bool > d_drawSpring; ///< draw Spring
    Data< type::RGBAColor > d_springColor; ///< spring color

    linearalgebra::EigenBaseSparseMatrix<typename DataTypes::Real> matS;

protected:
    AngularSpringForceField();
public:
    /// BaseObject initialization method.
    void bwdInit() override;
    void reinit() override;

    /// Add the forces.
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    /// Brings ForceField contribution to the global system stiffness matrix.
    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix ) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;
    void draw(const core::visual::VisualParams* vparams) override;

protected :

    core::behavior::MechanicalState<DataTypes> *mState;
    VecReal k;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_AngularSpringForceField_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API AngularSpringForceField<sofa::defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::solidmechanics::spring
