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

#include <sofa/component/solidmechanics/spring/SpringForceField.h>
#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologySubsetIndices.h>

namespace sofa::component::solidmechanics::spring
{

/** SpringForceField able to evaluate and apply its stiffness.
This allows to perform implicit integration.
Stiffness is evaluated and stored by the addForce method.
When explicit integration is used, SpringForceField is slightly more efficient.
*/

template<class DataTypes>
class StiffSpringForceField : public SpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(StiffSpringForceField,DataTypes), SOFA_TEMPLATE(SpringForceField,DataTypes));

    typedef SpringForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;
    typedef type::vector<sofa::Index> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;


    typedef typename Inherit::Spring Spring;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
    static constexpr auto N = DataTypes::spatial_dimensions;
    typedef type::Mat<N,N,Real> Mat;

    SetIndex d_indices1; ///< Indices of the source points on the first model
    SetIndex d_indices2; ///< Indices of the fixed points on the second model

    core::objectmodel::Data<sofa::type::vector<SReal> > d_lengths; ///< List of lengths to create the springs. Must have the same than indices1 & indices2, or if only one element, it will be applied to all springs. If empty, 0 will be applied everywhere
protected:
    sofa::type::vector<Mat>  dfdx;

    /// Accumulate the spring force and compute and store its stiffness
    void addSpringForce(Real& potentialEnergy, VecDeriv& f1,const  VecCoord& p1,const VecDeriv& v1, VecDeriv& f2,const  VecCoord& p2,const  VecDeriv& v2, sofa::Index i, const Spring& spring) override;

    using SpringForce = typename Inherit::SpringForce;
    struct StiffSpringForce : Inherit::SpringForce
    {
        type::MatNoInit<N, N, Real> dForce_dX;
    };
    std::unique_ptr<SpringForce> computeSpringForce(const VecCoord& p1, const VecDeriv& v1, const VecCoord& p2, const VecDeriv& v2, const Spring& spring) override;

    /// Apply the stiffness, i.e. accumulate df given dx
    virtual void addSpringDForce(VecDeriv& df1,const  VecDeriv& dx1, VecDeriv& df2,const  VecDeriv& dx2, sofa::Index i, const Spring& spring, SReal kFactor, SReal bFactor);

    virtual typename DataTypes::DPos computeSpringDForce(VecDeriv& df1,const  VecDeriv& dx1, VecDeriv& df2,const  VecDeriv& dx2, sofa::Index i, const Spring& spring, SReal kFactor, SReal bFactor);

    StiffSpringForceField(SReal ks=100.0, SReal kd=5.0);
    StiffSpringForceField(MechanicalState* object1, MechanicalState* object2, SReal ks=100.0, SReal kd=5.0);

    /// Will create the set of springs using \sa d_indices1 and \sa d_indices2 with \sa d_length
    void createSpringsFromInputs();

public:
    void init() override;

    /// Accumulate f corresponding to x,v
    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 ) override;
    /// Accumulate df corresponding to dx
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2) override;
    using Inherit::addKToMatrix;
    void addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;


protected:

    template<class Matrix>
    static void addToMatrix(Matrix* globalMatrix, const unsigned int offsetRow, const unsigned int offsetCol, const Mat& localMatrix);
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_STIFFSPRINGFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API StiffSpringForceField<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API StiffSpringForceField<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API StiffSpringForceField<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API StiffSpringForceField<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API StiffSpringForceField<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::solidmechanics::spring
