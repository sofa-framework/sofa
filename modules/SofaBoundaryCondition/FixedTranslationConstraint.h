/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDTRANSLATIONCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDTRANSLATIONCONSTRAINT_H
#include "config.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <SofaBaseTopology/TopologySubsetData.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class FixedTranslationConstraintInternalData
{
};

/** Attach given particles to their initial positions.
*/
template <class DataTypes>
class FixedTranslationConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedTranslationConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;
protected:
    FixedTranslationConstraintInternalData<DataTypes> data;
    friend class FixedTranslationConstraintInternalData<DataTypes>;

public:
    SetIndex f_indices;
    Data<bool> f_fixAll;
    Data<SReal> _drawSize;
    SetIndex f_coordinates;
protected:
    FixedTranslationConstraint();

    virtual ~FixedTranslationConstraint();
public:
    // methods to add/remove some indices
    void clearIndices();
    void addIndex(unsigned int index);
    void removeIndex(unsigned int index);

    // -- Constraint interface
    void init() override;

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;


    virtual void draw(const core::visual::VisualParams* vparams) override;

    class FCPointHandler : public sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >
    {
    public:
        typedef typename FixedTranslationConstraint<DataTypes>::SetIndexArray SetIndexArray;
        typedef sofa::core::topology::Point Point;
        FCPointHandler(FixedTranslationConstraint<DataTypes>* _fc, sofa::component::topology::PointSubsetData<SetIndexArray>* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >(_data), fc(_fc) {}



        void applyDestroyFunction(unsigned int /*index*/, value_type& /*T*/);


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        FixedTranslationConstraint<DataTypes> *fc;
    };

protected:
    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams, DataDeriv& dx);

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;

    /// Handler for subset Data
    FCPointHandler* pointHandler;

};

#ifndef SOFA_FLOAT
template<>
void FixedTranslationConstraint<defaulttype::Vec6dTypes>::draw(const core::visual::VisualParams* vparams);
#endif

#ifndef SOFA_DOUBLE
template<>
void FixedTranslationConstraint<defaulttype::Vec6fTypes>::draw(const core::visual::VisualParams* vparams);
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDTRANSLATIONCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API FixedTranslationConstraint<defaulttype::Rigid3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedTranslationConstraint<defaulttype::Rigid2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedTranslationConstraint<defaulttype::Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API FixedTranslationConstraint<defaulttype::Rigid3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedTranslationConstraint<defaulttype::Rigid2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedTranslationConstraint<defaulttype::Vec6fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif
