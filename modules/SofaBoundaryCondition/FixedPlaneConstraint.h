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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_H
#include "config.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <set>
#include <SofaBaseTopology/TopologySubsetData.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/// This class can be overriden if needed for additionnal storage within template specilizations.
template <class DataTypes>
class FixedPlaneConstraintInternalData
{
};

template <class DataTypes>
class FixedPlaneConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedPlaneConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;
public:
    /// direction on which the constraint applies
    Data<Coord> direction;

    Data<Real> dmin; ///< coordinates min of the plane for the vertex selection
    Data<Real> dmax;///< coordinates max of the plane for the vertex selection
protected:
    FixedPlaneConstraintInternalData<DataTypes> data;
    friend class FixedPlaneConstraintInternalData<DataTypes>;

    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams, DataDeriv& dx);

    SetIndex indices; ///< the set of vertex indices

    /// whether vertices should be selected from 2 parallel planes
    bool selectVerticesFromPlanes;


    FixedPlaneConstraint();

    ~FixedPlaneConstraint();
public:
    void addConstraint(int index);

    void removeConstraint(int index);

    // -- Constraint interface

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    // Implement projectMatrix for assembled solver of compliant
    virtual void projectMatrix( sofa::defaulttype::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

	// Implement applyConstraint for direct solvers
    virtual void applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    virtual void applyConstraint(const core::MechanicalParams* mparams, defaulttype::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;


    virtual void init() override;

    void setDirection (Coord dir);
    void selectVerticesAlongPlane();
    void setDminAndDmax(const Real _dmin,const Real _dmax) {dmin=_dmin; dmax=_dmax; selectVerticesFromPlanes=true;}

    void draw(const core::visual::VisualParams* vparams) override;

    class FCPointHandler : public sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >
    {
    public:
        typedef typename FixedPlaneConstraint<DataTypes>::SetIndexArray SetIndexArray;

        FCPointHandler(FixedPlaneConstraint<DataTypes>* _fc, sofa::component::topology::PointSubsetData<SetIndexArray>* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >(_data), fc(_fc) {}



        void applyDestroyFunction(unsigned int /*index*/, value_type& /*T*/);


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        FixedPlaneConstraint<DataTypes> *fc;
    };

protected:
    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;

    /// Handler for subset Data
    FCPointHandler* pointHandler;

    bool isPointInPlane(Coord p)
    {
        Real d=dot(p,direction.getValue());
        if ((d>dmin.getValue())&& (d<dmax.getValue()))
            return true;
        else
            return false;
    }
};

#ifndef SOFA_FLOAT
template <> template <class DataDeriv>
void FixedPlaneConstraint<defaulttype::Rigid3dTypes>::projectResponseT(const core::MechanicalParams* mparams, DataDeriv& /*res*/);

template <>
bool FixedPlaneConstraint<defaulttype::Rigid3dTypes>::isPointInPlane(Coord /*p*/);
#endif

#ifndef SOFA_DOUBLE
template <> template <class DataDeriv>
void FixedPlaneConstraint<defaulttype::Rigid3fTypes>::projectResponseT(const core::MechanicalParams* mparams, DataDeriv& /*res*/);

template <>
bool FixedPlaneConstraint<defaulttype::Rigid3fTypes>::isPointInPlane(Coord /*p*/);
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Rigid3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Rigid3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec6fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
