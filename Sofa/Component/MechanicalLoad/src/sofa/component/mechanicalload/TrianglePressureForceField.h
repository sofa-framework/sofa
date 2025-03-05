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
#include <sofa/component/mechanicalload/config.h>


#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/TopologySubsetData.h>
#include <sofa/type/MatSym.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::mechanicalload
{

template<class DataTypes>
class TrianglePressureForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TrianglePressureForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
    typedef type::Mat<3,3,Real> Mat33;
    typedef type::MatSym<3,Real> MatSym3;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    using Index = sofa::Index;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Deriv> pressure;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<MatSym3> cauchyStress;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::vector<Index> > triangleList;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> p_showForces;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> p_useConstantForce;


    Data<Deriv> d_pressure; ///< Pressure force per unit area
  	Data<MatSym3> d_cauchyStress; ///< Cauchy Stress applied on the normal of each triangle

    Data<sofa::type::vector<Index> > d_triangleList; ///< Indices of triangles separated with commas where a pressure is applied

    Data<bool> d_showForces; ///< draw triangles which have a given pressure
    Data<bool> d_useConstantForce; ///< applied force is computed as the pressure vector times the area at rest

    /// Link to be set to the topology container in the component graph.
    SingleLink<TrianglePressureForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    core::objectmodel::lifecycle::DeprecatedData normal{ this, "v24.06", "v24.12", "normal", "Plan selection using normal, dmin, dmax has been removed. Triangles should be selected using an Engine.Select and passed using Data triangleList" };
    core::objectmodel::lifecycle::DeprecatedData dmin{ this, "v24.06", "v24.12", "dmin", "Plan selection using normal, dmin, dmax has been removed. Triangles should be selected using an Engine.Select and passed using Data triangleList" };
    core::objectmodel::lifecycle::DeprecatedData dmax{ this, "v24.06", "v24.12", "dmax", "Plan selection using normal, dmin, dmax has been removed. Triangles should be selected using an Engine.Select and passed using Data triangleList" };

protected:

    class TrianglePressureInformation
    {
    public:
        Real area;
        Deriv force;
		Mat33 DfDx[3];

        TrianglePressureInformation() = default;
        TrianglePressureInformation(const TrianglePressureInformation &e)
            : area(e.area),force(e.force)
        { }

        TrianglePressureInformation & operator= (const TrianglePressureInformation & other) {
            area = other.area;
            force = other.force;
            DfDx[0] = other.DfDx[0];
            DfDx[1] = other.DfDx[1];
            DfDx[2] = other.DfDx[2];
            return *this;
        }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TrianglePressureInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TrianglePressureInformation& /*ei*/ )
        {
            return in;
        }
    };

    core::topology::TriangleSubsetData<sofa::type::vector<TrianglePressureInformation> > d_trianglePressureMap; ///< Map between triangle indices and their pressure

    sofa::core::topology::BaseMeshTopology* m_topology;

	TrianglePressureForceField();

    ~TrianglePressureForceField() override;
public:
    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;

    /// Constant pressure has null variation
    void addKToMatrix(sofa::linearalgebra::BaseMatrix * /*m*/, SReal /*kFactor*/, unsigned int & /*offset*/) override {}

    /// Constant pressure has null variation
    void addKToMatrix(const core::MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/ ) override {}

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override;
    void draw(const core::visual::VisualParams* vparams) override;

    void setPressure(Deriv _pressure) { this->d_pressure = _pressure; updateTriangleInformation(); }

protected :
    void updateTriangleInformation();
    void initTriangleInformation();
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGLEPRESSUREFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_MECHANICALLOAD_API TrianglePressureForceField<sofa::defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TrianglePressureForceField_CPP)


} // namespace sofa::component::mechanicalload
