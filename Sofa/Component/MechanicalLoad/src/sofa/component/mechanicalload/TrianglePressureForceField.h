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

    Data<Deriv> pressure; ///< pressure is a vector with specified direction
  	Data<MatSym3> cauchyStress; ///< the Cauchy stress applied on triangles

    Data<sofa::type::vector<Index> > triangleList; ///< Indices of triangles separated with commas where a pressure is applied

    /// the normal used to define the edge subjected to the pressure force.
    Data<Deriv> normal;

    Data<Real> dmin; ///< coordinates min of the plane for the vertex selection
    Data<Real> dmax;///< coordinates max of the plane for the vertex selection
    Data<bool> p_showForces; ///< draw triangles which have a given pressure
    Data<bool> p_useConstantForce; ///< applied force is computed as the pressure vector times the area at rest

    /// Link to be set to the topology container in the component graph.
    SingleLink<TrianglePressureForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
  
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

    core::topology::TriangleSubsetData<sofa::type::vector<TrianglePressureInformation> > trianglePressureMap; ///< map between triangle indices and their pressure

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

    void setDminAndDmax(const SReal _dmin, const SReal _dmax){dmin.setValue((Real)_dmin); dmax.setValue((Real)_dmax);}
    void setNormal(const Coord n) { normal.setValue(n);}
    void setPressure(Deriv _pressure) { this->pressure = _pressure; updateTriangleInformation(); }

protected :
    void selectTrianglesAlongPlane();
    void selectTrianglesFromString();
    void updateTriangleInformation();
    void initTriangleInformation();
    bool isPointInPlane(Coord p);
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGLEPRESSUREFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_MECHANICALLOAD_API TrianglePressureForceField<sofa::defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TrianglePressureForceField_CPP)


} // namespace sofa::component::mechanicalload
