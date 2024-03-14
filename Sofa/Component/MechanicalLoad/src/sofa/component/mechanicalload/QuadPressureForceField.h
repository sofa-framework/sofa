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

namespace sofa::component::mechanicalload
{

/**
 * @brief QuadPressureForceField Class
 *
 * Implements a pressure force applied on a quad surface.
 * The force applied to each vertex of a quad is equal to the quad surface*Pressure/4.
 * The force is constant during animation. 
 */
template<class DataTypes>
class QuadPressureForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(QuadPressureForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    using Index = sofa::Index;

    Data<Deriv> pressure; ///< Pressure force per unit area

    Data<sofa::type::vector<Index> > quadList; ///< Indices of quads separated with commas where a pressure is applied

    /// the normal used to define the edge subjected to the pressure force.
    Data<Deriv> normal;

    Data<Real> dmin; ///< coordinates min of the plane for the vertex selection
    Data<Real> dmax;///< coordinates max of the plane for the vertex selection
    Data<bool> p_showForces; ///< draw quads which have a given pressure

    /// Link to be set to the topology container in the component graph.
    SingleLink<QuadPressureForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:

    class QuadPressureInformation
    {
    public:
        Real area{};
        Deriv force{};

        QuadPressureInformation() {}
        QuadPressureInformation(const QuadPressureInformation &e)
            : area(e.area),force(e.force)
        { }

        QuadPressureInformation & operator= (const QuadPressureInformation & other) {
            area = other.area;
            force = other.force;
            return *this;
        }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const QuadPressureInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, QuadPressureInformation& /*ei*/ )
        {
            return in;
        }
    };

    sofa::core::topology::QuadSubsetData<sofa::type::vector<QuadPressureInformation> > quadPressureMap; ///< map between quad indices and their pressure

    /// Pointer to the current topology                                                                        /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* m_topology;

    QuadPressureForceField();

    virtual ~QuadPressureForceField();
public:
    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;

    /// Constant pressure has null variation
    void addKToMatrix(sofa::linearalgebra::BaseMatrix * /*m*/, SReal /*kFactor*/, unsigned int & /*offset*/) override {}

    /// Constant pressure has null variation
    void addKToMatrix(const core::MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/ ) override {}

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* /*matrix*/) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override { msg_warning() << "Method getPotentialEnergy not implemented yet."; return 0.0; }

    void draw(const core::visual::VisualParams* vparams) override;

    void setDminAndDmax(const SReal _dmin, const SReal _dmax) {dmin.setValue((Real)_dmin); dmax.setValue((Real)_dmax);}
    void setNormal(const Coord n) { normal.setValue(n);}
    void setPressure(Deriv _pressure) { this->pressure = _pressure; updateQuadInformation(); }

protected :
    void selectQuadsAlongPlane();
    void selectQuadsFromString();
    void updateQuadInformation();
    void initQuadInformation();
    bool isPointInPlane(Coord p);
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_QUADPRESSUREFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_MECHANICALLOAD_API QuadPressureForceField<defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_QUADPRESSUREFORCEFIELD_CPP)

} // namespace sofa::component::mechanicalload
