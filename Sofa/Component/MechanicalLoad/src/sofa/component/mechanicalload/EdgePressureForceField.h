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

template<class DataTypes>
class EdgePressureForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(EdgePressureForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::Real        Real        ;
    typedef typename DataTypes::Coord       Coord       ;
    typedef typename DataTypes::Deriv       Deriv       ;
    typedef typename DataTypes::VecCoord    VecCoord    ;
    typedef typename DataTypes::VecDeriv    VecDeriv    ;
    typedef typename DataTypes::VecReal     VecReal     ;
    typedef Data<VecCoord>                  DataVecCoord;
    typedef Data<VecDeriv>                  DataVecDeriv;
    typedef sofa::type::Vec3d        Vec3d;

    using Index = sofa::Index;
protected:

    class EdgePressureInformation
    {
    public:
        Real length;
        Deriv force;

        EdgePressureInformation(): length(0) {}
        EdgePressureInformation(const EdgePressureInformation &e)
            : length(e.length),force(e.force)
        { }
        constexpr EdgePressureInformation & operator=(const EdgePressureInformation & other) {
            length = other.length;
            force = other.force;
            return *this;
        }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgePressureInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgePressureInformation& /*ei*/ )
        {
            return in;
        }
    };

    sofa::core::topology::EdgeSubsetData<sofa::type::vector< EdgePressureInformation> > edgePressureMap; ///< map between edge indices and their pressure

    sofa::core::topology::BaseMeshTopology* _completeTopology{nullptr};

    Data<Deriv> pressure; ///< Pressure force per unit area
    Data<type::vector<Index> > edgeIndices; ///< Indices of edges separated with commas where a pressure is applied
    Data<type::vector<sofa::core::topology::Edge> > edges; ///< List of edges where a pressure is applied
    Data<Deriv> normal; ///< the normal used to define the edge subjected to the pressure force
    Data<Real> dmin; ///< coordinates min of the plane for the vertex selection
    Data<Real> dmax;///< coordinates max of the plane for the vertex selection
    Data< SReal > arrowSizeCoef; ///< for drawing. The sign changes the direction, 0 doesn't draw arrow
    Data< type::vector<Real> > p_intensity; ///< pressure intensity on edge normal
    Data<Coord> p_binormal; ///< binormal of the 2D plane
    Data<bool> p_showForces; ///< draw arrows of edge pressures

    /// Link to be set to the topology container in the component graph.
    SingleLink<EdgePressureForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    EdgePressureForceField();

    virtual ~EdgePressureForceField();

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* m_topology;
public:
    void init() override;

    void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */) override;

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override;

    void draw(const core::visual::VisualParams* vparams) override;

    void setDminAndDmax(const SReal _dmin, const SReal _dmax);
    void setNormal(const Coord n) { normal.setValue(n);}
    void setPressure(Deriv _pressure) { this->pressure = _pressure; updateEdgeInformation(); }

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) final;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

protected :
    void selectEdgesAlongPlane();
    void selectEdgesFromIndices(const type::vector<Index>& inputIndices);
    void selectEdgesFromString();
    void selectEdgesFromEdgeList();
    void updateEdgeInformation();
    void initEdgeInformation();
    bool isPointInPlane(Coord p);
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_MECHANICALLOAD_API EdgePressureForceField<sofa::defaulttype::Vec3Types>;
#endif // !defined(SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_CPP)

} // namespace sofa::component::mechanicalload
