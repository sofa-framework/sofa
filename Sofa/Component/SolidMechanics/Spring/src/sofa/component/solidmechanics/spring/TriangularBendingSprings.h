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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologyData.h>

namespace sofa::component::solidmechanics::spring
{

/**
Bending springs added between vertices of triangles sharing a common edge.
The springs connect the vertices not belonging to the common edge. It compresses when the surface bends along the common edge.

Note: This TriangularBendingSprings only support manifold triangulation. I.e an edge can only by adjacent to maximum 2 triangles.
If more than 2 triangles are connected to an edge, only one spring will be created (the first 2 triangles encountered during initialisation phase)

@author The SOFA team </www.sofa-framework.org>
*/
template<class DataTypes>
class TriangularBendingSprings : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularBendingSprings, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    enum { N=DataTypes::spatial_dimensions };
    typedef type::Mat<N,N,Real> Mat;
    using Index = sofa::Index;

    Data<Real> d_ks; ///< uniform stiffness for the all springs
    Data<Real> d_kd; ///< uniform damping for the all springs
    Data<bool> d_showSprings; ///< Option to enable/disable the spring display when showForceField is on. True by default

    /// Link to be set to the topology container in the component graph.
    SingleLink<TriangularBendingSprings<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    class EdgeInformation
    {
    public:
        Mat DfDx; ///< the edge stiffness matrix
        int   m1, m2;  ///< the two extremities of the spring: masses m1 and m2
        Real  ks;      ///< spring stiffness (initialized to the default value)
        Real  kd;      ///< damping factor (initialized to the default value)
        Real  restlength; ///< rest length of the spring

        bool is_activated;
        bool is_initialized;

        EdgeInformation(int m1=0, int m2=0, Real restlength=0.0, bool is_activated=false, bool is_initialized=false)
            : m1(m1), m2(m2), ks(Real(100000.0)), kd(Real(1.0)), restlength(restlength), is_activated(is_activated), is_initialized(is_initialized)
        {
        }
        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeInformation& /*ei*/ )
        {
            return in;
        }
    };

    sofa::core::topology::EdgeData<type::vector<EdgeInformation> > edgeInfo; ///< Internal Edge data storing @sa EdgeInformation per edge

protected:
    TriangularBendingSprings();

    virtual ~TriangularBendingSprings();

    /** Method to initialize @sa edgeInfo when a new edge is created. (by default everything is set to 0)
    * Will be set as creation callback in the EdgeData @sa edgeInfo
    */
    void applyEdgeCreation(Index edgeIndex,
        EdgeInformation& ei,
        const core::topology::BaseMeshTopology::Edge&, const sofa::type::vector< Index >&,
        const sofa::type::vector< SReal >&);

    /** Method to update @sa edgeInfo when a new triangle is created.
    * Will be set as callback in the EdgeData @sa edgeInfo when TRIANGLESADDED event is fired
    * to create a new spring between new created triangles.
    */
    void applyTriangleCreation(const type::vector<Index>& triangleAdded,
        const type::vector<core::topology::BaseMeshTopology::Triangle>&,
        const type::vector<type::vector<Index> >&,
        const type::vector<type::vector<SReal> >&);

    /** Method to update @sa edgeInfo when a triangle is removed.
    * Will be set as callback in the EdgeData @sa edgeInfo when TRIANGLESREMOVED event is fired
    * to remove spring if needed or update pair of triangles.
    */
    void applyTriangleDestruction(const type::vector<Index>& triangleRemoved);

    /// Method to update @sa edgeInfo when a point is removed. Will be set as callback when POINTSREMOVED event is fired
    void applyPointDestruction(const type::vector<Index>& pointIndices);

    /// Method to update @sa edgeInfo when points are renumbered. Will be set as callback when POINTSRENUMBERING event is fired
    void applyPointRenumbering(const type::vector<Index>& pointToRenumber);

public:
    // ForceField api
    void init() override;
    void reinit() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    void draw(const core::visual::VisualParams* vparams) override;

    /// Getter/setter on the mesh spring stiffness
    virtual Real getKs() const { return d_ks.getValue();}
    void setKs(const Real ks);

    /// Getter/setter on the mesh spring damping
    virtual Real getKd() const { return d_kd.getValue();}
    void setKd(const Real kd);

    /// Getter to global potential energy accumulated
    SReal getAccumulatedPotentialEnergy() const {return m_potentialEnergy;}

    /// Getter on the potential energy.
    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& d_x) const override;

    sofa::core::topology::EdgeData<type::vector<EdgeInformation> >& getEdgeInfo() { return edgeInfo; }

protected:
    /// poential energy accumulate in method @sa addForce
    SReal m_potentialEnergy;

    /// Pointer to the linked topology used to create this spring forcefield
    sofa::core::topology::BaseMeshTopology* m_topology;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API TriangularBendingSprings<defaulttype::Vec3Types>;
#endif // !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_CPP)


} // namespace sofa::component::solidmechanics::spring
