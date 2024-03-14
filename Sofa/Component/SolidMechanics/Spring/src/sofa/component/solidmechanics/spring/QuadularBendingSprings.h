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

#include <sofa/type/Mat.h>
#include <sofa/type/fixed_array.h>

#include <map>
#include <set>


namespace sofa::component::solidmechanics::spring
{

/**
Bending springs added between vertices of quads sharing a common edge.
The springs connect the vertices not belonging to the common edge. It compresses when the surface bends along the common edge.
*/
template<class DataTypes>
class QuadularBendingSprings : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(QuadularBendingSprings,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    //typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    enum { N=DataTypes::spatial_dimensions };
    typedef type::Mat<N,N,Real> Mat;

    using Index = sofa::Index;

    QuadularBendingSprings();

    ~QuadularBendingSprings();

protected:

    class EdgeInformation
    {
    public:

        struct Spring
        {
            sofa::topology::Edge edge;
            Real restLength;
            Mat DfDx; /// the edge stiffness matrix
        };

        sofa::type::fixed_array<Spring, 2> springs;

        SReal  ks {};      /// spring stiffness
        SReal  kd {};      /// damping factor

        bool is_activated;

        bool is_initialized;

        EdgeInformation(int m1 = 0, int m2 = 0, int m3 = 0, int m4 = 0,
                        SReal restlength1 = 0_sreal, SReal restlength2 = 0_sreal,
                        const bool is_activated = false, const bool is_initialized = false)
            : springs{
                Spring{ {m1, m2}, restlength1},
                Spring{ {m3, m4}, restlength2}
            }, is_activated(is_activated), is_initialized(is_initialized)
        { }

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

public:
    /// Searches quad topology and creates the bending springs
    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    SReal getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* d_x */) const override;

    virtual SReal getKs() const { return f_ks.getValue();}
    virtual SReal getKd() const { return f_kd.getValue();}

    void setKs(const SReal ks)
    {
        f_ks.setValue((SReal)ks);
    }
    void setKd(const SReal kd)
    {
        f_kd.setValue((SReal)kd);
    }

    void draw(const core::visual::VisualParams* vparams) override;

    sofa::core::topology::EdgeData<sofa::type::vector<EdgeInformation> > &getEdgeInfo() {return edgeInfo;}

    /** Method to initialize @sa EdgeInformation when a new edge is created.
    * Will be set as creation callback in the EdgeData @sa edgeInfo
    */
    void applyEdgeCreation(Index edgeIndex, EdgeInformation& ei,
        const core::topology::BaseMeshTopology::Edge&,
        const sofa::type::vector< Index >&,
        const sofa::type::vector< SReal >&);

    /** Method to update @sa edgeInfo when a new quad is created.
    * Will be set as callback in the EdgeData @sa edgeInfo when QUADSADDED event is fired
    * to create a new spring between new created triangles.
    */
    void applyQuadCreation(const sofa::type::vector<Index>& quadAdded,
        const sofa::type::vector<core::topology::BaseMeshTopology::Quad>&,
        const sofa::type::vector<sofa::type::vector<Index> >&,
        const sofa::type::vector<sofa::type::vector<SReal> >&);

    /** Method to update @sa edgeInfo when a quad is removed.
    * Will be set as callback in the EdgeData @sa edgeInfo when QUADSREMOVED event is fired
    * to remove spring if needed or update pair of quad.
    */
    void applyQuadDestruction(const sofa::type::vector<Index>& quadRemoved);

    /// Method to update @sa edgeInfo when a point is removed. Will be set as callback when POINTSREMOVED event is fired
    void applyPointDestruction(const sofa::type::vector<Index>& pointIndices);

    /// Method to update @sa edgeInfo when points are renumbered. Will be set as callback when POINTSRENUMBERING event is fired
    void applyPointRenumbering(const sofa::type::vector<Index>& pointToRenumber);


    Data<SReal> f_ks; ///< uniform stiffness for the all springs
    Data<SReal> f_kd; ///< uniform damping for the all springs

    /// Link to be set to the topology container in the component graph.
    SingleLink<QuadularBendingSprings<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    sofa::core::topology::EdgeData<sofa::type::vector<EdgeInformation> > edgeInfo; ///< Internal edge data

    struct ForceOutput
    {
        Deriv force;
        Real forceIntensity;
        Real inverseLength;
    };

    ForceOutput computeForce(const VecDeriv& v, const EdgeInformation& einfo, const typename EdgeInformation::Spring& spring, Coord direction, Real distance);
    Mat computeLocalJacobian(EdgeInformation& einfo, const Coord& direction, const ForceOutput& force);
    void computeSpringForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v,
                          EdgeInformation& einfo,
                          typename EdgeInformation::Spring& spring);

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* m_topology;

    bool updateMatrix;
    SReal m_potentialEnergy;
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_QUADULARBENDINGSPRINGS_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API QuadularBendingSprings<sofa::defaulttype::Vec3Types>;



#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_QUADULARBENDINGSPRINGS_CPP)

} // namespace sofa::component::solidmechanics::spring
