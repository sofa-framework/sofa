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

#include <map>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>

#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologyData.h>

#define LOCAL_OPTIM

namespace sofa::component::solidmechanics::spring
{
/**
Bending elastic force added between vertices of triangles sharing a common edge.

Adapted from: P. Volino, N. Magnenat-Thalmann. Simple Linear Bending Stiffness in Particle Systems.
Eurographics Symposium on Computer Animation (SIGGRAPH), pp. 101-105, September 2006. http://www.miralab.ch/repository/papers/165.pdf

 @author François Faure, 2012
*/
template<class _DataTypes>
class FastTriangularBendingSprings : public core::behavior::ForceField< _DataTypes>
{
public:
    typedef _DataTypes DataTypes;
    SOFA_CLASS(SOFA_TEMPLATE(FastTriangularBendingSprings, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    enum { N=DataTypes::spatial_dimensions };
    typedef type::Mat<N,N,Real> Mat;

    using Index = sofa::Index;

    Data<SReal> d_bendingStiffness;  ///< Material parameter
    Data<SReal> d_minDistValidity; ///< Minimal distance to consider a spring valid

    /// Link to be set to the topology container in the component graph. 
    SingleLink<FastTriangularBendingSprings<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    /// Searches triangle topology and creates the bending springs
    void init() override;

    void reinit() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    void addKToMatrix(sofa::linearalgebra::BaseMatrix *mat, SReal k, unsigned int &offset) override; // compute and add all the element stiffnesses to the global stiffness matrix
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;
    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& d_x) const override;

    void draw(const core::visual::VisualParams* vparams) override;

protected:



    class EdgeSpring
    {
    public:
        enum {A=0,B,C,D};     ///< vertex names as in Volino's paper
        sofa::type::Vec<4,unsigned> vid;  ///< vertex indices, in circular order
        sofa::type::Vec<4,Real> alpha;    ///< weight of each vertex in the bending vector
        //mutable Deriv dpKfact[4];
        Real lambda;          ///< bending stiffness

        bool is_activated;

        bool is_initialized;

        typedef type::Mat<12,12,Real> StiffnessMatrix;

        /// Store the vertex indices and perform all the precomputations
        void setEdgeSpring( const VecCoord& p, unsigned iA, unsigned iB, unsigned iC, unsigned iD, Real materialBendingStiffness );

        /// Accumulates force and return potential energy
        Real addForce( VecDeriv& f, const VecCoord& p, const VecDeriv& /*v*/) const;

#ifdef LOCAL_OPTIM
        // Optimized version of addDForce
        void addDForce( VecDeriv& df, const VecDeriv& dp, Real kfactor) const
        {
            if( !is_activated ) return;

            Deriv dpKfact[4];

            dpKfact[0] = dp[vid[0]] * lambda * kfactor * alpha[0];
            dpKfact[1] = dp[vid[1]] * lambda * kfactor * alpha[1];
            dpKfact[2] = dp[vid[2]] * lambda * kfactor * alpha[2];
            dpKfact[3] = dp[vid[3]] * lambda * kfactor * alpha[3];

            for( unsigned j=0; j<4; ++j)
            {
                for( unsigned k=0; k<4; ++k)
                {
                    df[vid[j]] -= dpKfact[k] * alpha[j];
                }
            }
        }
#else
        void addDForce( VecDeriv& df, const VecDeriv& dp, Real kfactor) const
        {
            if( !is_activated ) return;
            for( unsigned j=0; j<4; j++ )
                for( unsigned k=0; k<4; k++ )
                    df[vid[j]] -= dp[vid[k]] * lambda * alpha[j] * alpha[k] * kfactor;
        }
#endif

        /// Stiffness matrix assembly
        void addStiffness( sofa::linearalgebra::BaseMatrix *bm, unsigned int offset, SReal scale, core::behavior::ForceField< _DataTypes>* ff ) const;
        /// Compliant stiffness matrix assembly
        void getStiffness( StiffnessMatrix &K ) const;
        /// replace a vertex index with another one
        void replaceIndex( Index oldIndex, Index newIndex );
        /// replace all the vertex indices with the given ones
        void replaceIndices( const type::vector<Index> &newIndices );

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeSpring& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeSpring& /*ei*/ )
        {
            return in;
        }
    };

    /// The list of edge springs, one for each edge between two triangles
    sofa::core::topology::EdgeData<type::vector<EdgeSpring> > d_edgeSprings;

    /** Method to initialize @sa EdgeSpring when a new edge is created.
    * Will be set as creation callback in the EdgeData @sa d_edgeSprings
    */
    void applyEdgeCreation(Index edgeIndex,
        EdgeSpring& ei,
        const core::topology::BaseMeshTopology::Edge&, const sofa::type::vector< Index >&,
        const sofa::type::vector< SReal >&);

    /** Method to update @sa d_edgeSprings when a new triangle is created.
    * Will be set as callback in the EdgeData @sa d_edgeSprings when TRIANGLESADDED event is fired
    * to create a new spring between new created triangles.
    */
    void applyTriangleCreation(const sofa::type::vector<Index>& triangleAdded,
        const sofa::type::vector<core::topology::BaseMeshTopology::Triangle>&,
        const sofa::type::vector<sofa::type::vector<Index> >&,
        const sofa::type::vector<sofa::type::vector<SReal> >&);

    /** Method to update @sa d_edgeSprings when a triangle is removed.
    * Will be set as callback in the EdgeData @sa d_edgeSprings when TRIANGLESREMOVED event is fired
    * to remove spring if needed or update pair of triangles.
    */
    void applyTriangleDestruction(const sofa::type::vector<Index>& triangleRemoved);

    /// Method to update @sa d_edgeSprings when a point is removed. Will be set as callback when POINTSREMOVED event is fired
    void applyPointDestruction(const sofa::type::vector<Index>& pointIndices);

    /// Method to update @sa d_edgeSprings when points are renumbered. Will be set as callback when POINTSRENUMBERING event is fired
    void applyPointRenumbering(const sofa::type::vector<Index>& pointToRenumber);

    sofa::core::topology::BaseMeshTopology* m_topology;


    FastTriangularBendingSprings();

    virtual ~FastTriangularBendingSprings();

    sofa::core::topology::EdgeData<type::vector<EdgeSpring> > &getEdgeInfo() {return d_edgeSprings;}

    SReal m_potentialEnergy;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_FastTriangularBendingSprings_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FastTriangularBendingSprings<defaulttype::Vec3Types>;

#endif // !defined(SOFA_COMPONENT_FORCEFIELD_FastTriangularBendingSprings_CPP)


} // namespace sofa::component::solidmechanics::spring
