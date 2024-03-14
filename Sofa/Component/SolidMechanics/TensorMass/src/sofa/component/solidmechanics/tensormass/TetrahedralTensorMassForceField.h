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

#include <sofa/component/solidmechanics/tensormass/config.h>



#include <sofa/core/behavior/ForceField.h>
#include <sofa/type/fixed_array.h>
#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologyData.h>
#include <sofa/type/trait/Rebind.h>


namespace sofa::component::solidmechanics::tensormass
{


template<class DataTypes>
class TetrahedralTensorMassForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TetrahedralTensorMassForceField,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
    using VecType = sofa::type::rebind_to<VecCoord, VecCoord>;


    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    using Index = sofa::Index;

protected:

    class EdgeRestInformation
    {
    public:
        sofa::type::Mat<3, 3, Real> DfDx; /// the edge stiffness matrix
        float vertices[2]; // the vertices of this edge

        EdgeRestInformation()
        {
        }
        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeRestInformation& /*eri*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeRestInformation& /*eri*/ )
        {
            return in;
        }
    };
    using edgeRestInfoVector = type::rebind_to<VecCoord, EdgeRestInformation>;

    VecCoord  _initialPoints;///< the intial positions of the points

    bool updateMatrix;

    Data<Real> f_poissonRatio; ///< Poisson ratio in Hooke's law
    Data<Real> f_youngModulus; ///< Young modulus in Hooke's law

    Real lambda;  /// first Lame coefficient
    Real mu;    /// second Lame coefficient

    /// Link to be set to the topology container in the component graph.
    SingleLink<TetrahedralTensorMassForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;


    TetrahedralTensorMassForceField();

    virtual ~TetrahedralTensorMassForceField();

public:

    void init() override;
    void initNeighbourhoodPoints();

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    virtual Real getLambda() const { return lambda;}
    virtual Real getMu() const { return mu;}

    SReal getPotentialEnergy(const core::MechanicalParams* mparams) const override;
    void setYoungModulus(const Real modulus)
    {
        f_youngModulus.setValue(modulus);
    }
    void setPoissonRatio(const Real ratio)
    {
        f_poissonRatio.setValue(ratio);
    }
    void draw(const core::visual::VisualParams* vparams) override;
    /// compute lambda and mu based on the Young modulus and Poisson ratio
    void updateLameCoefficients();

    /** Method to initialize @sa EdgeRestInformation when a new edge is created.
    * Will be set as creation callback in the EdgeData @sa edgeInfo
    */
    void createEdgeRestInformation(Index edgeIndex, EdgeRestInformation& ei,
        const core::topology::BaseMeshTopology::Edge&,
        const sofa::type::vector< Index >&,
        const sofa::type::vector< SReal >&);

    /** Method to update @sa edgeInfo when a new Tetrahedron is created.
    * Will be set as callback in the EdgeData @sa edgeInfo when TETRAHEDRAADDED event is fired
    * to create a new spring in created Tetrahedron.
    */
    void applyTetrahedronCreation(const sofa::type::vector<Index>& tetrahedronAdded,
        const sofa::type::vector<core::topology::BaseMeshTopology::Tetrahedron>&,
        const sofa::type::vector<sofa::type::vector<Index> >&,
        const sofa::type::vector<sofa::type::vector<SReal> >&);

    /** Method to update @sa d_edgeSprings when a triangle is removed.
    * Will be set as callback in the EdgeData @sa edgeInfo when TETRAHEDRAREMOVED event is fired
    * to remove spring if needed or update adjacent Tetrahedron.
    */
    void applyTetrahedronDestruction(const sofa::type::vector<Index>& tetrahedronRemoved);

    core::topology::EdgeData < edgeRestInfoVector >& getEdgeInfo() { return edgeInfo; }

protected:
    core::topology::EdgeData < edgeRestInfoVector > edgeInfo; ///< Internal edge data

    sofa::core::topology::BaseMeshTopology* m_topology;

};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALTENSORMASSFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_TENSORMASS_API TetrahedralTensorMassForceField<sofa::defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALTENSORMASSFORCEFIELD_CPP)


} // namespace sofa::component::solidmechanics::tensormass
