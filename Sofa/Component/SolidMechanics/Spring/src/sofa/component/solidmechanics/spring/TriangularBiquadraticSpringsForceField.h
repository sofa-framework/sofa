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

template<class DataTypes>
class TriangularBiquadraticSpringsForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularBiquadraticSpringsForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    using Index = sofa::Index;

    class Mat3 : public sofa::type::fixed_array<Deriv,3>
    {
    public:
        Deriv operator*(const Deriv& v)
        {
            return Deriv((*this)[0]*v,(*this)[1]*v,(*this)[2]*v);
        }
        Deriv transposeMultiply(const Deriv& v)
        {
            return Deriv(v[0]*((*this)[0])[0]+v[1]*((*this)[1])[0]+v[2]*((*this)[2][0]),
                    v[0]*((*this)[0][1])+v[1]*((*this)[1][1])+v[2]*((*this)[2][1]),
                    v[0]*((*this)[0][2])+v[1]*((*this)[1][2])+v[2]*((*this)[2][2]));
        }
    };

protected:


    class EdgeRestInformation
    {
    public:
        Real  restSquareLength;	// the rest length
        Real  currentSquareLength; 	// the current edge length
        Real  deltaL2;  // the current unit direction
        Real stiffness;

        EdgeRestInformation() = default;

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

    class TriangleRestInformation
    {
    public:
        Real  gamma[3];	// the angular stiffness
        Real stiffness[3]; // the elongation stiffness
        Mat3 DfDx[3]; /// the edge stiffness matrix

        Coord currentNormal;
        Coord lastValidNormal;
        Real area;
        Real restArea;
        Coord areaVector[3];
        Deriv dp[3];
        Real J;

        TriangleRestInformation() = default;

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleRestInformation& /*tri*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleRestInformation& /*vec*/ )
        {
            return in;
        }
    };

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    Data<sofa::Index> triangleInfo;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    Data<sofa::Index> edgeInfo;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    Data<VecCoord> _initialPoints;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    Data<Real> f_poissonRatio;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    Data<Real> f_youngModulus;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    Data<Real> f_dampingRatio;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    Data<bool> f_useAngularSprings;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    Data<bool> f_compressible;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    Data<Real> f_stiffnessMatrixRegularizationWeight;



    sofa::core::topology::TriangleData<type::vector<TriangleRestInformation> > d_triangleInfo; ///< Internal triangle data
    sofa::core::topology::EdgeData<type::vector<EdgeRestInformation> > d_edgeInfo; ///< Internal edge data
    
    Data < VecCoord >  d_initialPoints; ///< Initial Position

    bool updateMatrix;



    Data<Real> d_poissonRatio; ///< Poisson ratio in Hooke's law
    Data<Real> d_youngModulus; ///< Young modulus in Hooke's law
    Data<Real> d_dampingRatio; ///< Ratio damping/stiffness
    Data<bool> d_useAngularSprings; ///< If Angular Springs should be used or not

    Data<bool> d_compressible; ///< If additional energy penalizing compressibility should be used
    /**** coefficient that controls how the material can cope with very compressible cases
    must be between 0 and 1 : if 0 then the deformation may diverge for large compression
    if 1 then the material can undergo large compression even inverse elements ***/
    Data<Real> d_stiffnessMatrixRegularizationWeight; ///< Regularization of the Stiffnes Matrix (between 0 and 1)

    Real lambda;  /// first Lame coefficient
    Real mu;    /// second Lame coefficient


    TriangularBiquadraticSpringsForceField();

    virtual ~TriangularBiquadraticSpringsForceField();
public:
    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    virtual Real getLambda() const { return lambda;}
    virtual Real getMu() const { return mu;}

    void setYoungModulus(const Real modulus)
    {
        d_youngModulus.setValue((Real)modulus);
    }
    void setPoissonRatio(const Real ratio)
    {
        d_poissonRatio.setValue((Real)ratio);
    }

    void draw(const core::visual::VisualParams* vparams) override;
    /// compute lambda and mu based on the Young modulus and Poisson ratio
    void updateLameCoefficients();

    /** Method to initialize @sa EdgeRestInformation when a new edge is created.
    * Will be set as creation callback in the EdgeData @sa d_edgeInfo
    */
    void applyEdgeCreation(Index edgeIndex,
        EdgeRestInformation& ei,
        const core::topology::BaseMeshTopology::Edge& edge, 
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs);

    /** Method to initialize @sa TriangleRestInformation when a new triangle is created.
    * Will be set as creation callback in the TriangleData @sa d_triangleInfo
    */
    void applyTriangleCreation(Index triangleIndex, TriangleRestInformation& tinfo,
        const core::topology::BaseMeshTopology::Triangle& triangle,
        const sofa::type::vector<Index>& ancestors,
        const sofa::type::vector<SReal>& coefs);

    /** Method to update @sa d_triangleInfo when a triangle is removed.
    * Will be set as destruction callback in the TriangleData @sa d_triangleInfo
    */
    void applyTriangleDestruction(Index triangleIndex, TriangleRestInformation& tinfo);

    /// Link to be set to the topology container in the component graph.
    SingleLink<TriangularBiquadraticSpringsForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected :
    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* m_topology;

    sofa::core::topology::EdgeData<type::vector<EdgeRestInformation> > &getEdgeInfo() {return d_edgeInfo;}
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARBIQUADRATICSPRINGSFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API TriangularBiquadraticSpringsForceField<sofa::defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARBIQUADRATICSPRINGSFORCEFIELD_CPP)

} //namespace sofa::component::solidmechanics::spring
