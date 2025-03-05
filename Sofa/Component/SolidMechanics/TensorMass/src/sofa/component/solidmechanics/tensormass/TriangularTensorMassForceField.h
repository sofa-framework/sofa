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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologyData.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::solidmechanics::tensormass
{

template<class DataTypes>
class TriangularTensorMassForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularTensorMassForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

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
        Mat3 DfDx; /// the edge stiffness matrix

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
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_TENSORMASS()
    sofa::core::objectmodel::lifecycle::RenamedData <sofa::type::vector<EdgeRestInformation>> edgeInfo;

    sofa::core::topology::EdgeData<sofa::type::vector<EdgeRestInformation> > d_edgeInfo; ///< Internal edge data

    /** Method to initialize @sa EdgeRestInformation when a new edge is created.
    * Will be set as creation callback in the EdgeData @sa d_edgeInfo
    */
    void applyEdgeCreation(Index edgeIndex, EdgeRestInformation&,
        const core::topology::BaseMeshTopology::Edge& e,
        const sofa::type::vector<Index>&,
        const sofa::type::vector<SReal>&);

    /** Method to update @sa d_edgeInfo when a new triangle is created.
    * Will be set as callback in the EdgeData @sa d_edgeInfo when TRIANGLESADDED event is fired
    * to create a new spring between new created triangles.
    */
    void applyTriangleCreation(const sofa::type::vector<Index>& triangleAdded,
        const sofa::type::vector<core::topology::BaseMeshTopology::Triangle>&,
        const sofa::type::vector<sofa::type::vector<Index> >&,
        const sofa::type::vector<sofa::type::vector<SReal> >&);

    /** Method to update @sa d_edgeInfo when a triangle is removed.
    * Will be set as callback in the EdgeData @sa d_edgeInfo when TRIANGLESREMOVED event is fired
    * to remove spring if needed or update pair of triangles.
    */
    void applyTriangleDestruction(const sofa::type::vector<Index>& triangleRemoved);

    
    VecCoord  _initialPoints;///< the initial positions of the points

    bool updateMatrix;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_TENSORMASS()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> f_poissonRatio;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_TENSORMASS()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> f_youngModulus;

    Data<Real> d_poissonRatio; ///< Poisson ratio in Hooke's law
    Data<Real> d_youngModulus; ///< Young's modulus in Hooke's law

    /// Link to be set to the topology container in the component graph.
    SingleLink<TriangularTensorMassForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;


    Real lambda;  /// first Lame coefficient
    Real mu;    /// second Lame coefficient

    TriangularTensorMassForceField();

    virtual ~TriangularTensorMassForceField();
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


protected :

    sofa::core::topology::EdgeData<sofa::type::vector<EdgeRestInformation> > &getEdgeInfo() {return d_edgeInfo;}

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* m_topology;

};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARTENSORMASSFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_TENSORMASS_API TriangularTensorMassForceField<sofa::defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARTENSORMASSFORCEFIELD_CPP)

} // namespace sofa::component::solidmechanics::tensormass
