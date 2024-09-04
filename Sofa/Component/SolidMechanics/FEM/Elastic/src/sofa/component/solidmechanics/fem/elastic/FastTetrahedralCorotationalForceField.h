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

#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/TopologyData.h>
#include <sofa/type/fixed_array.h>
#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/type/trait/Rebind.h>

#include <sofa/core/objectmodel/RenamedData.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes>
class FastTetrahedralCorotationalForceField;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class FastTetrahedralCorotationalForceFieldData
{
public:
    typedef FastTetrahedralCorotationalForceField<DataTypes> Main;
    void reinit(Main* m) { SOFA_UNUSED(m); }
};


template<class DataTypes>
class FastTetrahedralCorotationalForceField : public BaseLinearElasticityFEMForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FastTetrahedralCorotationalForceField,DataTypes), SOFA_TEMPLATE(BaseLinearElasticityFEMForceField,DataTypes));

    typedef BaseLinearElasticityFEMForceField<DataTypes> Inherited;
    typedef typename DataTypes::Real        Real        ;
    typedef typename DataTypes::Coord       Coord       ;
    typedef typename DataTypes::Deriv       Deriv       ;
    typedef typename DataTypes::VecCoord    VecCoord    ;
    typedef typename DataTypes::VecDeriv    VecDeriv    ;
    typedef typename DataTypes::VecReal     VecReal     ;
    typedef Data<VecCoord>                  DataVecCoord;
    typedef Data<VecDeriv>                  DataVecDeriv;

    using Mat3x3 = type::Mat<3, 3, Real>;
    using Mat3x3NoInit = type::MatNoInit<3, 3, Real>;

    typedef enum
    {
        POLAR_DECOMPOSITION,
        QR_DECOMPOSITION,
	    POLAR_DECOMPOSITION_MODIFIED,
		LINEAR_ELASTIC
    } RotationDecompositionMethod;

    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::EdgesInTetrahedron EdgesInTetrahedron;
    typedef core::topology::BaseMeshTopology::Tetra Tetrahedron;
    typedef sofa::Index Index;
    
    /// data structure stored for each tetrahedron
    class TetrahedronRestInformation
    {
    public:
        /// shape vector at the rest configuration
        Coord shapeVector[4];
        /// rest volume
        Real restVolume;
        Coord restEdgeVector[6];
        Mat3x3 linearDfDxDiag[4];  // the diagonal 3x3 block matrices that makes the 12x12 linear elastic matrix
        Mat3x3 linearDfDx[6];  // the off-diagonal 3x3 block matrices that makes the 12x12 linear elastic matrix
        Mat3x3 rotation; // rotation from deformed to rest configuration
        Mat3x3 restRotation; // used for QR decomposition

        Real edgeOrientation[6];

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TetrahedronRestInformation& /*eri*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TetrahedronRestInformation& /*eri*/ )
        {
            return in;
        }
    };


public:
    /// Topology Data
    using VecTetrahedronRestInformation = type::rebind_to<VecCoord, TetrahedronRestInformation>;
    using VecMat3x3 = type::rebind_to<VecCoord, Mat3x3>;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::RenamedData<VecMat3x3 > pointInfo;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::RenamedData<VecMat3x3 >  edgeInfo;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::RenamedData<VecTetrahedronRestInformation >  tetrahedronInfo;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::RenamedData<std::string> f_method;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::RenamedData<Real> f_poissonRatio;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    SOFA_ATTRIBUTE_DISABLED("", "v24.12", "Use d_youngModulus instead") DeprecatedAndRemoved f_youngModulus;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::RenamedData<bool> f_drawing;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::RenamedData<sofa::type::RGBAColor> drawColor1;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::RenamedData<sofa::type::RGBAColor> drawColor2;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::RenamedData<sofa::type::RGBAColor> drawColor3;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::RenamedData<sofa::type::RGBAColor> drawColor4;


    core::topology::PointData<VecMat3x3 > d_pointInfo; ///< Internal point data
    core::topology::EdgeData<VecMat3x3 > d_edgeInfo; ///< Internal edge data
    core::topology::TetrahedronData<VecTetrahedronRestInformation > d_tetrahedronInfo; ///< Internal tetrahedron data

    VecCoord  _initialPoints;///< the intial positions of the points
    Data<std::string> d_method; ///<  method for rotation computation :"qr" (by QR) or "polar" or "polar2" or "none" (Linear elastic)
    RotationDecompositionMethod m_decompositionMethod;

    Data<bool> d_drawing; ///<  draw the forcefield if true
    Data<sofa::type::RGBAColor> d_drawColor1; ///<  draw color for faces 1
    Data<sofa::type::RGBAColor> d_drawColor2; ///<  draw color for faces 2
    Data<sofa::type::RGBAColor> d_drawColor3; ///<  draw color for faces 3
    Data<sofa::type::RGBAColor> d_drawColor4; ///<  draw color for faces 4

    using Inherit1::l_topology;

    FastTetrahedralCorotationalForceField();

    virtual ~FastTetrahedralCorotationalForceField();

public:

    void init() override;


    void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;
    void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    using Inherit1::addKToMatrix;
    void addKToMatrix(sofa::linearalgebra::BaseMatrix *m, SReal kFactor, unsigned int &offset) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* matrix) override;

    void updateTopologyInformation();

    void setRotationDecompositionMethod( const RotationDecompositionMethod m)
    {
        m_decompositionMethod = m;
    }
    void draw(const core::visual::VisualParams* vparams) override;

protected :
    static void computeQRRotation( Mat3x3 &r, const Coord *dp);

    /** Method to initialize @sa TetrahedronRestInformation when a new Tetrahedron is created.
    * Will be set as creation callback in the TetrahedronData @sa d_tetrahedronInfo
    */
    void createTetrahedronRestInformation(Index, TetrahedronRestInformation& t,
        const core::topology::BaseMeshTopology::Tetrahedron&,
        const sofa::type::vector<Index>&,
        const sofa::type::vector<SReal>&);

    core::topology::EdgeData< VecMat3x3 > &getEdgeInfo() {return d_edgeInfo;}

    bool updateMatrix;

    typedef FastTetrahedralCorotationalForceFieldData<DataTypes> ExtraData;
    ExtraData m_data;
};

#if !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_FASTTETRAHEDRALCOROTATIONALFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FastTetrahedralCorotationalForceField<sofa::defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::solidmechanics::fem::elastic
