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
#include <sofa/component/solidmechanics/fem/elastic/config.h>

#include <sofa/core/behavior/ForceField.h>

#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>

#include <sofa/core/topology/TopologyData.h>

namespace sofa::component::solidmechanics::fem::elastic
{

/** Compute Finite Element forces based on hexahedral elements.
*
* Corotational hexahedron from
* @Article{NMPCPF05,
*   author       = "Nesme, Matthieu and Marchal, Maud and Promayon, Emmanuel and Chabanas, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
*   title        = "Physically Realistic Interactive Simulation for Biological Soft Tissues",
*   journal      = "Recent Research Developments in Biomechanics",
*   volume       = "2",
*   year         = "2005",
*   keywords     = "surgical simulation physical animation truth cube",
*   url          = "http://www-evasion.imag.fr/Publications/2005/NMPCPF05"
* }
*
* indices ordering (same as in HexahedronSetTopology):
*
*     Y  7---------6
*     ^ /         /|
*     |/    Z    / |
*     3----^----2  |
*     |   /     |  |
*     |  4------|--5
*     | /       | /
*     |/        |/
*     0---------1-->X
*/
template<class DataTypes>
class HexahedralFEMForceField : virtual public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HexahedralFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef helper::ReadAccessor< DataVecCoord > RDataRefVecCoord;
    typedef helper::WriteAccessor< DataVecDeriv > WDataRefVecDeriv;

    typedef core::topology::BaseMeshTopology::Index Index;
    typedef core::topology::BaseMeshTopology::Hexa Element;
    typedef core::topology::BaseMeshTopology::SeqHexahedra VecElement;

    typedef type::Vec<24, Real> Displacement;		///< the displacement vector

    typedef type::Mat<6, 6, Real> MaterialStiffness;	///< the matrix of material stiffness
    typedef type::vector<MaterialStiffness> VecMaterialStiffness;  ///< a vector of material stiffness matrices
    typedef type::Mat<24, 24, Real> ElementMass;

    typedef type::Mat<24, 24, Real> ElementStiffness;
    typedef type::vector<ElementStiffness> VecElementStiffness;


    enum
    {
        LARGE = 0,   ///< Symbol of large displacements hexahedron solver
        POLAR = 1,   ///< Symbol of polar displacements hexahedron solver
    };

protected:

    typedef type::Mat<3, 3, Real> Mat33;
    typedef Mat33 Transformation; ///< matrix for rigid transformations like rotations

    typedef std::pair<int,Real> Col_Value;
    typedef type::vector< Col_Value > CompressedValue;
    typedef type::vector< CompressedValue > CompressedMatrix;

    /// the information stored for each hexahedron
    class HexahedronInformation
    {
    public:
        /// material stiffness matrices of each hexahedron
        MaterialStiffness materialMatrix;

        // large displacement method
        type::fixed_array<Coord,8> rotatedInitialElements;

        Transformation rotation;
        ElementStiffness stiffness;

        HexahedronInformation() {}

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const HexahedronInformation& /*hi*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, HexahedronInformation& /*hi*/ )
        {
            return in;
        }
    };


    HexahedralFEMForceField();
    virtual ~HexahedralFEMForceField();
public:
    void setPoissonRatio(Real val) { this->f_poissonRatio.setValue(val); }

    void setYoungModulus(Real val) { this->f_youngModulus.setValue(val); }

    void setMethod(int val) { method = val; }

    void init() override;
    void reinit() override;

    void addForce (const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    void addDForce (const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    void draw(const core::visual::VisualParams* vparams) override;

protected:

    virtual void computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const type::Vec<8,Coord> &nodes);
    Mat33 integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  );

    /// compute the hookean material matrix
    void computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio);

    void computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K );


    ////////////// large displacements method
    void initLarge(const int i);
    void computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey);
    virtual void accumulateForceLarge( WDataRefVecDeriv& f, RDataRefVecCoord& p, const int i);

    ////////////// polar decomposition method
    void initPolar(const int i);
    void computeRotationPolar( Transformation &r, type::Vec<8,Coord> &nodes);
    virtual void accumulateForcePolar( WDataRefVecDeriv& f, RDataRefVecCoord & p, const int i);

public:
    int method;
    Data<std::string> f_method; ///< the computation method of the displacements
    Data<Real> f_poissonRatio;
    Data<Real> f_youngModulus;
    /// container that stotes all requires information for each hexahedron
    core::topology::HexahedronData<sofa::type::vector<HexahedronInformation> > hexahedronInfo;

    /** Method to create @sa HexahedronInformation when a new hexahedron is created.
    * Will be set as creation callback in the HexahedronData @sa hexahedronInfo
    */
    void createHexahedronInformation(Index, HexahedronInformation& t, const core::topology::BaseMeshTopology::Hexahedron&,
        const sofa::type::vector<Index>&, const sofa::type::vector<SReal>&);

protected:
    core::topology::BaseMeshTopology* _topology;

    type::Mat<8,3,int> _coef; ///< coef of each vertices to compute the strain stress matrix
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API HexahedralFEMForceField<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::solidmechanics::fem::elastic
