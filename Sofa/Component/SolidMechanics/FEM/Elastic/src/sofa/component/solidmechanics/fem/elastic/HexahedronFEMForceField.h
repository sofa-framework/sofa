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
#include <sofa/component/solidmechanics/fem/elastic/fwd.h>

#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/component/topology/container/grid/SparseGridTopology.h>
#include <sofa/type/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Mat.h>
#include <sofa/core/behavior/BaseRotationFinder.h>
#include <sofa/helper/decompose.h>
#include <sofa/helper/OptionsGroup.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes>
class HexahedronFEMForceField;

template<class DataTypes>
class HexahedronFEMForceFieldInternalData
{
public:
    typedef HexahedronFEMForceField<DataTypes> Main;
    void initPtrData(Main * m)
    {
        auto gatherPt = sofa::helper::getWriteOnlyAccessor(m->d_gatherPt);
        auto gatherBsize = sofa::helper::getWriteOnlyAccessor(m->d_gatherBsize);

        gatherPt.wref().setNames({" "});
        gatherBsize.wref().setNames({" "});
    }
};

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
* WARNING: indices ordering is different than in topology node
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
class HexahedronFEMForceField : virtual public BaseLinearElasticityFEMForceField<DataTypes>, public sofa::core::behavior::BaseRotationFinder
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HexahedronFEMForceField, DataTypes), SOFA_TEMPLATE(BaseLinearElasticityFEMForceField, DataTypes));

    typedef BaseLinearElasticityFEMForceField<DataTypes> InheritForceField;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef helper::ReadAccessor< Data< VecCoord > > RDataRefVecCoord;
    typedef helper::WriteAccessor< Data< VecDeriv > > WDataRefVecDeriv;

    typedef core::topology::BaseMeshTopology::Index Index;
    typedef core::topology::BaseMeshTopology::Hexa Element;
    typedef core::topology::BaseMeshTopology::SeqHexahedra VecElement;

    typedef type::Mat<3, 3, Real> Mat33;

    enum
    {
        LARGE = 0,   ///< Symbol of mean large displacements tetrahedron solver (frame = edges mean on the 3 directions)
        POLAR = 1,   ///< Symbol of polar displacements tetrahedron solver
        SMALL = 2,
    };

public:
    typedef Mat33 Transformation; ///< matrix for rigid transformations like rotations

    int method;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<std::string> f_method;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> f_poissonRatio;

    SOFA_ATTRIBUTE_DISABLED("", "v24.12", "Use d_youngModulus instead") DeprecatedAndRemoved f_youngModulus;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> f_updateStiffnessMatrix;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData< sofa::helper::OptionsGroup >  _gatherPt;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData< sofa::helper::OptionsGroup >  _gatherBsize;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> f_drawing;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> f_drawPercentageOffset;



    Data<std::string> d_method; ///< "large" or "polar" or "small" displacements
    Data<bool> d_updateStiffnessMatrix;
    Data< sofa::helper::OptionsGroup > d_gatherPt; ///< number of dof accumulated per threads during the gather operation (Only use in GPU version)
    Data< sofa::helper::OptionsGroup > d_gatherBsize; ///< number of dof accumulated per threads during the gather operation (Only use in GPU version)
    Data<bool> d_drawing; ///< draw the forcefield if true
    Data<Real> d_drawPercentageOffset; ///< size of the hexa
    bool needUpdateTopology;

    using Inherit1::l_topology;

    void setMethod(int val) ;

    void setUpdateStiffnessMatrix(bool val) { this->d_updateStiffnessMatrix.setValue(val); }

    void init() override;
    void reinit() override;
    void addForce (const core::MechanicalParams* mparams, DataVecDeriv& f,
                           const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce (const core::MechanicalParams* mparams, DataVecDeriv& df,
                            const DataVecDeriv& dx) override;

    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;

    const Transformation& getElementRotation(const sofa::Index elemidx);

    void getNodeRotation(Transformation& R, sofa::Index nodeIdx) ;
    void getRotations(linearalgebra::BaseMatrix * rotations,int offset = 0) override ;

    using Inherit1::addKToMatrix;
    void addKToMatrix(sofa::linearalgebra::BaseMatrix * matrix, SReal kFact, unsigned int &offset) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /* matrices */) override {}

    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

    void draw(const core::visual::VisualParams* vparams) override;

    void handleTopologyChange() override ;

    // Make other overloaded version of getPotentialEnergy() to show up in subclass.
    using InheritForceField::getPotentialEnergy;

protected:
    typedef type::Vec<24, Real> Displacement;                ///< the displacement vector

    typedef type::Mat<6, 6, Real> MaterialStiffness;         ///< the matrix of material stiffness
    typedef type::vector<MaterialStiffness> VecMaterialStiffness; ///< a vector of material stiffness matrices
    VecMaterialStiffness _materialsStiffnesses;                     ///< the material stiffness matrices vector

    typedef type::Mat<24, 24, Real> ElementStiffness;
    typedef type::vector<ElementStiffness> VecElementStiffness;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<VecElementStiffness> _elementStiffnesses;

    Data<VecElementStiffness> d_elementStiffnesses; ///< Stiffness matrices per element (K_i)

    typedef std::pair<int,Real> Col_Value;
    typedef type::vector< Col_Value > CompressedValue;
    typedef type::vector< CompressedValue > CompressedMatrix;
    CompressedMatrix _stiffnesses;
    SReal m_potentialEnergy;

    topology::container::grid::SparseGridTopology* _sparseGrid;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<VecCoord> _initialPoints;

    Data< VecCoord > d_initialPoints; ///< Initial Position

    type::Mat<8,3,int> _coef; ///< coef of each vertices to compute the strain stress matrix

    HexahedronFEMForceFieldInternalData<DataTypes> *data;
    friend class HexahedronFEMForceFieldInternalData<DataTypes>;

protected:
    HexahedronFEMForceField();

    inline const VecElement *getIndexedElements(){ return & (this->l_topology->getHexahedra()); }

    virtual void computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M,
                                          const type::Vec<8, Coord> &nodes, const sofa::Index elementIndice,
                                          double stiffnessFactor=1.0) const;
    static Mat33 integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1,
                              const Real u, const Real v, const Real w, const Mat33& J_1  );

    void computeMaterialStiffness(sofa::Index i);

    static void computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K );


    ////////////// large displacements method
    type::vector<type::fixed_array<Coord,8> > _rotatedInitialElements;   ///< The initials positions in its frame
    type::vector<Transformation> _rotations;
    type::vector<Transformation> _initialrotations;
    void initLarge(sofa::Index i, const Element&elem);
    static void computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey);
    virtual void accumulateForceLarge( WDataRefVecDeriv &f, RDataRefVecCoord &p, sofa::Index i, const Element&elem  );

    ////////////// polar decomposition method
    void initPolar(sofa::Index i, const Element&elem);
    void computeRotationPolar( Transformation &r, type::Vec<8,Coord> &nodes);
    virtual void accumulateForcePolar( WDataRefVecDeriv &f, RDataRefVecCoord &p, sofa::Index i, const Element&elem  );

    ////////////// small decomposition method
    void initSmall(sofa::Index i, const Element&elem);
    virtual void accumulateForceSmall( WDataRefVecDeriv &f, RDataRefVecCoord &p, sofa::Index i, const Element&elem  );

    bool _alreadyInit;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API HexahedronFEMForceField<defaulttype::Vec3Types>;

#endif

} //namespace sofa::component::solidmechanics::fem::elastic
