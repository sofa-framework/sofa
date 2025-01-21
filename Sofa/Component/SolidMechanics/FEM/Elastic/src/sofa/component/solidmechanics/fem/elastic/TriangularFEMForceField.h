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
#include <sofa/component/solidmechanics/fem/elastic/TriangleFEMUtils.h>
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Mat.h>
#include <sofa/core/topology/TopologyData.h>

#ifdef PLOT_CURVE
#include <map>
#endif

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::helper
{
    class ColorMap;
}

namespace sofa::component::solidmechanics::fem::elastic
{

/** corotational triangle from
* @InProceedings{NPF05,
*   author       = "Nesme, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
*   title        = "Efficient, Physically Plausible Finite Elements",
*   booktitle    = "Eurographics (short papers)",
*   month        = "august",
*   year         = "2005",
*   editor       = "J. Dingliana and F. Ganovelli",
*   keywords     = "animation, physical model, elasticity, finite elements",
*   url          = "http://www-evasion.imag.fr/Publications/2005/NPF05"
* }
*/
template<class DataTypes>
class TriangularFEMForceField : public BaseLinearElasticityFEMForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularFEMForceField, DataTypes), SOFA_TEMPLATE(BaseLinearElasticityFEMForceField, DataTypes));

    typedef BaseLinearElasticityFEMForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::BaseMeshTopology::Index Index;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Element;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles VecElement;
    typedef sofa::core::topology::BaseMeshTopology::TrianglesAroundVertex TrianglesAroundVertex;

    typedef sofa::type::Quat<Real> Quat;

    enum {
        LARGE = 0,   ///< Symbol of small displacements triangle solver
        SMALL = 1,   ///< Symbol of large displacements triangle solver
    };

protected:

    bool _anisotropicMaterial;			                 	    /// used to turn on / off optimizations
    typedef type::Vec<6, Real> Displacement;					    ///< the displacement vector
    typedef type::Mat<3, 3, Real> MaterialStiffness;				    ///< the matrix of material stiffness
    typedef sofa::type::vector<MaterialStiffness> VecMaterialStiffness;   ///< a vector of material stiffness matrices
    typedef type::Mat<6, 3, Real> StrainDisplacement;				    ///< the strain-displacement matrix
    typedef type::Mat<6, 6, Real> Stiffness;					    ///< the stiffness matrix
    typedef sofa::type::vector<StrainDisplacement> VecStrainDisplacement; ///< a vector of strain-displacement matrices
    typedef type::Mat<3, 3, Real > Transformation;				    ///< matrix for rigid transformations like rotations


protected:
    /// ForceField API
    TriangularFEMForceField();

    ~TriangularFEMForceField() override;
public:
    void init() override;
    void reinit() override;
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;
    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;

    void computeElementStiffnessMatrix(type::Mat<9, 9, Real>& S, type::Mat<9, 9, Real>& SR, const MaterialStiffness& K, const StrainDisplacement& J, const Transformation& Rot);
    void addKToMatrix(sofa::linearalgebra::BaseMatrix *mat, SReal k, unsigned int &offset) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;

    void draw(const core::visual::VisualParams* vparams) override;

    /// Class to store FEM information on each triangle, for topology modification handling
    class TriangleInformation
    {
    public:


        /// material stiffness matrices of each triangle
        MaterialStiffness materialMatrix;
        ///< the strain-displacement matrices vector
        StrainDisplacement strainDisplacementMatrix;
        ///< the stiffness matrix
        Stiffness stiffness;
        Real area;
        // large displacement method
        type::fixed_array<Coord,3> rotatedInitialElements;
        Transformation rotation;
        // strain vector
        type::Vec<3,Real> strain;
        // stress vector
        type::Vec<3,Real> stress;
        Transformation initialTransformation;
        Coord principalStressDirection;
        Real maxStress;
        Coord principalStrainDirection;
        Real maxStrain;
        Real differenceToCriteria;

        type::vector<Coord> lastNStressDirection;

        TriangleInformation() = default;

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleInformation& /*ti*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleInformation& /*ti*/ )
        {
            return in;
        }
    };


    /// Class to store FEM information on each vertex, for topology modification handling
    class VertexInformation
    {
    public:
        VertexInformation()
            :sumEigenValues(0.0), stress(0.0) {}

        Coord meanStrainDirection;
        double sumEigenValues;
        Transformation rotation;

        double stress; //average stress of triangles around (used only for drawing)

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const VertexInformation& /*vi*/)
        {
            return os;
        }
        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, VertexInformation& /*vi*/)
        {
            return in;
        }
    };

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::vector<TriangleInformation>> triangleInfo;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::vector<VertexInformation> > vertexInfo;

    /// Topology Data
    core::topology::TriangleData<sofa::type::vector<TriangleInformation> > d_triangleInfo;
    core::topology::PointData<sofa::type::vector<VertexInformation> > d_vertexInfo; ///< Internal point data

    /** Method to initialize @sa TriangleInformation when a new Triangle is created.
    * Will be set as creation callback in the TriangleData @sa d_triangleInfo
    */
    void createTriangleInformation(Index triangleIndex, TriangleInformation&,
        const core::topology::BaseMeshTopology::Triangle& t,
        const sofa::type::vector< Index >&,
        const sofa::type::vector< SReal >&);

    /// Get/Set methods

    int  getMethod() { return method; }
    void setMethod(int val);
    void setMethod(const std::string& methodName);

public:
    void getFractureCriteria(int element, Deriv& direction, Real& value);
    /// Compute value of stress along a given direction (typically the fiber direction and transverse direction in anisotropic materials)
    void computeStressAlongDirection(Real &stress_along_dir, Index elementIndex, const Coord &direction, const type::Vec<3,Real> &stress);
    /// Compute value of stress along a given direction (typically the fiber direction and transverse direction in anisotropic materials)
    void computeStressAlongDirection(Real &stress_along_dir, Index elementIndex, const Coord &direction);
    /// Compute value of stress across a given direction (typically the fracture direction)
    void computeStressAcrossDirection(Real &stress_across_dir, Index elementIndex, const Coord &direction, const type::Vec<3,Real> &stress);
    /// Compute value of stress across a given direction (typically the fracture direction)
    void computeStressAcrossDirection(Real &stress_across_dir, Index elementIndex, const Coord &direction);
    /// Compute current stress
    void computeStress(type::Vec<3,Real> &stress, Index elementIndex);

    // Getting the rotation of the vertex by averaing the rotation of neighboring elements
    void getRotation(Transformation& R, Index nodeIdx);
    void getRotations();

protected :
    /// Forcefield computations
    void computeStiffness(Stiffness &K, const StrainDisplacement& J, const MaterialStiffness &D);
    void computePrincipalStrain(Index elementIndex, TriangleInformation& triangleInfo);
    void computePrincipalStress(Index elementIndex, TriangleInformation& triangleInfo);
    void computeStressPerVertex(); ///< Method to compute the averageStress per vertex. Call if @sa d_showStressValue is true

    /// f += Kx where K is the stiffness matrix and x a displacement
    virtual void applyStiffness( VecCoord& f, Real h, const VecCoord& x, const Real &kFactor );
    virtual void computeMaterialStiffness(int i, Index& a, Index& b, Index& c);

    ////////////// small displacements method
    void initSmall(int i, Index&a, Index&b, Index&c);
    void accumulateForceSmall( VecCoord& f, const VecCoord & p);
    void applyStiffnessSmall( VecCoord& f, Real h, const VecCoord& x, const Real &kFactor );

    ////////////// large displacements method
    void initLarge(int i, Index&a, Index&b, Index&c);
    void accumulateForceLarge( VecCoord& f, const VecCoord & p);
    void applyStiffnessLarge( VecCoord& f, Real h, const VecCoord& x, const Real &kFactor );

    bool updateMatrix;

public:

    /// Forcefield intern parameters
    int method;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<std::string> f_method;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<type::vector<Real>> f_poisson;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<type::vector<Real>> f_young;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::vector<type::fixed_array<Coord,3> >  > m_rotatedInitialElements;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::vector<Transformation> > m_initialTransformation;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> hosfordExponant;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> criteriaValue;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> showStressValue;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> showStressVector;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> showFracturableTriangles;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> f_computePrincipalStress;

    Data<std::string> d_method; ///< large: large displacements, small: small displacements

    /// Initial strain parameters (if FEM is initialised with predefine values)
    Data< sofa::type::vector<type::fixed_array<Coord,3> > > d_rotatedInitialElements;
    Data< sofa::type::vector<Transformation> > d_initialTransformation; ///< Flag activating rendering of stress directions within each triangle

    /// Fracture parameters
    Data<Real> d_hosfordExponant; ///< Exponent in the Hosford yield criteria
    Data<Real> d_criteriaValue; ///< Fracturable threshold used to draw fracturable triangles

    /// Display parameters
    Data<bool> d_showStressValue;
    Data<bool> d_showStressVector; ///< Flag activating rendering of stress directions within each triangle
    Data<bool> d_showFracturableTriangles; ///< Flag activating rendering of triangles to fracture

    Data<bool> d_computePrincipalStress; ///< Compute principal stress for each triangle

#ifdef PLOT_CURVE
    //structures to save values for each element along time
    sofa::type::vector<std::map<std::string, sofa::type::vector<double> > > allGraphStress;
    sofa::type::vector<std::map<std::string, sofa::type::vector<double> > > allGraphCriteria;
    sofa::type::vector<std::map<std::string, sofa::type::vector<double> > > allGraphOrientation;

    //the index of element we want to display the graphs
    Data<Real>  elementID; ///< element id to follow in the graphs

    //data storing the values along time for the element with index elementID
    Data<std::map < std::string, sofa::type::vector<double> > > f_graphStress; ///< Graph of max stress corresponding to the element id
    Data<std::map < std::string, sofa::type::vector<double> > > f_graphCriteria; ///< Graph of the fracture criteria corresponding to the element id
    Data<std::map < std::string, sofa::type::vector<double> > > f_graphOrientation; ///< Graph of the orientation of the principal stress direction corresponding to the element id
#endif

private:
    bool p_computeDrawInfo; ///< bool set to true if at least one of @sa d_showStressValue, @sa d_showStressVector or @sa d_showFracturableTriangles is true
    sofa::helper::ColorMap* p_drawColorMap; ///< colormap to display the gradiant of stress if @sa d_showStressValue is set to true
    Real m_minStress = 0; ///< min stress computed for @sa d_showStressValue
    Real m_maxStress = 0; ///< max stress computed for @sa d_showStressValue

    TriangleFEMUtils<DataTypes> m_triangleUtils;
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API TriangularFEMForceField<defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_CPP)

} // namespace sofa::component::solidmechanics::fem::elastic
