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
#include <BeamPlastic/config.h>
#include <BeamPlastic/constitutiveLaw/PlasticConstitutiveLaw.h>
#include <BeamPlastic/quadrature/gaussian.h>

#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

#include <Eigen/Geometry>
#include <string>


namespace sofa::plugin::beamplastic::component::forcefield
{

namespace _beamplasticfemforcefield_
{

using sofa::component::topology::TopologyDataHandler;
using core::topology::BaseMeshTopology;
using sofa::component::topology::EdgeData;
using sofa::plugin::beamplastic::component::constitutivelaw::PlasticConstitutiveLaw;
using type::Vec;
using type::Mat;
using type::Vector3;
using type::Quat;


/** \class BeamPlasticFEMForceField
 *  \brief Compute Finite Element forces based on 6D plastic beam elements.
 * 
 *  This class extends the BeamFEMForceField component to nonlinear plastic
 *  behaviours. The main difference with the linear elastic scenario is that
 *  the stiffness matrix used in the force computations is no longer constant
 *  and has to be recomputed at each time step (as soon as plastic deformation
 *  occurs).
 *  This type of mechanical behaviour allows to simulate irreversible
 *  deformation, which typically occurs in metals.
 */
template<class DataTypes>
class SOFA_BeamPlastic_API BeamPlasticFEMForceField : public core::behavior::ForceField<DataTypes>
{

public:

    SOFA_CLASS(SOFA_TEMPLATE(BeamPlasticFEMForceField,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

    typedef typename DataTypes::Real        Real        ;
    typedef typename DataTypes::Coord       Coord       ;
    typedef typename DataTypes::Deriv       Deriv       ;
    typedef typename DataTypes::VecCoord    VecCoord    ;
    typedef typename DataTypes::VecDeriv    VecDeriv    ;
    typedef typename DataTypes::VecReal     VecReal     ;
    typedef Data<VecCoord>                  DataVecCoord;
    typedef Data<VecDeriv>                  DataVecDeriv;
    typedef VecCoord Vector;

    typedef unsigned int Index;
    typedef core::topology::BaseMeshTopology::Edge Element;
    typedef type::vector<core::topology::BaseMeshTopology::Edge> VecElement;
    typedef sofa::type::RGBAColor RGBAColor;

    typedef Vec<3, Real> Vec3;
    typedef Vec<9, Real> Vec9;
    typedef Vec<12, Real> Vec12;

    typedef Mat<3, 12, Real> Matrix3x12; // Matrix form of the beam element shape functions
    typedef Mat<6, 6, Real> Matrix6x6; ///< Fourth-order order tensor, in Voigt notation.
    typedef Mat<6, 12, Real> Matrix6x12; // Homogeneous to the derivative of the shape function matrix, in Voigt notation
    typedef Mat<9, 9, Real> Matrix9x9;  ///< Fourth-order tensor in vector notation
    typedef Mat<9, 12, Real> Matrix9x12; // Homogeneous to the derivative of the shape function matrix, in vector notation
    typedef Mat<12, 1, Real> Matrix12x1; ///< Nodal displacement, forces
    typedef Mat<12, 3, Real> Matrix12x3;
    typedef Mat<12, 6, Real> Matrix12x6;
    typedef Mat<12, 12, Real> Matrix12x12; ///< Tangent stiffness matrix

    typedef Mat<6, 1, Real> VoigtTensor2; ///< Symmetrical tensor of order 2, written with Voigt notation
    typedef Mat<9, 1, Real> VectTensor2; ///< Symmetrical tensor of order 2, written with vector notation
    typedef Mat<6, 6, Real> VoigtTensor4; ///< Symmetrical tensor of order 4, written with Voigt notation
    typedef Mat<9, 9, Real> VectTensor4; ///< Symmetrical tensor of order 4, written with vector notation

    /** \enum class MechanicalState
     *  \brief Types of mechanical state associated with the (Gauss) integration
     *  points. The POSTPLASTIC state corresponds to points which underwent plastic
     *  deformation, but on which constraints were released so that the plasticity
     *  process stopped.
     */
    enum class MechanicalState {
        ELASTIC = 0,
        PLASTIC = 1,
        POSTPLASTIC = 2,
    };

    ///<3-dimensional Gauss point for reduced integration
    class GaussPoint3
    {
    public:
        GaussPoint3() {}
        GaussPoint3(Real x, Real y, Real z, Real w1, Real w2, Real w3);
        ~GaussPoint3() {}

        auto getNx() const -> const Matrix3x12&;
        void setNx(Matrix3x12 Nx);

        auto getGradN() const -> const Matrix9x12&;
        void setGradN(Matrix9x12 gradN);

        auto getMechanicalState() const -> const MechanicalState;
        void setMechanicalState(MechanicalState newState);

        auto getPrevStress() const -> const Vec9&;
        void setPrevStress(Vec9 newStress);

        auto getWeights() const -> const Vec3&;
        void setWeights(Vec3 weights);

        auto getCoord() const -> const Vec3&;
        void setCoord(Vec3 coord);

        auto getBackStress() const -> const Vec9&;
        void setBackStress(Vec9 backStress);

        auto getYieldStress() const ->const Real;
        void setYieldStress(Real yieldStress);

        auto getPlasticStrain() const -> const Vec9&;
        void setPlasticStrain(Vec9 plasticStrain);

        auto getEffectivePlasticStrain() const ->const Real;
        void setEffectivePlasticStrain(Real effectivePlasticStrain);

    protected:
        Vec3 m_coordinates;
        Vec3 m_weights;

        Matrix3x12 m_Nx; /// Shape functions value for the Gauss point (matrix form)
        Matrix9x12 m_gradN; /// Small strain hypothesis deformation gradient, applied to the beam shape functions (matrix form)
        Matrix6x12 m_gradNVoigt; /// Same in Voigt notation
        MechanicalState m_mechanicalState; /// State of the Gauss point deformation (elastic, plastic, or postplastic)
        Vec9 m_prevStress; /// Value of the stress tensor at previous time step
        Vec9 m_backStress; /// Centre of the yield surface, in stress space
        Real m_yieldStress; /// Elastic limit, varying if plastic deformation occurs

        //Plasticity history variables
        Vec9 m_plasticStrain;
        Real m_effectivePlasticStrain;
    };

    ///<3 Real intervals [a1,b1], [a2,b2] and [a3,b3], for 3D reduced integration
    class Interval3
    {
    public:
        //By default, integration is considered over [-1,1]*[-1,1]*[-1,1].
        Interval3();
        Interval3(Real a1, Real b1, Real a2, Real b2, Real a3, Real b3);
        ~Interval3() {}

        auto geta1() const->Real;
        auto getb1() const->Real;
        auto geta2() const->Real;
        auto getb2() const->Real;
        auto geta3() const->Real;
        auto getb3() const->Real;

    protected:
        Real m_a1, m_b1, m_a2, m_b2, m_a3, m_b3;
    };

protected:

    /**
     * \struct BeamInfo
     * \brief Data structure containing the main characteristics of the beam
     * elements. This includes mechanical and geometric parameters (Young's
     * modulus, Poisson ratio, length, section dimensions, ...), computation
     * variables (stiffness matrix, plasticity history, ...) and visualisation
     * data (shape functions, discretisation parameters).
     */
    struct BeamInfo
    {
        /*********************************************************************/
        /*                     Virtual Displacement method                   */
        /*********************************************************************/

        /// Precomputed stiffness matrix, used for elastic deformation.
        Matrix12x12 _Ke_loc;
        /**
         * Linearised stiffness matrix (tangent stiffness), updated at each time
         * step for plastic deformation.
         */
        Matrix12x12 _Kt_loc;

        /**
         * Generalised Hooke's law (4th order tensor connecting strain and stress,
         * expressed in Voigt notation)
         */
        Matrix6x6 _materialBehaviour;

        /**
         * \brief Integration ranges for Gaussian reduced integration.
         * Data structure defined in the quadrature library used here for
         * Gaussian integration, containing the limits of the integration
         * ranges. Here the integration is performed in 3D, and the variable
         * contains 6 real numbers, corresponding to 3 pairs of limits. These
         * numbers depend on the beam element dimensions.
         */
        ozp::quadrature::detail::Interval<3> _integrationInterval;

        /// Shape function matrices, evaluated in each Gauss point used in reduced integration.
        Vec<27, Matrix3x12> _N;
        // TO DO : define the "27" constant properly ! static const ? ifdef global definition ?

        /// Derivatives of the shape function matrices in _N, also evaluated in each Gauss point
        Vec<27, Matrix6x12> _BeMatrices;

        /// Mechanical states (elastic, plastic, or postplastic) of all gauss points in the beam element.
        Vec<27, MechanicalState> _pointMechanicalState;
        /**
         * Indicates which type of mechanical computation should be used.
         * The meaning of the three cases is the following :
         *   - ELASTIC: all the element Gauss points are in an ELASTIC state
         *   - PLASTIC: at least one Gauss point is in a PLASTIC state.
         *   - POSTPLASTIC: Gauss points are either in an ELASTIC or POSTPLASTIC state.
         */
        MechanicalState _beamMechanicalState;

        //---------- Plastic variables ----------//

        /// History of plastic strain, one tensor for each Gauss point in the element.
        Vec<27, VoigtTensor2> _plasticStrainHistory;
        /**
         * Effective plastic strain, for each Gauss point in the element.
         * The effective plastic strain is only used to compute the tangent
         * modulus if it is not constant.
         */
        Vec<27, Real> _effectivePlasticStrains;

        /// Tensor representing the yield surface centre, one for each Gauss point in the element.
        Vec<27, VoigtTensor2> _backStresses;
        /// Yield threshold, one for each Gauss point in the element.
        Vec<27, Real> _localYieldStresses;

        //---------- Visualisation ----------//

        /// Number of interpolation segments to visualise the centreline of the beam element
        int _nbCentrelineSeg = 10;

        /// Precomputation of the shape functions matrices for each centreline point coordinates.
        Vec<9, Matrix3x12> _drawN; //TO DO: allow parameterisation of the number of segments
                                     //       which discretise the centreline (here : 10)
                                     // NB: we use 9 shape functions because extremity points are known

        /*********************************************************************/

        double _E; ///< Young Modulus
        double _nu; ///< Poisson ratio
        double _L; ///< Length of the beam element
        double _zDim; ///< for rectangular beams: dimension of the cross-section along the local z axis
        double _yDim; ///< for rectangular beams: dimension of the cross-section along the local y axis
        double _G; ///< Shear modulus
        double _Iy; ///< 2nd moment of area with regard to the y axis, for a rectangular beam section
        double _Iz; ///< 2nd moment of area with regard to the z axis, for a rectangular beam section
        double _J; ///< Polar moment of inertia (J = Iy + Iz)
        double _A; ///< Cross-sectional area
        Matrix12x12 _k_loc; ///< Precomputed stiffness matrix, used only for elastic deformation if d_usePrecomputedStiffness = true

        type::Quat<SReal> quat;

        /// Initialisation of BeamInfo members from constructor parameters
        void init(double E, double yS, double L, double nu, double zSection, double ySection, bool isTimoshenko);

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const BeamInfo& bi )
        {
            os << bi._E << " "
                << bi._nu << " "
                << bi._L << " "
                << bi._zDim << " "
                << bi._yDim << " "
                << bi._G << " "
                << bi._Iy << " "
                << bi._Iz << " "
                << bi._J << " "
                << bi._A << " "
                << bi._Ke_loc << " "
                << bi._Kt_loc << " "
                << bi._k_loc;
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, BeamInfo& bi )
        {
            in	>> bi._E
                >> bi._nu
                >> bi._L
                >> bi._zDim
                >> bi._yDim
                >> bi._G
                >> bi._Iy
                >> bi._Iz
                >> bi._J
                >> bi._A
                >> bi._Ke_loc
                >> bi._Kt_loc
                >> bi._k_loc;
            return in;
        }
    };

    EdgeData<  type::vector<BeamInfo> > m_beamsData;

    virtual void reset() override;

    /**************************************************************************/
    /*                     Virtual Displacement Method                        */
    /**************************************************************************/

public:

    typedef Vec<27, GaussPoint3> beamGaussPoints;

protected:

    // Rather than computing the elastic stiffness matrix _Ke_loc by Gaussian
    // reduced integration, we can use a precomputed form, as the matrix remains
    // constant during deformation. The precomputed form _k_loc can be found in
    // litterature, for instance in : Theory of Matrix Structural Analysis,
    // Przemieniecki, 1968, McGraw-Hill, New-York.
    // /!\ This option does not imply that all computations will be made with
    // linear elasticity using _k_loc. It only means that _k_loc will be used
    // instead of _Ke_loc, saving the time of one Gaussian integration per beam
    // element. For purely elastic beam elements, the BeamFEMForceField component
    // should be used.
    Data<bool> d_usePrecomputedStiffness;

    /**
     * In the elasto-plastic model, the tangent operator can be computed either
     * in a straightforward way, or in a way consistent with the radial return
     * algorithm. This data field is used to determine which method will be used.
     * For more information on the consistent tangent operator, we recommend
     * reading the following publications :
     *  - Consistent tangent operators for rate-independent elastoplasticity, Simo and Taylor, 1985
     *  - Studies in anisotropic plasticity with reference to the Hill criterion, De Borst and Feenstra, 1990
     */
    Data<bool> d_useConsistentTangentOperator;

    /**
     * Computes the elastic stiffness matrix _Ke_loc using reduced intergation.
     * The alternative is a precomputation of the elastic stiffness matrix, which is
     * possible for beam elements. The corresponding matrix _k_loc is close of the
     * reduced integration matrix _Ke_loc.
     */
    void computeVDStiffness(int i, Index a, Index b);
    /// Computes the generalised Hooke's law matrix.
    void computeMaterialBehaviour(int i, Index a, Index b);

     /// Used to store stress tensor information (in Voigt notation) for each of the 27 points of integration.
    typedef Vec<27, VoigtTensor2> gaussPointStresses;
    /// Stress tensors fo each Gauss point in every beam element, computed at the previous time step.
    /// These stresses are required for the iterative radial return algorithm if plasticity is detected.
    type::vector<gaussPointStresses> m_prevStresses;
    /// Stress tensors corresponding to the elastic prediction step of the radial return algorithm.
    /// These are stored for the update of the tangent stiffness matrix
    type::vector<gaussPointStresses> m_elasticPredictors;

    /// Position at the last time step, to handle increments for the plasticity resolution
    VecCoord m_lastPos;

    /**
     * Indicates if the plasticity model is perfect plasticity, or if hardening
     * is represented.
     * The only hardening model we implement is a linear combination of isotropic
     * and kinematic hardening, as described in :
     * Theoretical foundation for large scale computations for nonlinear material
     * behaviour, Hugues(et al) 1984.
     */
    Data<bool> d_isPerfectlyPlastic;

    BeamPlasticFEMForceField<DataTypes>* ff;

    //---------- Plastic modulus ----------//
    /**
     * 1D Contitutive law model, which is in charge of computing the
     * plastic modulus during plastic deformation.
     * The constitutive law is used to retrieve a non-constant plastic
     * modulus, with computePlasticModulusFromStress or
     * computePlasticModulusFromStress, but the computeConstPlasticModulus
     * method can be used instead.
     */
    std::unique_ptr<PlasticConstitutiveLaw<DataTypes>> m_ConstitutiveLaw;
    Data<std::string> d_modelName; ///< name of the model, for specialisation

    double computePlasticModulusFromStress(const VoigtTensor2& stressState);
    double computePlasticModulusFromStrain(int index, int gaussPointId);
    static double computeConstPlasticModulus();
    //-------------------------------------//

    /// Tests if the stress tensor of a material point in an elastic state
    /// actually corresponds to plastic deformation.
    bool goToPlastic(const VoigtTensor2 &stressTensor, const double yieldStress, const bool verbose=false);

    /// Computes local displacement of a beam element using the corotational model
    void computeLocalDisplacement(const VecCoord& x, Vec12 &localDisp, int i, Index a, Index b);
    /// Computes a displacement increment between to positions of a beam element (with respect to its local frame)
    void computeDisplacementIncrement(const VecCoord& pos, const VecCoord& lastPos, Vec12 &currentDisp, Vec12 &lastDisp,
                                      Vec12 &dispIncrement, int i, Index a, Index b);

    //---------- Force computation ----------//

    /// Force computation and tangent stiffness matrix update for perfect plasticity
    void computeForceWithPerfectPlasticity(Matrix12x1& internalForces, const VecCoord& x, int index, Index a, Index b);

    /// Stress increment computation for perfect plasticity, based on the radial return algorithm
    void computePerfectPlasticStressIncrement(int index, int gaussPointIt, const VoigtTensor2& lastStress, VoigtTensor2& newStressPoint,
                                              const VoigtTensor2& strainIncrement, MechanicalState& pointMechanicalState);

    /// Force computation and tangent stiffness matrix update for linear mixed (isotropic and kinematic) hardening
    void computeForceWithHardening(Matrix12x1& internalForces, const VecCoord& x, int index, Index a, Index b);

    /// Stress increment computation for linear mixed (isotropic and kinematic) hardening, based on the radial return algorithm
    void computeHardeningStressIncrement(int index, int gaussPointIt, const VoigtTensor2 &lastStress, VoigtTensor2 &newStressPoint,
                                         const VoigtTensor2 &strainIncrement, MechanicalState &pointMechanicalState);

    //---------------------------------------//


    //---------- Gaussian integration ----------//

    /**
     * Vector containing a set of integration Gauss points for each beam element.
     * These Gauss points contain both the necessary coordinates and weights for
     * the gaussian quadrature method, and local mechanical information required
     * for the plasticity computation (shape function matrix, yield stress,
     * back stress, mechanical state, ...)
     */
    type::vector<beamGaussPoints> m_gaussPoints;
    /**
     * Vector containing a set of 3 intervals (Interval3) for each beam element,
     * corresponding to the 3D integration intervals used in the Gaussian
     * quadrature method.
     */
    type::vector<Interval3> m_integrationIntervals;

    /// Initialises the integration intervals of a beam element, for the Gaussian quadrature method.
    void initialiseInterval(int beam, type::vector<Interval3>& integrationIntervals);

    /// Initialises the Gauss points of a beam element, based on its geometrical info.
    void initialiseGaussPoints(int beam, type::vector<beamGaussPoints>& gaussPoints, const Interval3& integrationInterval);

    /**
     * Computes the matrix form of the beam shape functions, used to interpolate
     * a continuous displacement inside the element from the nodes discrete
     * displacement. A timoshenko beam model is used.
     */
    auto computeNx(Real x, Real y, Real z, Real L, Real A, Real Iy, Real Iz,
                   Real E, Real nu, Real kappaY = 1.0, Real kappaZ = 1.0)->Matrix3x12;

    /**
     * Computes the derivative of the matrix form of the beam shape functions.
     * The derivation implements the small strain hypothesis. A timoshenko beam
     * model is used.
     */
    auto computeGradN(Real x, Real y, Real z, Real L, Real A, Real Iy, Real Iz,
                      Real E, Real nu, Real kappaY = 1.0, Real kappaZ = 1.0)->Matrix9x12;

    /**
     * Auxiliary method to change the integration interval for Gaussian quadrature,
     * if it differs from [-1, 1].
     */
    static double changeCoordinate(double x, double a, double b)
    {
        return 0.5 * ((b - a) * x + a + b);
    }
    /**
     * Auxiliary method to change the integration weights for Gaussian quadrature,
     * if the integration interval is not [-1, 1].
     */
    static double changeWeight(double w, double a, double b)
    {
        return 0.5 * (b - a) * w;
    }

    /**
     * Generic implementation of a Gaussian quadrature. This method simply applies
     * a lambda function to a provided set of Gauss points with precomputed weights
     * and coordinates. The actual integration has to be implemented by the lambda
     * function.
     */
    template <typename LambdaType>
    static void integrateBeam(beamGaussPoints& gaussPoints, LambdaType integrationFun);

    //------------------------------------------//


    //---------- Auxiliary methods for Voigt to vector notation conversion ----------//

    // TO DO :
    /* The Voigt notation consists in reducing the dimension of symmetrical tensors by
     * not representing explicitly the symmetrical terms. However these termes have to
     * be taken into account in some operations (such as scalar products) for which
     * they have to be represented explicitly.
     * For the moment, we mostly rely on a full vector notation of all symmetrical
     * variables, and convert them afterwards to Voigt notation in order to reduce the
     * storage cost. In the long term, we should implement all generic functions
     * correcting algebric operations made with Voigt variables (such as voigtDotProduct
     * or voigtTensorNorm), and remove the vector notation.
     */

    /// Converts the 6D Voigt representation of a 2nd-order tensor to a 9D vector representation
    auto voigtToVect2(const VoigtTensor2 &voigtTensor) -> VectTensor2;
    /// Converts the 6x6 Voigt representation of a 4th-order tensor to a 9x9 matrix representation
    auto voigtToVect4(const VoigtTensor4 &voigtTensor) -> VectTensor4;
    /// Converts the 9D vector representation of a 2nd-order tensor to a 6D Voigt representation
    auto vectToVoigt2(const VectTensor2 &vectTensor) -> VoigtTensor2;
    /// Converts the 9x9 matrix representation of a 4th-order tensor to a 6x6 Voigt representation
    auto vectToVoigt4(const VectTensor4 &vectTensor) -> VoigtTensor4;

    // Special implementation for second-order tensor operations, with the Voigt notation.
    static double voigtDotProduct(const VoigtTensor2& t1, const VoigtTensor2& t2);
    double voigtTensorNorm(const VoigtTensor2& t);
    auto beTTensor2Mult(const Matrix12x6& BeT, const VoigtTensor2& T) -> Matrix12x1;
    auto beTCBeMult(const Matrix12x6& BeT, const VoigtTensor4& C,
                    const double nu, const double E) -> Matrix12x12;
    //-------------------------------------------------------------------------------//

    /// Computes the deviatoric stress from a tensor in Voigt notation
    auto deviatoricStress(const VoigtTensor2 &stressTensor) -> VoigtTensor2;
    /// Computes the equivalent stress from a tensor in Voigt notation
    double equivalentStress(const VoigtTensor2 &stressTensor);
    /// Evaluates the Von Mises yield function for given stress tensor (in Voigt notation) and yield stress
    double vonMisesYield(const VoigtTensor2 &stressTensor, const double yieldStress);
    /// Computes the Von Mises yield function gradient (in Voigt notation) at a given stress tensor (in Voigt notation)
    auto vonMisesGradient(const VoigtTensor2 &stressTensor) -> VoigtTensor2;
    /// Computes the Von Mises yield function hessian (in matrix notation) at a given stress tensor (in Voigt notation)
    auto vonMisesHessian(const VoigtTensor2 &stressTensor, const double yieldStress) -> VectTensor4;

    //----- Alternative expressions of the above functions with vector notations -----//
    /// Computes the equivalent stress from a tensor in vector notation
    double vectEquivalentStress(const VectTensor2 &stressTensor);
    /// Evaluates the Von Mises yield function for given stress tensor (in vector notation) and yield stress
    double vectVonMisesYield(const VectTensor2 &stressTensor, const double yieldStress);
    /// Computes the Von Mises yield function gradient (in vector notation) at a given stress tensor (in vector notation)
    VectTensor2 vectVonMisesGradient(const VectTensor2 &stressTensor);

    //----- Alternative functions using the deviatoric stress expression -----//
    // TO DO : is deviatoric computation more efficient than direct computation ?
    /// Computes the equivalent stress from a tensor in Voigt notation, using the deviatoric stress
    double devEquivalentStress(const VoigtTensor2& stressTensor);
    /// Evaluates the Von Mises yield function for given stress tensor (in Voigt notation), using the deviatoric stress
    double devVonMisesYield(const VoigtTensor2& stressTensor, const double yieldStress);
    /// Computes the Von Mises yield function gradient (in Voigt notation) at a given stress tensor (in Voigt notation),
    ///  using the deviatoric stress
    auto devVonMisesGradient(const VoigtTensor2& stressTensor) -> VoigtTensor2;

    //Methods called by addForce, addDForce and addKToMatrix when deforming plasticly
    void accumulateNonLinearForce(VecDeriv& f, const VecCoord& x, int i, Index a, Index b);
    void applyNonLinearStiffness(VecDeriv& df, const VecDeriv& dx, int i, Index a, Index b, double fact);
    void updateTangentStiffness(int i, Index a, Index b);


    /**********************************************************/


    const VecElement *m_indexedElements;

    Data<Real> d_poissonRatio;
    Data<Real> d_youngModulus;
    Data<Real> d_initialYieldStress;
    Data<Real> d_zSection;
    Data<Real> d_ySection;
    Data<bool> d_useSymmetricAssembly;
    Data<bool> d_isTimoshenko;

    Data<std::string> d_sectionShape;

    /// Link to be set to the topology container in the component graph.
    SingleLink<BeamPlasticFEMForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    double m_lastUpdatedStep;

    // Threshold used to compare stress tensor norms to 0. See detailed explanation
    // at the computation of the threshold in the init() method.
    Real m_stressComparisonThreshold;

    Quat<SReal>& beamQuat(int i);

    BaseMeshTopology* m_topology;

    BeamPlasticFEMForceField();
    BeamPlasticFEMForceField(Real poissonRatio, Real youngModulus, Real yieldStress, Real zSection, Real ySection, bool useVD,
                        bool isPlasticMuller, bool isTimoshenko, bool isPlasticKrabbenhoft, bool isPerfectlyPlastic,
                        type::vector<Quat<SReal>> localOrientations);
    ~BeamPlasticFEMForceField() override;

public:
    void init() override;
    void bwdInit() override;
    void reinit() override;
    virtual void reinitBeam(unsigned int i);

    void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;
    void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) override;
    void addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix ) override;

    // TO DO : necessary ?
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    void draw(const core::visual::VisualParams* vparams) override;
    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

    void setBeam(unsigned int i, double E, double yS, double L, double nu, double zSection, double ySection);
    void initBeams(size_t size);

protected:

    void drawElement(int i, std::vector<Vec3> &gaussPoints,
                     std::vector<Vec3> &centrelinePoints,
                     std::vector<RGBAColor> &colours, const VecCoord& x);

    void computeStiffness(int i, Index a, Index b);

};

#if !defined(SOFA_COMPONENT_FORCEFIELD_BEAMPLASTICFEMFORCEFIELD_CPP)
extern template class SOFA_BeamPlastic_API BeamPlasticFEMForceField<defaulttype::Rigid3Types>;
#endif

} // namespace _beamplasticfemforcefield_

} // namespace sofa::plugin::beamplastic::component::forcefield
