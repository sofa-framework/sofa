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
#ifndef SOFA_COMPONENT_FORCEFIELD_BEAMPLASTICFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_BEAMPLASTICFEMFORCEFIELD_H

#include <StentExp/config.h>
#include <StentExp/initStentExp.h>
#include "PlasticConstitutiveLaw.h"
#include <StentExp/quadrature/gaussian.h>
#include <StentExp/quadrature/quadrature.h>

#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/TopologyData.h>

#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <string>


namespace sofa
{

namespace component
{

namespace container
{
class StiffnessContainer;
class PoissonContainer;
} // namespace container

namespace forcefield
{

namespace _beamplasticfemforcefield_
{

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
class SOFA_StentExp_API BeamPlasticFEMForceField : public core::behavior::ForceField<DataTypes>
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
    typedef sofa::helper::vector<core::topology::BaseMeshTopology::Edge> VecElement;
    typedef helper::vector<unsigned int> VecIndex;
    typedef defaulttype::Vec<3, Real> Vec3;

    /** \enum MechanicalState
     *  \brief Types of mechanical state associated with the (Gauss) integration
     *  points. The POSTPLASTIC state corresponds to points which underwent plastic
     *  deformation, but on which constraints were released so that the plasticity
     *  process stopped.
     */
    enum MechanicalState {
        ELASTIC = 0,
        PLASTIC = 1,
        POSTPLASTIC = 2,
    };

protected:
    /// Vector representing the displacement of a beam element.
    typedef defaulttype::Vec<12, Real> Displacement;
    /// Matrix for rigid transformations like rotations.
    typedef defaulttype::Mat<3, 3, Real> Transformation;
    /// Stiffness matrix associated to a beam element.
    typedef defaulttype::Mat<12, 12, Real> StiffnessMatrix;

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
        StiffnessMatrix _Ke_loc;
        /**
         * Linearised stiffness matrix (tangent stiffness), updated at each time
         * step for plastic deformation.
         */
        StiffnessMatrix _Kt_loc;

        /// Homogeneous type to a 4th order tensor, in Voigt notation.
        typedef Eigen::Matrix<double, 6, 6> BehaviourMatrix;
        /**
         * Generalised Hooke's law (4th order tensor connecting strain and stress,
         * expressed in Voigt notation)
         */
        BehaviourMatrix _materialBehaviour;

        //Base interval for reduced integration: same for all the beam elements
        ozp::quadrature::detail::Interval<3> _integrationInterval;

        typedef Eigen::Matrix<double, 3, 12> shapeFunction;
        helper::fixed_array<shapeFunction, 27> _N;

        typedef Eigen::Matrix<double, 6, 12> deformationGradientFunction; ///< derivatives of the shape functions (Be)
        helper::fixed_array<deformationGradientFunction, 27> _BeMatrices; /// One Be function for each Gauss Point (27 in one beam element)

        helper::fixed_array<MechanicalState, 27> _pointMechanicalState;

        ///< Indicates which type of mechanical computation should be used.
        ///  The meaning of the three cases is the following :
        ///     - ELASTIC: all the element Gauss points are in an ELASTIC state
        ///     - PLASTIC: at least one Gauss point is in a PLASTIC state.
        ///     - POSTPLASTIC: Gauss points are either in an ELASTIC or POSTPLASTIC state.
        MechanicalState _beamMechanicalState;

        // Plastic strain
        helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> _plasticStrainHistory; ///< history of the plastic strain, one tensor for each Gauss point
        helper::fixed_array<Real, 27> _effectivePlasticStrains;

        // For hardening
        helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> _backStresses;
        helper::fixed_array<Real, 27> _localYieldStresses;

        ///< For drawing
        int _nbCentrelineSeg = 10;
        helper::fixed_array<shapeFunction, 9> _drawN; //TO DO: allow parameterisation of the number of segments
                                                      //       which discretise the centreline (here : 10)
                                                      // NB: we use 9 shape functions bewause extremity points are known

        /*********************************************************************/

        double _E0,_E; //Young
        double _nu;//Poisson
        double _L; //length
        double _zDim; //for rectangular beams: dimension of the cross-section along z axis
        double _yDim; //for rectangular beams: dimension of the cross-section along y axis
        double _G; //shear modulus
        double _Iy;
        double _Iz; //Iz is the cross-section moment of inertia (assuming mass ratio = 1) about the z axis;
        double _J;  //Polar moment of inertia (J = Iy + Iz)
        double _A; // A is the cross-sectional area;
        double _Asy; //_Asy is the y-direction effective shear area =  10/9 (for solid circular section) or 0 for a non-Timoshenko beam
        double _Asz; //_Asz is the z-direction effective shear area;
        StiffnessMatrix _k_loc;

        defaulttype::Quat quat;

        //void localStiffness();
        void init(double E, double yS, double L, double nu, double zSection, double ySection, bool isTimoshenko);

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const BeamInfo& bi )
        {
            os << bi._E0 << " "
                << bi._E << " "
                << bi._nu << " "
                << bi._L << " "
                << bi._zDim << " "
                << bi._yDim << " "
                << bi._G << " "
                << bi._Iy << " "
                << bi._Iz << " "
                << bi._J << " "
                << bi._A << " "
                << bi._Asy << " "
                << bi._Asz << " "
                << bi._Ke_loc << " "
                << bi._Kt_loc << " "
                << bi._k_loc;
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, BeamInfo& bi )
        {
            in	>> bi._E0
                >> bi._E
                >> bi._nu
                >> bi._L
                >> bi._zDim
                >> bi._yDim
                >> bi._G
                >> bi._Iy
                >> bi._Iz
                >> bi._J
                >> bi._A
                >> bi._Asy
                >> bi._Asz
                >> bi._Ke_loc
                >> bi._Kt_loc
                >> bi._k_loc;
            return in;
        }
    };

    topology::EdgeData< sofa::helper::vector<BeamInfo> > m_beamsData;

    class BeamFFEdgeHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,sofa::helper::vector<BeamInfo> >
    {
    public:
        typedef typename BeamPlasticFEMForceField<DataTypes>::BeamInfo BeamInfo;
        BeamFFEdgeHandler(BeamPlasticFEMForceField<DataTypes>* ff, topology::EdgeData<sofa::helper::vector<BeamInfo> >* data)
            :topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,sofa::helper::vector<BeamInfo> >(data),ff(ff) {}

        void applyCreateFunction(unsigned int edgeIndex, BeamInfo&,
                                 const core::topology::BaseMeshTopology::Edge& e,
                                 const sofa::helper::vector<unsigned int> &,
                                 const sofa::helper::vector< double > &);

    protected:
        BeamPlasticFEMForceField<DataTypes>* ff;

    };

    virtual void reset() override;

    /**************************************************************************/
    /*                     Virtual Displacement Method                        */
    /**************************************************************************/

public:

    typedef defaulttype::Vec<12, Real> nodalForces; ///<  Intensities of the nodal forces in the Timoshenko beam element

    typedef Eigen::Matrix<double, 6, 1> VoigtTensor2; ///< Symmetrical tensor of order 2, written with Voigt notation
    typedef Eigen::Matrix<double, 9, 1> VectTensor2; ///< Symmetrical tensor of order 2, written with in vector notation
    typedef Eigen::Matrix<double, 6, 6> VoigtTensor4; ///< Symmetrical tensor of order 4, written with Voigt notation
    typedef Eigen::Matrix<double, 9, 9> VectTensor4; ///< Symmetrical tensor of order 4, written in vector notation
    typedef Eigen::Matrix<double, 12, 1> EigenDisplacement; ///<Nodal displacement
    typedef Eigen::Matrix<double, 12, 1> EigenNodalForces;
    typedef Eigen::Matrix<double, 12, 12> tangentStiffnessMatrix;

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

    // In the elasto-plastic model, the tangent operator can be computed either
    // in a straightforward way, or in a way consistent with the radial return
    // algorithm. This field is used to determine which method will be used.
    // For more information on the consistent tangent operator, we recommend
    // reading the following publications :
    //   - Consistent tangent operators for rate-independent elastoplasticity, Simo and Taylor, 1985
    //   - Studies in anisotropic plasticity with reference to the Hill criterion, De Borst and Feenstra, 1990
    Data<bool> d_useConsistentTangentOperator;

    void computeVDStiffness(int i, Index a, Index b);
    void computeMaterialBehaviour(int i, Index a, Index b);

    typedef helper::fixed_array<VoigtTensor2, 27>  gaussPointStresses; ///< one 6x1 strain tensor for each of the 27 points of integration
    helper::vector<gaussPointStresses> m_prevStresses;
    helper::vector<gaussPointStresses> m_elasticPredictors;

    //Position at the last time step, to handle increments for the plasticity resolution
    VecCoord m_lastPos;

    /************** Plasticity elements ***********************/

    //Newton-Raphson parameters
    double m_NRThreshold;
    unsigned int m_NRMaxIterations;

    // Indicates if the plasticity model is perfect plasticity, or if hardening
    // is represented. The only hardening model we implement is a linear
    // combination of isotropic and kinematic hardening, as described in :
    // Theoretical foundation for large scale computations for nonlinear material
    // behaviour, Hugues(et al) 1984
    Data<bool> d_isPerfectlyPlastic;

    BeamPlasticFEMForceField<DataTypes>* ff;

    // 1D Contitutive law model, which is in charge of computing the
    // tangent modulus during plastic deformation
    fem::PlasticConstitutiveLaw<DataTypes> *m_ConstitutiveLaw;
    Data<std::string> d_modelName; ///< name of the model, for specialisation

    bool goToPlastic(const VoigtTensor2 &stressTensor, const double yieldStress, const bool verbose=FALSE);
    bool goToPostPlastic(const VoigtTensor2 &stressTensor, const VoigtTensor2 &stressIncrement,
                         const bool verbose = FALSE);

    void computeLocalDisplacement(const VecCoord& x, Displacement &localDisp, int i, Index a, Index b);
    void computeDisplacement(const VecCoord& x, const VecCoord& xRef, Displacement &localDisp, int i, Index a, Index b);
    void computeDisplacementWithoutCo(const VecCoord& x, const VecCoord& xRef, Displacement &localDisp, int i, Index a, Index b);
    void computeDisplacementIncrement(const VecCoord& pos, const VecCoord& lastPos, Displacement &currentDisp, Displacement &lastDisp,
                                      Displacement &dispIncrement, int i, Index a, Index b);

    void computeStressIncrement(int index, int gaussPointIt, const VoigtTensor2 &initialStress, VoigtTensor2 &newStressPoint, const VoigtTensor2 &strainIncrement,
                                double &lambdaIncrement, MechanicalState &pointMechanicalState, const Displacement &lastDisp);

    // Plastic modulus
    double computePlasticModulusFromStress(const Eigen::Matrix<double, 6, 1> &stressState);
    double computePlasticModulusFromStrain(int index, int gaussPointId);
    double computeConstPlasticModulus();

    void computeElasticForce(Eigen::Matrix<double, 12, 1> &internalForces, const VecCoord& x, int index, Index a, Index b);
    void computePlasticForce(Eigen::Matrix<double, 12, 1> &internalForces, const VecCoord& x, int index, Index a, Index b);
    void computePostPlasticForce(Eigen::Matrix<double, 12, 1> &internalForces, const VecCoord& x, int index, Index a, Index b);

    //Hardening
    void computeHardeningStressIncrement(int index, int gaussPointIt, const VoigtTensor2 &lastStress, VoigtTensor2 &newStressPoint, const VoigtTensor2 &strainIncrement,
                                         MechanicalState &pointMechanicalState);
    void computeForceWithHardening(Eigen::Matrix<double, 12, 1> &internalForces, const VecCoord& x, int index, Index a, Index b);

    //TESTING : incremental perfect plasticity
    void computePerfectPlasticStressIncrement(int index, int gaussPointIt, const VoigtTensor2 &lastStress, VoigtTensor2 &newStressPoint, const VoigtTensor2 &strainIncrement,
        MechanicalState &pointMechanicalState);
    void computeForceWithPerfectPlasticity(Eigen::Matrix<double, 12, 1> &internalForces, const VecCoord& x, int index, Index a, Index b);

    // Auxiliary methods for hardening
    VectTensor2 voigtToVect2(const VoigtTensor2 &voigtTensor);
    VectTensor4 voigtToVect4(const VoigtTensor4 &voigtTensor);
    VoigtTensor2 vectToVoigt2(const VectTensor2 &vectTensor);
    VoigtTensor4 vectToVoigt4(const VectTensor4 &vectTensor);

    VoigtTensor2 deviatoricStress(const VoigtTensor2 &stressTensor);
    double equivalentStress(const VoigtTensor2 &stressTensor);
    double vonMisesYield(const VoigtTensor2 &stressTensor, const double yieldStress);
    VoigtTensor2 vonMisesGradient(const VoigtTensor2 &stressTensor);
    VectTensor4 vonMisesHessian(const VoigtTensor2 &stressTensor, const double yieldStress);

    //Alternative expressions of the Von Mises functions, for debug
    double vectEquivalentStress(const VectTensor2 &stressTensor);
    double devEquivalentStress(const VoigtTensor2 &stressTensor);
    double devVonMisesYield(const VoigtTensor2 &stressTensor, const double yieldStress);
    double vectVonMisesYield(const VectTensor2 &stressTensor, const double yieldStress);
    VectTensor2 vectVonMisesGradient(const VectTensor2 &stressTensor);
    VoigtTensor2 devVonMisesGradient(const VoigtTensor2 &stressTensor);

    // Special implementation for second-order tensor operations, with the Voigt notation.
    double voigtDotProduct(const VoigtTensor2 &t1, const VoigtTensor2 &t2);
    double voigtTensorNorm(const VoigtTensor2 &t);
    Eigen::Matrix<double, 12, 1> beTTensor2Mult(const Eigen::Matrix<double, 12, 6> &BeT, const VoigtTensor2 &T);
    Eigen::Matrix<double, 12, 12> beTCBeMult(const Eigen::Matrix<double, 12, 6> &BeT, const VoigtTensor4 &C,
                                             const double nu, const double E);

    //Methods called by addForce, addDForce and addKToMatrix when deforming plasticly
    void accumulateNonLinearForce(VecDeriv& f, const VecCoord& x, int i, Index a, Index b);
    void applyNonLinearStiffness(VecDeriv& df, const VecDeriv& dx, int i, Index a, Index b, double fact);
    void updateTangentStiffness(int i, Index a, Index b);


    /**********************************************************/


    const VecElement *m_indexedElements;

    Data<Real> d_poissonRatio;
    Data<Real> d_youngModulus;
    Data<Real> d_yieldStress;
    Data<Real> d_zSection;
    Data<Real> d_ySection;
    Data<bool> d_useSymmetricAssembly;
    Data<bool> d_isTimoshenko;

    double m_lastUpdatedStep;

    container::StiffnessContainer* m_stiffnessContainer;
    container::PoissonContainer* m_poissonContainer;

    defaulttype::Quat& beamQuat(int i);

    sofa::core::topology::BaseMeshTopology* m_topology;
    BeamFFEdgeHandler* m_edgeHandler;

    BeamPlasticFEMForceField();
    BeamPlasticFEMForceField(Real poissonRatio, Real youngModulus, Real yieldStress, Real zSection, Real ySection, bool useVD,
                        bool isPlasticMuller, bool isTimoshenko, bool isPlasticKrabbenhoft, bool isPerfectlyPlastic,
                        helper::vector<defaulttype::Quat> localOrientations);
    virtual ~BeamPlasticFEMForceField();

public:

    virtual void init();
    virtual void bwdInit();
    virtual void reinit();
    virtual void reinitBeam(unsigned int i);

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV );
    virtual void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv&   datadF , const DataVecDeriv&   datadX );
    virtual void addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix );

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    void draw(const core::visual::VisualParams* vparams);
    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

    void setBeam(unsigned int i, double E, double yS, double L, double nu, double zSection, double ySection);
    void initBeams(size_t size);

protected:

    void drawElement(int i, std::vector< defaulttype::Vector3 >* gaussPoints,
                     std::vector< defaulttype::Vector3 >* centrelinePoints,
                     std::vector<defaulttype::Vec<4, float>>* colours, const VecCoord& x);

    void computeStiffness(int i, Index a, Index b);

};

#if !defined(SOFA_COMPONENT_FORCEFIELD_BEAMPLASTICFEMFORCEFIELD_CPP)
extern template class SOFA_StentExp_API BeamPlasticFEMForceField<defaulttype::Rigid3Types>;
#endif

} // namespace _beamplasticfemforcefield_

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_BEAMPLASTICFEMFORCEFIELD_H
