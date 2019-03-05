/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_MULTIBEAMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_MULTIBEAMFORCEFIELD_H

#include <StentExp/config.h>
#include <StentExp/initStentExp.h>
#include "PlasticConstitutiveLaw.h"
#include <StentExp/quadrature/Gaussian.h>
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

/** Compute Finite Element forces based on 6D beam elements.
*/
template<class DataTypes>
class SOFA_StentExp_API MultiBeamForceField : public core::behavior::ForceField<DataTypes>
{

public:

    SOFA_CLASS(SOFA_TEMPLATE(MultiBeamForceField,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

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

    //Types of mechanical state in which the (Gauss) integration points can be
    enum MechanicalState {
        ELASTIC = 0,
        PLASTIC = 1,
        POSTPLASTIC = 2,
    };

protected:

    typedef defaulttype::Vec<12, Real> Displacement;        ///< the displacement vector
    typedef defaulttype::Mat<3, 3, Real> Transformation; ///< matrix for rigid transformations like rotations

    typedef defaulttype::Mat<12, 12, Real> StiffnessMatrix;
    typedef defaulttype::Mat<12, 8, Real> plasticityMatrix; ///< contribution of plasticity to internal forces

    struct BeamInfo
    {
        /*********************************************************************/
        /*                     Virtual Displacement method                   */
        /*********************************************************************/

        plasticityMatrix _M_loc;
        StiffnessMatrix _Ke_loc; //elastic stiffness
        StiffnessMatrix _Kt_loc; //tangent stiffness

        typedef Eigen::Matrix<double, 6, 6> BehaviourMatrix;
        BehaviourMatrix _materialBehaviour;
        BehaviourMatrix _materialInv;

        //Base interval for reduced integration: same for all the beam elements
        ozp::quadrature::detail::Interval<3> _integrationInterval;

        typedef Eigen::Matrix<double, 3, 12> shapeFunction;
        helper::fixed_array<shapeFunction, 27> _N;

        typedef Eigen::Matrix<double, 6, 12> deformationGradientFunction; ///< derivatives of the shape functions (Be)
        helper::fixed_array<deformationGradientFunction, 27> _BeMatrices; /// One Be function for each Gauss Point (27 in one beam element)

        helper::fixed_array<MechanicalState, 27> _pointMechanicalState;
        helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> _plasticStrainHistory; ///< history of the plastic strain, one tensor for each Gauss point

        ///< Indicates which type of mechanical computation should be used.
        ///  The meaning of the three cases is the following :
        ///     - ELASTIC: all the element Gauss points are in an ELASTIC state
        ///     - PLASTIC: at least one Gauss point is in a PLASTIC state.
        ///     - POSTPLASTIC: Gauss points are either in an ELASTIC or POSTPLASTIC state.
        MechanicalState _beamMechanicalState;


        ///< For drawing
        int _nbCentrelineSeg = 10;
        helper::fixed_array<shapeFunction, 9> _drawN; //TO DO: allow parameterisation of the number of segments
                                                      //       which discretise the centreline (here : 10)
                                                      // NB: we use 9 shape functions bewause extremity points are known

        /*********************************************************************/

        // 	static const double FLEXIBILITY=1.00000; // was 1.00001
        double _E0,_E; //Young
        double _yS; //yield Stress
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
        //new: k_loc is the stiffness in the local frame... to compute Ke we only change lambda
        //NewMAT::Matrix  _k_loc;

        // _eigenvalue_loc are 4 diagonal matrices (6x6) representing the eigenvalues of each
        // 6x6 block of _k_loc. _eigenvalue_loc[1] = _eigenvalue_loc[2] since _k_loc[1] = _k_loc[2]
        //NewMAT::DiagonalMatrix  _eigenvalue_loc[4], _inv_eigenvalue_loc[4];
        // k_flex is the stiffness matrix + reinforcement of diagonal (used in gauss-seidel process)
        //NewMAT::Matrix  _k_flex;
        //lambda is a matrix that contains the direction of the local frame in the global frame
        //NewMAT::Matrix  _lambda;
        //non-linear value of the internal forces (computed with previous time step positions) (based on k_loc)
        //NewMAT::ColumnVector  _f_k;
        //initial deformation of the beam (gives the curvature) on the local frame
        //NewMAT::ColumnVector _u_init;
        //actual deformation of the beam on the local frame
        //NewMAT::ColumnVector _u_actual;

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
                << bi._M_loc << " "
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
                >> bi._M_loc
                >> bi._Ke_loc
                >> bi._Kt_loc
                >> bi._k_loc;
            return in;
        }
    };

    topology::EdgeData< sofa::helper::vector<BeamInfo> > beamsData;

    class BeamFFEdgeHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,sofa::helper::vector<BeamInfo> >
    {
    public:
        typedef typename MultiBeamForceField<DataTypes>::BeamInfo BeamInfo;
        BeamFFEdgeHandler(MultiBeamForceField<DataTypes>* ff, topology::EdgeData<sofa::helper::vector<BeamInfo> >* data)
            :topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,sofa::helper::vector<BeamInfo> >(data),ff(ff) {}

        void applyCreateFunction(unsigned int edgeIndex, BeamInfo&,
                                 const core::topology::BaseMeshTopology::Edge& e,
                                 const sofa::helper::vector<unsigned int> &,
                                 const sofa::helper::vector< double > &);

    protected:
        MultiBeamForceField<DataTypes>* ff;

    };

    /**************************************************************************/
    /*                         Multi-beams connexion                          */
    /**************************************************************************/

    //Vector containing all the local orientations, one for each beam element
    //TO DO: for the moment, this orientations can be manually provided with the
    //       Data field _inputLocalOrientations. In this case, the orientations
    //       must be coherent with the corresponding topology
    //       the second fields (_beamLocalOrientation) allows to correct these
    //       user orientations if they are incoherent, and facilitate their
    //       handling
    //       In the future, it would be better to automatically compute the
    //       orientation of each beam, directly from the topology.
    //Handling of the global to local transform
    //TO DO: this is a prototype for a model which has the ability to connect
    //       more than 2 beams. If it works, every redundancy with BeamInfo.quat
    //       should be suppressed
    Data< helper::vector<defaulttype::Quat> > _inputLocalOrientations;
    helper::vector<defaulttype::Quat> _beamLocalOrientations;

    /**************************************************************************/



    /**************************************************************************/
    /*                         Virtual Force Method                           */
    /**************************************************************************/

    /// Virtual Force method, same as in BeamFEMForceField

    defaulttype::Vec<6, Real> _VFPlasticYieldThreshold;
    Real _VFPlasticMaxThreshold;
    Real _VFPlasticCreep;

    typedef defaulttype::Vec<8, Real> VFStrain; ///< 6 strain components used in the 3D Timoshenko beam model
    typedef defaulttype::Vec<12, Real> nodalForces; ///<  Intensities of the nodal forces in the Timoshenko beam element
    typedef defaulttype::Mat<6, 2, Real> plasticLimits; ///< 6 pairs of 1D interval limits (one for each strain)
    typedef defaulttype::Vec<6, bool> completePlasticZones; ///< true if the corresponding plastic zone limits are [0,l]

    helper::vector<VFStrain> _VFPlasticStrains; ///< one plastic strain vector per element
    helper::vector<VFStrain> _VFTotalStrains; ///< one total strain vector per element
    helper::vector<nodalForces> _nodalForces;
    helper::vector<plasticLimits> _plasticZones; ///< one plastic zone per element
    helper::vector<completePlasticZones> _isPlasticZoneComplete;

    virtual void reset() override;

    void initPlasticityMatrix(int i, Index a, Index b);
    void updatePlasticityMatrix(int i, Index a, Index b);
    void updatePlasticity(int i, Index a, Index b);
    void totalStrainEvaluation(int i, Index a, Index b);

    /**************************************************************************/


    /**************************************************************************/
    /*                     Virtual Displacement Method                        */
    /**************************************************************************/

public:

    typedef Eigen::Matrix<double, 6, 1> VoigtTensor2; ///< Symmetrical tensor of order 2, written with Voigt notation
    typedef Eigen::Matrix<double, 6, 6> VoigtTensor4; ///< Symmetrical tensor of order 4, written with Voigt notation
    typedef Eigen::Matrix<double, 12, 1> EigenDisplacement; ///<Nodal displacement
    typedef Eigen::Matrix<double, 12, 1> EigenNodalForces;
    typedef Eigen::Matrix<double, 12, 12> tangentStiffnessMatrix;

protected:

    /// virtual displacement method, same as in TetrahedronFEMForceField
    Data<bool> _virtualDisplacementMethod;
    Data<bool> _isPlasticMuller;

    void computeVDStiffness(int i, Index a, Index b);
    void computeMaterialBehaviour(int i, Index a, Index b);

    typedef helper::fixed_array<VoigtTensor2, 27>  elementPreviousStresses; ///< one 6x1 strain tensor for each of the 27 points of integration
    helper::vector<elementPreviousStresses> _prevStresses;

    //Position at the last time step, to handle increments for the plasticity resolution
    VecCoord _lastPos;

    /************** Plasticity elements ***********************/

    //Newton-Raphson parameters
    double _NRThreshold;
    unsigned int _NRMaxIterations;

    Data<bool> _isPlasticKrabbenhoft;
    Data<bool> _isPerfectlyPlastic;

    MultiBeamForceField<DataTypes>* ff;

    // 1D Contitutive law model, which is in charge of computing the
    // tangent modulus during plastic deformation
    fem::PlasticConstitutiveLaw<DataTypes> *m_ConstitutiveLaw;
    Data<std::string> d_modelName; ///< name of the model, for specialisation

    void updateYieldStress(int beamIndex, double yieldStressIncrement);

    bool goToPlastic(const VoigtTensor2 &stressTensor, const double yieldStress, const bool verbose=FALSE);
    bool goToPostPlastic(const VoigtTensor2 &stressTensor, const VoigtTensor2 &stressIncrement,
                         const bool verbose = FALSE);

    void computeLocalDisplacement(const VecCoord& x, Displacement &localDisp, int i, Index a, Index b);
    void computeDisplacementIncrement(const VecCoord& pos, const VecCoord& lastPos, Displacement &currentDisp, Displacement &lastDisp,
                                      Displacement &dispIncrement, int i, Index a, Index b);

    void computeStressIncrement(int index, int gaussPointIt, const VoigtTensor2 &initialStress, VoigtTensor2 &newStressPoint, const VoigtTensor2 &strainIncrement,
                                double &lambdaIncrement, MechanicalState &pointMechanicalState, const Displacement &lastDisp);

    void computeElasticForce(Eigen::Matrix<double, 12, 1> &internalForces, const VecCoord& x, int index, Index a, Index b);
    void computePlasticForce(Eigen::Matrix<double, 12, 1> &internalForces, const VecCoord& x, int index, Index a, Index b);
    void computePostPlasticForce(Eigen::Matrix<double, 12, 1> &internalForces, const VecCoord& x, int index, Index a, Index b);

    double equivalentStress(const VoigtTensor2 &stressTensor);
    double vonMisesYield(const VoigtTensor2 &stressTensor, const double yieldStress);
    VoigtTensor2 vonMisesGradient(const VoigtTensor2 &stressTensor);
    VoigtTensor4 vonMisesHessian(const VoigtTensor2 &stressTensor, const double yieldStress);

    //*************************************************************** DEBUG ***************************************************************//
    VoigtTensor2 vonMisesGradientFD(const VoigtTensor2 &currentStressTensor, const double increment, const double yieldStress);
    VoigtTensor4 vonMisesHessianFD(const VoigtTensor2 &lastStressTensor, const VoigtTensor2 &currentStressTensor, const double yieldStress);
    //************************************************************************************************************************************//

    // Special implementation for second-order tensor dot product, with the Voigt notation.
    double voigtDotProduct(const VoigtTensor2 &t1, const VoigtTensor2 &t2);

    //Methods called by addForce, addDForce and addKToMatrix when deforming plasticly
    void accumulateNonLinearForce(VecDeriv& f, const VecCoord& x, int i, Index a, Index b);
    void applyNonLinearStiffness(VecDeriv& df, const VecDeriv& dx, int i, Index a, Index b, double fact);
    void updateTangentStiffness(int i, Index a, Index b);


    /**********************************************************/


    const VecElement *_indexedElements;

    Data<Real> _poissonRatio;
    Data<Real> _youngModulus;
    Data<Real> _yieldStress;
    Data<Real> _zSection;
    Data<Real> _ySection;
    Data< bool> _useSymmetricAssembly;
    Data<bool> _isTimoshenko;

    double lastUpdatedStep;

    container::StiffnessContainer* stiffnessContainer;
    container::PoissonContainer* poissonContainer;

    defaulttype::Quat& beamQuat(int i)
    {
        helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
        return bd[i].quat;
    }
    sofa::core::topology::BaseMeshTopology* _topology;
    BeamFFEdgeHandler* edgeHandler;

    MultiBeamForceField();
    MultiBeamForceField(Real poissonRatio, Real youngModulus, Real yieldStress, Real zSection, Real ySection, bool useVD,
                        bool isPlasticMuller, bool isTimoshenko, bool isPlasticKrabbenhoft, bool isPerfectlyPlastic,
                        helper::vector<defaulttype::Quat> localOrientations);
    virtual ~MultiBeamForceField();

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

#if !defined(SOFA_COMPONENT_FORCEFIELD_MULTIBEAMFORCEFIELD_CPP)
extern template class SOFA_StentExp_API MultiBeamForceField<defaulttype::Rigid3Types>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_MULTIBEAMFORCEFIELD_H
