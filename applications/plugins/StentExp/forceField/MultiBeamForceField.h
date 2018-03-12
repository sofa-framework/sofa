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
#include "../config.h"
#include "../initStentExp.h"


#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/TopologyData.h>

#include <SofaEigen2Solver/EigenSparseMatrix.h>


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

protected:

    typedef defaulttype::Vec<12, Real> Displacement;        ///< the displacement vector
    typedef defaulttype::Mat<3, 3, Real> Transformation; ///< matrix for rigid transformations like rotations


    typedef defaulttype::Mat<12, 12, Real> StiffnessMatrix;
    typedef defaulttype::Mat<12, 8, Real> plasticityMatrix; ///< contribution of plasticity to internal forces

    struct BeamInfo
    {
        /*********************************************************************/
        /*                              Plasticity                           */
        /*********************************************************************/

        plasticityMatrix _M_loc;
        StiffnessMatrix _Ke_loc;
        Eigen::Matrix<double, 6, 6> _materialBehaviour;

        /*********************************************************************/

        // 	static const double FLEXIBILITY=1.00000; // was 1.00001
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
        void init(double E, double L, double nu, double zSection, double ySection);
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
                >> bi._k_loc;
            return in;
        }
    };

    //just for draw forces
    VecDeriv _forces;

    topology::EdgeData< sofa::helper::vector<BeamInfo> > beamsData;
    linearsolver::EigenBaseSparseMatrix<typename DataTypes::Real> matS;

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

    /// virtual displacement method, same as in TetrahedronFEMForceField
    Data<bool> _virtualDisplacementMethod;

    void computeVDStiffness(int i, Index a, Index b);
    void computeMaterialBehaviour(int i, Index a, Index b);


    Data<bool> _isPlastic;
    /// Symmetrical 3x3 stensor written as a vector following the Voigt notation
    typedef Eigen::Matrix<double, 6, 1> VoigtTensor;
    typedef Eigen::Matrix<double, 27, 6> elementPlasticStrain; ///< one 6x1 strain tensor for each of the 27 points of integration
    helper::vector<elementPlasticStrain> _VDPlasticStrains;

    Real _VDPlasticYieldThreshold;
    Real _VDPlasticCreep;

    void computePlasticForces(int i, Index a, Index b, const Displacement& totalDisplacement, nodalForces& plasticForces);
    void updatePlasticStrain(int i, Index a, Index b, VoigtTensor& totalStrain, int gaussPointIterator);

    /**************************************************************************/

    const VecElement *_indexedElements;

    Data<Real> _poissonRatio;
    Data<Real> _youngModulus;
    Data<Real> _zSection;
    Data<Real> _ySection;
    Data< VecIndex > _list_segment;
    Data< bool> _useSymmetricAssembly;
    bool _partial_list_segment;

    bool _updateStiffnessMatrix;
    bool _assembling;

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
    MultiBeamForceField(Real poissonRatio, Real youngModulus, Real zSection, Real ySection, bool useVD, bool isPlastic);
    virtual ~MultiBeamForceField();
public:
    void setUpdateStiffnessMatrix(bool val) { this->_updateStiffnessMatrix = val; }

    void setComputeGlobalMatrix(bool val) { this->_assembling= val; }

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

    void setBeam(unsigned int i, double E, double L, double nu, double zSection, double ySection);
    void initBeams(unsigned int size);

protected:

    void drawElement(int i, std::vector< defaulttype::Vector3 >* points, const VecCoord& x);

    Real peudo_determinant_for_coef ( const defaulttype::Mat<2, 3, Real>&  M );

    void computeStiffness(int i, Index a, Index b);

    ////////////// large displacements method
    helper::vector<Transformation> _nodeRotations;
    void initLarge(int i, Index a, Index b);
    void accumulateForceLarge( VecDeriv& f, const VecCoord& x, int i, Index a, Index b);
    void applyStiffnessLarge( VecDeriv& f, const VecDeriv& x, int i, Index a, Index b, double fact=1.0);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_MULTIBEAMFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_StentExp_API MultiBeamForceField<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_StentExp_API MultiBeamForceField<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_MULTIBEAMFORCEFIELD_H
