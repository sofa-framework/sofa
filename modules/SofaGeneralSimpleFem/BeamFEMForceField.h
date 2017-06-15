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
#ifndef SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H
#include "config.h"


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
class BeamFEMForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BeamFEMForceField,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

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

    //typedef Mat<6, 6, Real> MaterialStiffness;    ///< the matrix of material stiffness
    //typedef vector<MaterialStiffness> VecMaterialStiffness;         ///< a vector of material stiffness matrices
    //VecMaterialStiffness _materialsStiffnesses;                    ///< the material stiffness matrices vector

    //typedef Mat<12, 6, Real> StrainDisplacement;    ///< the strain-displacement matrix
    //typedef vector<StrainDisplacement> VecStrainDisplacement;        ///< a vector of strain-displacement matrices
    //VecStrainDisplacement _strainDisplacements;                       ///< the strain-displacement matrices vector

    typedef defaulttype::Mat<3, 3, Real> Transformation; ///< matrix for rigid transformations like rotations


    typedef defaulttype::Mat<12, 12, Real> StiffnessMatrix;
    //typedef topology::EdgeData<StiffnessMatrix> VecStiffnessMatrices;         ///< a vector of stiffness matrices
    //VecStiffnessMatrices _stiffnessMatrices;                    ///< the material stiffness matrices vector

    struct BeamInfo
    {
        // 	static const double FLEXIBILITY=1.00000; // was 1.00001
        double _E0,_E; //Young
        double _nu;//Poisson
        double _L; //length
        double _r; //radius of the section
        double _rInner; //inner radius of the section if beam is hollow
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

        //NewMAT::Matrix _Ke;

        defaulttype::Quat quat;

        //void localStiffness();
        void init(double E, double L, double nu, double r, double rInner);
        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const BeamInfo& bi )
        {
            os	<< bi._E0 << " "
                << bi._E << " "
                << bi._nu << " "
                << bi._L << " "
                << bi._r << " "
                << bi._rInner << " "
                << bi._G << " "
                << bi._Iy << " "
                << bi._Iz << " "
                << bi._J << " "
                << bi._A << " "
                << bi._Asy << " "
                << bi._Asz << " "
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
                >> bi._r
                >> bi._rInner
                >> bi._G
                >> bi._Iy
                >> bi._Iz
                >> bi._J
                >> bi._A
                >> bi._Asy
                >> bi._Asz
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
        typedef typename BeamFEMForceField<DataTypes>::BeamInfo BeamInfo;
        BeamFFEdgeHandler(BeamFEMForceField<DataTypes>* ff, topology::EdgeData<sofa::helper::vector<BeamInfo> >* data)
            :topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,sofa::helper::vector<BeamInfo> >(data),ff(ff) {}

        void applyCreateFunction(unsigned int edgeIndex, BeamInfo&,
                                 const core::topology::BaseMeshTopology::Edge& e,
                                 const sofa::helper::vector<unsigned int> &,
                                 const sofa::helper::vector< double > &);

    protected:
        BeamFEMForceField<DataTypes>* ff;

    };



    const VecElement *_indexedElements;
//	unsigned int maxPoints;
//	int _method; ///< the computation method of the displacements
    Data<Real> _poissonRatio;
    Data<Real> _youngModulus;
//	Data<bool> _timoshenko;
    Data<Real> _radius;
    Data<Real> _radiusInner;
    Data< VecIndex > _list_segment;
    Data< bool> _useSymmetricAssembly;
    bool _partial_list_segment;

    bool _updateStiffnessMatrix;
    bool _assembling;

    double lastUpdatedStep;

    container::StiffnessContainer* stiffnessContainer;
//	container::LengthContainer* lengthContainer;
    container::PoissonContainer* poissonContainer;
//	container::RadiusContainer* radiusContainer;

    defaulttype::Quat& beamQuat(int i)
    {
        helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
        return bd[i].quat;
    }
    sofa::core::topology::BaseMeshTopology* _topology;
    BeamFFEdgeHandler* edgeHandler;


    BeamFEMForceField();
    BeamFEMForceField(Real poissonRatio, Real youngModulus, Real radius, Real radiusInner);
    virtual ~BeamFEMForceField();
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

    void setBeam(unsigned int i, double E, double L, double nu, double r, double rInner);
    void initBeams(unsigned int size);

protected:

    void drawElement(int i, std::vector< defaulttype::Vector3 >* points, const VecCoord& x);

    //void computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c, Coord d );
    Real peudo_determinant_for_coef ( const defaulttype::Mat<2, 3, Real>&  M );

    //void computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacement &J, const Transformation& Rot );

    //void computeMaterialStiffness(int i, Index&a, Index&b);
    void computeStiffness(int i, Index a, Index b);

    //void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J );

    ////////////// large displacements method
    //vector<fixed_array<Coord,4> > _rotatedInitialElements;   ///< The initials positions in its frame
    //VecReal _initialLength;
    helper::vector<Transformation> _nodeRotations;
    //vector<Quat> _beamQuat;
    void initLarge(int i, Index a, Index b);
    //void computeRotationLarge( Transformation &r, const Vector &p, Index a, Index b);
    void accumulateForceLarge( VecDeriv& f, const VecCoord& x, int i, Index a, Index b);
    //void accumulateDampingLarge( Vector& f, Index elementIndex );
    void applyStiffnessLarge( VecDeriv& f, const VecDeriv& x, int i, Index a, Index b, double fact=1.0);

    //sofa::helper::vector< sofa::helper::vector <Real> > subMatrix(unsigned int fr, unsigned int lr, unsigned int fc, unsigned int lc);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_SIMPLE_FEM_API BeamFEMForceField<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_SIMPLE_FEM_API BeamFEMForceField<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H
