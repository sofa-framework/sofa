#ifndef SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/FittedRegularGridTopology.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using sofa::helper::vector;

/** Compute Finite Element forces based on 6D beam elements.
*/
template<class DataTypes>
class BeamFEMForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef topology::MeshTopology::index_type Index;
    typedef topology::MeshTopology::Line Element;
    typedef topology::MeshTopology::SeqLines VecElement;


protected:
    //component::MechanicalObject<DataTypes>* object;

    typedef Vec<12, Real> Displacement;        ///< the displacement vector

    typedef Mat<6, 6, Real> MaterialStiffness;    ///< the matrix of material stiffness
    typedef vector<MaterialStiffness> VecMaterialStiffness;         ///< a vector of material stiffness matrices
    //VecMaterialStiffness _materialsStiffnesses;                    ///< the material stiffness matrices vector

    //typedef Mat<12, 6, Real> StrainDisplacement;    ///< the strain-displacement matrix
    //typedef vector<StrainDisplacement> VecStrainDisplacement;        ///< a vector of strain-displacement matrices
    //VecStrainDisplacement _strainDisplacements;                       ///< the strain-displacement matrices vector

    typedef Mat<3, 3, Real> Transformation; ///< matrix for rigid transformations like rotations


    typedef Mat<12, 12, Real> StiffnessMatrix;
    typedef vector<StiffnessMatrix> VecStiffnessMatrixs;         ///< a vector of stiffness matrices
    VecStiffnessMatrixs _stiffnessMatrices;                    ///< the material stiffness matrices vector
    //typedef typename matrix<Real,rectangle<>,compressed<>,row_major >::type CompressedMatrix;
    //CompressedMatrix *_stiffnesses;


    typedef std::pair<int,Real> Col_Value;
    typedef vector< Col_Value > CompressedValue;
    typedef vector< CompressedValue > CompressedMatrix;
    CompressedMatrix _stiffnesses;

    //just for draw forces
    VecDeriv _forces;

    topology::MeshTopology* _mesh;
    topology::FittedRegularGridTopology* _trimgrid;
    const VecElement *_indexedElements;
    Data< VecCoord > _initialPoints; ///< the intial positions of the points
    int _method; ///< the computation method of the displacements
    Data<Real> _poissonRatio;
    Data<Real> _youngModulus;
    Data<bool> _timoshenko;
    Data<Real> _radius;
    bool _updateStiffnessMatrix;
    bool _assembling;

    double _E; //Young
    double _nu;//Poisson
    double _L; //length
    double _r; //radius of the section
    double _G; //shear modulus
    double _Iy;
    double _Iz; //Iz is the cross-section moment of inertia (assuming mass ratio = 1) about the z axis;
    double _J;  //Polar moment of inertia (J = Iy + Iz)
    double _A; // A is the cross-sectional area;
    double _Asy; //_Asy is the y-direction effective shear area =  10/9 (for solid circular section) or 0 for a non-Timoshenko beam
    double _Asz; //_Asz is the z-direction effective shear area;

public:
    BeamFEMForceField()
        : _mesh(NULL), _trimgrid(NULL)
        , _indexedElements(NULL)
        , _initialPoints(initData(&_initialPoints, "initialPoints", "Initial Position"))
        , _method(0)
        , _poissonRatio(initData(&_poissonRatio,(Real)0.49f,"poissonRatio","Potion Ratio"))
        , _youngModulus(initData(&_youngModulus,(Real)5000,"youngModulus","Young Modulus"))
        , _timoshenko(initData(&_timoshenko,true,"timoshenko","use Timoshenko beam (non-null section shear area)"))
        , _radius(initData(&_radius,(Real)0.1,"radius","radius of the section"))
        , _updateStiffnessMatrix(true)
        , _assembling(false)
    {
    }

    void setUpdateStiffnessMatrix(bool val) { this->_updateStiffnessMatrix = val; }

    void setComputeGlobalMatrix(bool val) { this->_assembling= val; }

//    component::MechanicalObject<DataTypes>* getObject() { return object; }

    virtual void init();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord&) { return 0; }


    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

protected:

    //void computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c, Coord d );
    Real peudo_determinant_for_coef ( const Mat<2, 3, Real>&  M );

    //void computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacement &J, const Transformation& Rot );

    //void computeMaterialStiffness(int i, Index&a, Index&b);
    void computeStiffness(int i, Index a, Index b);

    //void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J );

////////////// large displacements method
    //vector<fixed_array<Coord,4> > _rotatedInitialElements;   ///< The initials positions in its frame
    VecReal _initialLength;
    vector<Transformation> _rotations;
    vector<Transformation> _nodeRotations;
    void initLarge(int i, Index a, Index b);
    //void computeRotationLarge( Transformation &r, const Vector &p, Index a, Index b);
    void accumulateForceLarge( VecDeriv& f, const VecCoord& x, int i, Index a, Index b);
    //void accumulateDampingLarge( Vector& f, Index elementIndex );
    void applyStiffnessLarge( VecDeriv& f, const VecDeriv& x, int i, Index a, Index b );

    /*
        Mat3x3d MatrixFromEulerXYZ(double thetaX, double thetaY, double thetaZ)
        {
            double cosX = cos(thetaX);
            double sinX = sin(thetaX);
            double cosY = cos(thetaY);
            double sinY = sin(thetaY);
            double cosZ = cos(thetaZ);
            double sinZ = sin(thetaZ);
            return
                Mat3x3d(Vec3d( cosZ, -sinZ,     0),
                        Vec3d( sinZ,  cosZ,     0),
                        Vec3d(    0,     0,     1)) *
                Mat3x3d(Vec3d( cosY,     0,  sinY),
                        Vec3d(    0,     1,     0),
                        Vec3d(-sinY,     0,  cosY)) *
                Mat3x3d(Vec3d(    1,     0,     0),
                        Vec3d(    0,  cosX, -sinX),
                        Vec3d(    0,  sinX,  cosX)) ;
        }
    */
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
