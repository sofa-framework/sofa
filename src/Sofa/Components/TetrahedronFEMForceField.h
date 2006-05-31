#ifndef SOFA_COMPONENTS_TETRAHEDRONFEMFORCEFIELD_H
#define SOFA_COMPONENTS_TETRAHEDRONFEMFORCEFIELD_H

#include "Sofa/Core/ForceField.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Components/MeshTopology.h"
#include "Common/Vec.h"
#include "Common/Mat.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template<class DataTypes>
class TetrahedronFEMForceField : public Core::ForceField<DataTypes>, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef MeshTopology::index_type Index;
    typedef MeshTopology::Tetra Element;
    typedef MeshTopology::SeqTetras VecElement;

    static const int SMALL = 0;   ///< Symbol of small displacements tetrahedron solver
    static const int LARGE = 1;   ///< Symbol of large displacements tetrahedron solver
    static const int POLAR = 2;   ///< Symbol of polar displacements tetrahedron solver

protected:
    //Core::MechanicalObject<DataTypes>* object;

    typedef Vec<12, Real> Displacement;		///< the displacement vector

    typedef Mat<6, 6, Real> MaterialStiffness;	///< the matrix of material stiffness
    typedef std::vector<MaterialStiffness> VecMaterialStiffness;         ///< a vector of material stiffness matrices
    VecMaterialStiffness _materialsStiffnesses;					///< the material stiffness matrices vector

    typedef Mat<12, 6, Real> StrainDisplacement;	///< the strain-displacement matrix
    typedef std::vector<StrainDisplacement> VecStrainDisplacement;		///< a vector of strain-displacement matrices
    VecStrainDisplacement _strainDisplacements;					   ///< the strain-displacement matrices vector

    typedef Mat<3, 3, Real> Transformation; ///< matrix for rigid transformations like rotations


    typedef Mat<12, 12, Real> StiffnessMatrix;
    //typedef typename matrix<Real,rectangle<>,compressed<>,row_major >::type CompressedMatrix;
    //CompressedMatrix *_stiffnesses;


    typedef std::pair<int,Real> Col_Value;
    typedef std::vector< Col_Value > CompressedValue;
    typedef std::vector< CompressedValue > CompressedMatrix;
    CompressedMatrix _stiffnesses;

    //just for draw forces
    VecDeriv _forces;

    MeshTopology* _mesh;
    const VecElement *_indexedElements;
    VecCoord _initialPoints; ///< the intial positions of the points
    int _method; ///< the computation method of the displacements
    Real _poissonRatio;
    Real _youngModulus;
    Real _dampingRatio;
    bool _updateStiffnessMatrix;
    bool _assembling;

public:
    TetrahedronFEMForceField(Core::MechanicalObject<DataTypes>* /*object*/=NULL)
        : _mesh(NULL)
        , _indexedElements(NULL)
        , _method(0)
        , _poissonRatio(0)
        , _youngModulus(0)
        , _dampingRatio(0)
        , _updateStiffnessMatrix(true)
        , _assembling(false)
    {
    }

    void setPoissonRatio(Real val) { this->_poissonRatio = val; }

    void setYoungModulus(Real val) { this->_youngModulus = val; }

    void setMethod(int val) { this->_method = val; }

    void setUpdateStiffnessMatrix(bool val) { this->_updateStiffnessMatrix = val; }

    void setComputeGlobalMatrix(bool val) { this->_assembling= val; }

//	Core::MechanicalObject<DataTypes>* getObject() { return object; }

    virtual void init();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecCoord& x, const VecDeriv& v, const VecDeriv& dx);

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

protected:

    void computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c, Coord d );
    Real peudo_determinant_for_coef ( const Mat<2, 3, Real>&  M );

    void computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacement &J, const Transformation& Rot );

    void computeMaterialStiffness(int i, Index&a, Index&b, Index&c, Index&d);

    void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J );

////////////// small displacements method
    void initSmall(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForceSmall( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );
    void accumulateDampingSmall( Vector& f, Index elementIndex );
    void applyStiffnessSmall( Vector& f, Real h, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3  );

////////////// large displacements method
    std::vector<fixed_array<Coord,4> > _rotatedInitialElements;   ///< The initials positions in its frame
    std::vector<Transformation> _rotations;
    void initLarge(int i, Index&a, Index&b, Index&c, Index&d);
    void computeRotationLarge( Transformation &r, const Vector &p, const Index &a, const Index &b, const Index &c);
    void accumulateForceLarge( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );
    void accumulateDampingLarge( Vector& f, Index elementIndex );
    void applyStiffnessLarge( Vector& f, Real h, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3 );

////////////// polar decomposition method
    std::vector<Transformation> _initialTransformation;
    void initPolar(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForcePolar( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );
    void applyStiffnessPolar( Vector& f, Real h, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3  );
};

} // namespace Components

} // namespace Sofa

#endif
