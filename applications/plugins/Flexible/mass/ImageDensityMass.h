#ifndef SOFA_FLEXIBLE_ImageDensityMass_H
#define SOFA_FLEXIBLE_ImageDensityMass_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/State.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>

#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

#include "../shapeFunction/BaseShapeFunction.h"
#include "../deformationMapping/LinearJacobianBlock_point.inl"
#include "../deformationMapping/LinearJacobianBlock_rigid.inl"
#include "../deformationMapping/LinearJacobianBlock_affine.inl"
#include "../deformationMapping/LinearJacobianBlock_quadratic.inl"

#include <image/ImageTypes.h>



namespace sofa
{



namespace component
{

namespace mass
{

/**
* Compute mass matrices based on a density map
* Mass is defined as a global matrix (including non diagonal terms)
* The interpolation weights are given by a BaseShapeFunction component present in the scene
* @warning the interpolation is done by a LinearJacobianBlock hard-coded in this component
* @todo find a way to describe the mass interpolation as a sofa graph with regular mappings
*
* @author Matthieu Nesme
*
*/
template <class DataTypes,class ShapeFunctionTypes,class MassType>
class ImageDensityMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(ImageDensityMass,DataTypes,ShapeFunctionTypes,MassType), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;


    /** @name Shape function stuff */
    //@{
    typedef core::behavior::BaseShapeFunction<ShapeFunctionTypes> BaseShapeFunction;
    typedef typename BaseShapeFunction::VReal VReal;
    typedef typename BaseShapeFunction::Gradient Gradient;
    typedef typename BaseShapeFunction::Hessian Hessian;
    typedef typename BaseShapeFunction::VRef VRef;
    typedef typename BaseShapeFunction::Coord sCoord; ///< spatial coordinates
    BaseShapeFunction* m_shapeFunction;        ///< the component where the weights are computed
    //@}

    /** @name Interpolation stuff */
    //@{
    typedef defaulttype::LinearJacobianBlock<DataTypes,defaulttype::Vec3Types > LinearJacobianBlock;
    typedef helper::vector<LinearJacobianBlock> VecLinearJacobianBlock;
    //@}


    /** @name Image map stuff */
    //@{
#ifndef SOFA_FLOAT
    Data< defaulttype::ImageD > f_densityImage; ///< the density map
#else
	Data< defaulttype::ImageF > f_densityImage; ///< A density map (ratio kg/dm^3)
#endif 

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    Data< TransformType > f_transform;   ///< transform of the density map
    //@}


    /** @name Mass stuff */
    //@{

    typedef linearsolver::CompressedRowSparseMatrix<MassType> MassMatrix; ///< the global mass matrix type
    MassMatrix m_massMatrix; ///< the global mass matrix

    enum { NO_LUMPING=0, BLOCK_LUMPING=1, DIAGONAL_LUMPING=2 };
    Data< int > f_lumping; ///< is the mass matrix lumped? (copy each non-diagonal term on the diagonal term of the same line)  0->no, 1->by bloc, 2->diagonal matrix

    //@}

    Data< bool > f_printMassMatrix; ///< Should the mass matrix be print in console after being precomputed?

    Real m_totalMass, m_totalVolume; ///< for debug purpose

protected:

    ImageDensityMass()
        : m_shapeFunction(NULL)
        , f_densityImage( initData(&f_densityImage, "densityImage", "A density map (ratio kg/dm^3)") )
        , f_transform( initData( &f_transform, TransformType(), "transform", "The density map transform" ) )
        , f_lumping( initData( &f_lumping, (int)NO_LUMPING, "lumping", "Should the mass matrix be lumped? 0->no, 1->by bloc, 2->diagonal matrix" ) )
        , f_printMassMatrix( initData( &f_printMassMatrix, false, "printMassMatrix", "Should the mass matrix be print in console after being precomputed?" ) )
    {}

	// note: (max) this is useless
    virtual ~ImageDensityMass()
    {
    }

	SReal getDampingRatio() { return 0; }

    /// \returns a pointer to the dof rest position
    virtual const VecCoord* getX0();

public:


    void clear();

    virtual void reinit();
    virtual void init();



    // -- Mass interface
    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

    void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f);

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

    SReal getKineticEnergy(const core::MechanicalParams* mparams, const DataVecDeriv& v) const;  ///< vMv/2 using dof->getV()

    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const;   ///< Mgx potential in a uniform gravity field, null at origin

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v);

    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix);


    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;


    bool isDiagonal() { return false; }

    void draw(const core::visual::VisualParams* vparams);


    static std::string templateName(const ImageDensityMass<DataTypes, ShapeFunctionTypes, MassType>* = NULL)
    {
        return DataTypes::Name()+std::string(",")+ShapeFunctionTypes::Name()/*+","+MassType::Name()*/;
    }

    /// \returns the volume of one voxel in 3D or try to approximate the surface of the pixel in 2D
    Real getVoxelVolume( const TransformType& transform ) const;

protected:

    /// \returns the cross contribution (J1^T.voxelMass.J0) to the dof mass
    /// notNull is set to true iff one entry of the returned matrix is not null
    MassType J1tmJ0( /*const*/ LinearJacobianBlock& J0, /*const*/ LinearJacobianBlock& J1, Real voxelMass, bool& notNull );
};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_FLEXIBLE_ImageDensityMass_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API ImageDensityMass<defaulttype::Vec3dTypes,core::behavior::ShapeFunctiond,defaulttype::Mat3x3d>; // volume FEM (tetra, hexa)
//extern template class SOFA_Flexible_API ImageDensityMass<defaulttype::Rigid3dTypes,core::behavior::ShapeFunctiond,defaulttype::Rigid3dMass>; // rigid frame
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API ImageDensityMass<defaulttype::Vec3fTypes,core::behavior::ShapeFunctionf,defaulttype::Mat3x3f>;
//extern template class SOFA_Flexible_API ImageDensityMass<defaulttype::Rigid3fTypes,core::behavior::ShapeFunctionf,defaulttype::Rigid3fMass>; // rigid frame
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif // SOFA_FLEXIBLE_ImageDensityMass_H
