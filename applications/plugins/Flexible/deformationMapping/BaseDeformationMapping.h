/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_BaseDeformationMAPPING_H
#define SOFA_COMPONENT_MAPPING_BaseDeformationMAPPING_H

#include <Flexible/config.h>
#include <sofa/core/Mapping.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include <sofa/simulation/Simulation.h>

#include "../shapeFunction/BaseShapeFunction.h"
#include "../quadrature/BaseGaussPointSampler.h"
#include <SofaBaseTopology/TopologyData.inl>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/kdTree.h>

#include <SofaEigen2Solver/EigenSparseMatrix.h>

namespace sofa
{


///Class used to provide material_dimensions and tells if position/defo gradients are mapped or not
template< class OutDataTypes>
class OutDataTypesInfo
{
public:
    enum {material_dimensions = OutDataTypes::spatial_dimensions};
    static const bool positionMapped=true; ///< tells if spatial positions are included in output state
    static const bool FMapped=false;        ///< tells if deformation gradients are included in output state
    static defaulttype::Mat<OutDataTypes::spatial_dimensions,material_dimensions,typename OutDataTypes::Real> getF(const typename OutDataTypes::Coord&)  { return defaulttype::Mat<OutDataTypes::spatial_dimensions,material_dimensions,typename OutDataTypes::Real>(); }
};

template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
class OutDataTypesInfo<defaulttype::DefGradientTypes<_spatial_dimensions, _material_dimensions, _order, _Real> >
{
public:
    enum {material_dimensions = _material_dimensions};
    static const bool positionMapped=false;
    static const bool FMapped=true;
    typedef defaulttype::DefGradientTypes<_spatial_dimensions, _material_dimensions, _order, _Real> inherit;
    static typename inherit::Frame getF(const typename inherit::Deriv & outC)  { return outC.getF(); }
};

template<class TCoord, class TDeriv, class TReal>
class OutDataTypesInfo<defaulttype::StdVectorTypes<TCoord, TDeriv, TReal> >
{
public:
    enum {material_dimensions = TCoord::spatial_dimensions};
    static const bool positionMapped=true;
    static const bool FMapped=false;
    static defaulttype::Mat<TCoord::spatial_dimensions,material_dimensions,TReal> getF(const TCoord&)  { return defaulttype::Mat<TCoord::spatial_dimensions,material_dimensions,TReal>(); }
};

template<class TCoord, class TDeriv, class TReal>
class OutDataTypesInfo<defaulttype::ExtVectorTypes<TCoord, TDeriv, TReal> >
{
public:
    enum {material_dimensions = TCoord::spatial_dimensions};
    static const bool positionMapped=true;
    static const bool FMapped=false;
    static defaulttype::Mat<TCoord::spatial_dimensions,material_dimensions,TReal> getF(const TCoord&)  { return defaulttype::Mat<TCoord::spatial_dimensions,material_dimensions,TReal>(); }
};




namespace component
{
namespace mapping
{


/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class BaseDeformationMappingInternalData
{
public:
};


///Abstract interface to allow forward/backward mapping of arbitrary points (no need to know exact in/output types)
template <int spatial_dimensions,typename Real>
class BasePointMapper : public virtual core::objectmodel::BaseObject
{
protected:
    BasePointMapper() {}
    BasePointMapper(const BasePointMapper& b);
    BasePointMapper& operator=(const BasePointMapper& b);

public:
    typedef defaulttype::Vec<spatial_dimensions,Real> Coord ; ///< spatial coordinates

    virtual void ForwardMapping(Coord& p,const Coord& p0)=0;      ///< returns spatial coord p in deformed configuration corresponding to the rest coord p0
    virtual void BackwardMapping(Coord& p0,const Coord& p,const Real Thresh=1e-5, const size_t NbMaxIt=10)=0;     ///< iteratively approximate spatial coord p0 in rest configuration corresponding to the deformed coord p (warning! p0 need to be initialized in the object first, for instance using closest point matching)
    virtual unsigned int getClosestMappedPoint(const Coord& p, Coord& x0,Coord& x, bool useKdTree=false)=0; ///< returns closest mapped point x from input point p, its rest pos x0, and its index

    virtual void resizeOut(const helper::vector<Coord>& position0, helper::vector<helper::vector<unsigned int> > index,helper::vector<helper::vector<Real> > w, helper::vector<helper::vector<defaulttype::Vec<spatial_dimensions,Real> > > dw, helper::vector<helper::vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > > ddw, helper::vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > F0)=0; /// resizing given custom positions and weights
};


// not templated BaseDeformationMapping for identification
class BaseDeformationMapping
{
protected:

    virtual ~BaseDeformationMapping() {}

public:

    /// \returns the from model size
    virtual size_t getFromSize() const = 0;
    /// \returns the to model size
    virtual size_t getToSize() const = 0;
};




///Abstract mapping (one parent->several children with different influence) using JacobianBlocks or sparse eigen matrix
template <class JacobianBlockType>
class BaseDeformationMappingT : public BaseDeformationMapping, public core::Mapping<typename JacobianBlockType::In,typename JacobianBlockType::Out>, public BasePointMapper<JacobianBlockType::Out::spatial_dimensions,typename JacobianBlockType::In::Real>
{
public:
    typedef core::Mapping<typename JacobianBlockType::In, typename JacobianBlockType::Out> Inherit;
    SOFA_ABSTRACT_CLASS2(SOFA_TEMPLATE(BaseDeformationMappingT,JacobianBlockType), SOFA_TEMPLATE2(core::Mapping,typename JacobianBlockType::In,typename JacobianBlockType::Out), SOFA_TEMPLATE2(BasePointMapper,JacobianBlockType::Out::spatial_dimensions,typename JacobianBlockType::In::Real) );

    /** @name  Input types    */
    //@{
    typedef typename JacobianBlockType::In In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real Real;
    //@}

    /** @name  Output types    */
    //@{
    typedef typename JacobianBlockType::Out Out;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real OutReal;
    enum { spatial_dimensions = Out::spatial_dimensions };
    enum { material_dimensions = OutDataTypesInfo<Out>::material_dimensions };
    //@}

    /** @name  Shape Function types    */
    //@{
    typedef core::behavior::ShapeFunctionTypes<spatial_dimensions,Real> ShapeFunctionType;
    typedef core::behavior::BaseShapeFunction<ShapeFunctionType> BaseShapeFunction;
    typedef typename BaseShapeFunction::VReal VReal;
    typedef typename BaseShapeFunction::VecVReal VecVReal;
    typedef typename BaseShapeFunction::Gradient Gradient;
    typedef typename BaseShapeFunction::VGradient VGradient;
    typedef typename BaseShapeFunction::VecVGradient VecVGradient;
    typedef typename BaseShapeFunction::Hessian Hessian;
    typedef typename BaseShapeFunction::VHessian VHessian;
    typedef typename BaseShapeFunction::VecVHessian VecVHessian;
    typedef typename BaseShapeFunction::VRef VRef;
    typedef typename BaseShapeFunction::VecVRef VecVRef;
    typedef typename BaseShapeFunction::Coord mCoord; ///< material coordinates
    //@}

    /** @name  Coord types    */
    //@{
    typedef defaulttype::Vec<spatial_dimensions,Real> Coord ; ///< spatial coordinates
    typedef helper::vector<Coord> VecCoord;
    typedef defaulttype::Mat<spatial_dimensions,material_dimensions,Real> MaterialToSpatial;     ///< local liner transformation from material space to world space = deformation gradient type
    typedef helper::vector<MaterialToSpatial> VMaterialToSpatial;
    typedef helper::kdTree<Coord> KDT;      ///< kdTree for fast search of closest mapped points
    typedef typename KDT::distanceSet distanceSet;
    //@}

    /** @name  Jacobian types    */
    //@{
    typedef JacobianBlockType BlockType;
    typedef helper::vector<helper::vector<BlockType> >  SparseMatrix;

    typedef typename BlockType::MatBlock  MatBlock;  ///< Jacobian block matrix
    typedef linearsolver::EigenSparseMatrix<In,Out>    SparseMatrixEigen;

    typedef typename BlockType::KBlock  KBlock;  ///< stiffness block matrix
    typedef linearsolver::EigenSparseMatrix<In,In>    SparseKMatrixEigen;
    //@}	

    typedef typename Inherit::ForceMask ForceMask;



    ///@brief Update \see f_index_parentToChild from \see f_index
//    void updateIndex();
//    void updateIndex(const size_t parentSize, const size_t childSize);
    void resizeOut(); /// automatic resizing (of output model and jacobian blocks) when input samples have changed. Recomputes weights from shape function component.
    virtual void resizeOut(const helper::vector<Coord>& position0, helper::vector<helper::vector<unsigned int> > index,helper::vector<helper::vector<Real> > w, helper::vector<helper::vector<defaulttype::Vec<spatial_dimensions,Real> > > dw, helper::vector<helper::vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > > ddw, helper::vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > F0); /// resizing given custom positions and weights

    /*!
     * \brief Resize all required data and initialize jacobian blocks
     * \param p0 parent initial positions
     * \param c0 child initial positions
     * \param x0 child initial positions as Vec3r
     * \param index child to parent index
     * \param w child weights
     * \param dw child weight derivatives
     * \param ddw child weight hessians
     * \param F child initial frame
     */
    virtual void resizeAll(const InVecCoord& p0, const OutVecCoord& c0, const VecCoord& x0, const VecVRef& index, const VecVReal& w, const VecVGradient& dw, const VecVHessian& ddw, const VMaterialToSpatial& F0);

    /** @name Mapping functions */
    //@{
    virtual void init();
    virtual void reinit();

    using Inherit::apply;
    using Inherit::applyJ;
    using Inherit::applyJT;

    virtual void apply(OutVecCoord& out, const InVecCoord& in);
    virtual void apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn);
    virtual void applyJ(OutVecDeriv& out, const InVecDeriv& in);
    virtual void applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn);
    virtual void applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut);
    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId );
    virtual void applyJT(const core::ConstraintParams * /*cparams*/ , Data<InMatrixDeriv>& /*out*/, const Data<OutMatrixDeriv>& /*in*/);

    const defaulttype::BaseMatrix* getJ(const core::MechanicalParams * /*mparams*/);

    // Compliant plugin experimental API
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs();

    virtual void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId );
    virtual const defaulttype::BaseMatrix* getK();

    void draw(const core::visual::VisualParams* vparams);

    //@}

    ///@brief Get parent state size
    virtual size_t getFromSize() const { return this->fromModel->getSize(); }
    ///@brief Get child state size
    virtual size_t getToSize()  const { return this->toModel->getSize(); }
    ///@brief Get child to parent indices as a const reference
    virtual const VecVRef& getChildToParentIndex() { return  f_index.getValue(); }
    ///@brief Get parent indices of the i-th child
        virtual const VRef& getChildToParentIndex( int i) { return  f_index.getValue()[i]; }
    ///@brief Get a structure storing parent to child indices as a const reference
//    ///@see f_index_parentToChild to know how to properly use it
//    virtual const vector<VRef>& getParentToChildIndex() { return f_index_parentToChild; }
    ///@brief Get a pointer to the shape function where the weights are computed
    virtual BaseShapeFunction* getShapeFunction() { return _shapeFunction; }
    ///@brief Get parent's influence weights on each child
    virtual VecVReal getWeights(){ return f_w.getValue(); }
    ///@brief Get parent's influence weights gradient on each child
    virtual VecVGradient getWeightsGradient(){ return f_dw.getValue(); }
    ///@brief Get parent's influence weights hessian on each child
    virtual VecVHessian getWeightsHessian(){ return f_ddw.getValue(); }
    ///@brief Get mapped positions
    VecCoord getMappedPositions() { return f_pos; }
    ///@brief Get init positions
    VecCoord getInitPositions() { return f_pos0.getValue(); }

    /** @name PointMapper functions */
    //@{
    virtual void ForwardMapping(Coord& p,const Coord& p0);
    virtual void BackwardMapping(Coord& p0,const Coord& p,const Real Thresh=1e-5, const size_t NbMaxIt=10);
    virtual unsigned int getClosestMappedPoint(const Coord& p, Coord& x0,Coord& x, bool useKdTree=false);

    virtual void mapPosition(Coord& p,const Coord &p0, const VRef& ref, const VReal& w)=0;
    virtual void mapDeformationGradient(MaterialToSpatial& F, const Coord &p0, const MaterialToSpatial& M, const VRef& ref, const VReal& w, const VGradient& dw)=0;
    virtual void mapDeformationGradientRate(MaterialToSpatial& /*F*/, const Coord &/*p0*/, const MaterialToSpatial& /*M*/, const VRef& /*ref*/, const VReal& /*w*/, const VGradient& /*dw*/)
    {
        std::cout << SOFA_CLASS_METHOD << " : not implemented here, see child classes." << std::endl;
    }

    //@}

    SparseMatrix& getJacobianBlocks();

    void setWeights(const VecVReal& weights, const VecVRef& indices)
    {
        f_index = indices;
        f_w = weights;
    }

    Data<std::string> f_shapeFunction_name; ///< Name of the shape function component (optional: if not specified, will searchup)
    BaseShapeFunction* _shapeFunction;      ///< Where the weights are computed
    engine::BaseGaussPointSampler* _sampler;
    Data<VecVRef > f_index;            ///< Store child to parent relationship. index[i][j] is the index of the j-th parent influencing child i.
//    vector<VRef> f_index_parentToChild;     ///< Store parent to child relationship.
//                                            /**< @warning For each parent i, child index <b>and parent index (again)</b> are stored.
//                                                 @warning Therefore to get access to parent's child index only you have to perform a loop over index[i] with an offset of size 2.
//                                             */
    Data<VecVReal >       f_w;         ///< Influence weights of the parents for each child
    Data<VecVGradient >   f_dw;        ///< Influence weight gradients
    Data<VecVHessian >    f_ddw;       ///< Influence weight hessians
    Data<VMaterialToSpatial>    f_F0;       ///< initial value of deformation gradients
    Data< helper::vector<int> > f_cell;    ///< indices required by shape function in case of overlapping elements


    Data<bool> assemble; ///< Assemble the matrices (Jacobian/Geometric Stiffness) or use optimized Jacobian/vector multiplications

    Data<VecCoord >    f_pos0; ///< initial spatial positions of children
    VecCoord f_pos;
    VMaterialToSpatial f_F;         ///< current value of deformation gradients (for visualisation)
    KDT f_KdTree;

protected:
    BaseDeformationMappingT (core::State<In>* from = NULL, core::State<Out>* to= NULL);
    virtual ~BaseDeformationMappingT() { }

public:

    void mapPositions() ///< map initial spatial positions stored in f_pos0 to f_pos (used for visualization)
    {
        this->f_pos.resize(this->f_pos0.getValue().size());
        for(size_t i=0; i<this->f_pos.size(); i++ ) mapPosition(f_pos[i],this->f_pos0.getValue()[i],this->f_index.getValue()[i],this->f_w.getValue()[i]);
    }

    void mapDeformationGradients() ///< map initial deform  gradients stored in f_F0 to f_F      (used for visualization)
    {
        this->f_F.resize(this->f_pos0.getValue().size());
        for(size_t i=0; i<this->f_F.size(); i++ ) mapDeformationGradient(f_F[i],this->f_pos0.getValue()[i],this->f_F0.getValue()[i],this->f_index.getValue()[i],this->f_w.getValue()[i],this->f_dw.getValue()[i]);
    }

protected :
    bool missingInformationDirty;  ///< tells if pos or F need to be updated (to speed up visualization)
    bool KdTreeDirty;              ///< tells if kdtree need to be updated (to speed up closest point search)

    SparseMatrix jacobian;   ///< Jacobian of the mapping
    virtual void initJacobianBlocks()=0;
    virtual void initJacobianBlocks(const InVecCoord& /*inCoord*/, const OutVecCoord& /*outCoord*/){ std::cout << "Only implemented in LinearMapping for now." << std::endl;}

    SparseMatrixEigen eigenJacobian/*, maskedEigenJacobian*/;  ///< Assembled Jacobian matrix
    helper::vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Vector of jacobian matrices, for the Compliant plugin API
    void updateJ();
//    void updateMaskedJ();
//    size_t previousMaskHash; ///< storing previous dof maskTo to check if it changed from last time step to updateJ in consequence

    SparseKMatrixEigen K;  ///< Assembled geometric stiffness matrix

    const core::topology::BaseMeshTopology::SeqTriangles *triangles; // Used for visualization
    const defaulttype::ResizableExtVector<core::topology::BaseMeshTopology::Triangle> *extTriangles;
    const defaulttype::ResizableExtVector<int> *extvertPosIdx;

    void updateForceMask();

public:

    Data< float > showDeformationGradientScale; ///< Scale for deformation gradient display
    Data< helper::OptionsGroup > showDeformationGradientStyle; ///< Visualization style for deformation gradients
    Data< helper::OptionsGroup > showColorOnTopology; ///< Color mapping method
    Data< float > showColorScale; ///< Color mapping scale
    Data< unsigned > d_geometricStiffness; ///< 0=no GS, 1=non symmetric, 2=symmetrized
    Data< bool > d_parallel;		///< use openmp ?
};


} // namespace mapping
} // namespace component
} // namespace sofa

#endif
