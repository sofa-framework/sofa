/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_BaseDeformationMAPPING_H
#define SOFA_COMPONENT_MAPPING_BaseDeformationMAPPING_H

#include "../initFlexible.h"
#include <sofa/core/Mapping.inl>
#include <sofa/component/component.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/DeformationGradientTypes.h"
#include <sofa/simulation/common/Simulation.h>

#include "../shapeFunction/BaseShapeFunction.h"
#include <SofaBaseTopology/TopologyData.inl>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/kdTree.inl>

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

using helper::vector;

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
public:
    typedef defaulttype::Vec<spatial_dimensions,Real> Coord ; ///< spatial coordinates

    virtual void ForwardMapping(Coord& p,const Coord& p0)=0;      ///< returns spatial coord p in deformed configuration corresponding to the rest coord p0
    virtual void BackwardMapping(Coord& p0,const Coord& p,const Real Thresh=1e-5, const size_t NbMaxIt=10)=0;     ///< iteratively approximate spatial coord p0 in rest configuration corresponding to the deformed coord p (warning! p0 need to be initialized in the object first, for instance using closest point matching)
    virtual unsigned int getClosestMappedPoint(const Coord& p, Coord& x0,Coord& x, bool useKdTree=false)=0; ///< returns closest mapped point x from input point p, its rest pos x0, and its index

    virtual void resizeOut(const vector<Coord>& position0, vector<vector<unsigned int> > index,vector<vector<Real> > w, vector<vector<defaulttype::Vec<spatial_dimensions,Real> > > dw, vector<vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > > ddw, vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > F0)=0; /// resizing given custom positions and weights
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
    enum { spatial_dimensions = Out::spatial_dimensions };
    enum { material_dimensions = OutDataTypesInfo<Out>::material_dimensions };
    //@}

    /** @name  Shape Function types    */
    //@{
    typedef core::behavior::ShapeFunctionTypes<spatial_dimensions,Real> ShapeFunctionType;
    typedef core::behavior::BaseShapeFunction<ShapeFunctionType> BaseShapeFunction;
    typedef typename BaseShapeFunction::VReal VReal;
    typedef typename BaseShapeFunction::Gradient Gradient;
    typedef typename BaseShapeFunction::VGradient VGradient;
    typedef typename BaseShapeFunction::Hessian Hessian;
    typedef typename BaseShapeFunction::VHessian VHessian;
    typedef typename BaseShapeFunction::VRef VRef;
    typedef typename BaseShapeFunction::MaterialToSpatial MaterialToSpatial ; ///< MaterialToSpatial transformation = deformation gradient type
    typedef typename BaseShapeFunction::VMaterialToSpatial VMaterialToSpatial;
    typedef typename BaseShapeFunction::Coord mCoord; ///< material coordinates
    //@}

    /** @name  Coord types    */
    //@{
    typedef defaulttype::Vec<spatial_dimensions,Real> Coord ; ///< spatial coordinates
    typedef vector<Coord> VecCoord;
    typedef helper::kdTree<Coord> KDT;      ///< kdTree for fast search of closest mapped points
    typedef typename KDT::distanceSet distanceSet;
    //@}

    /** @name  Jacobian types    */
    //@{
    typedef JacobianBlockType BlockType;
    typedef vector<vector<BlockType> >  SparseMatrix;

    typedef typename BlockType::MatBlock  MatBlock;  ///< Jacobian block matrix
    typedef linearsolver::EigenSparseMatrix<In,Out>    SparseMatrixEigen;

    typedef typename BlockType::KBlock  KBlock;  ///< stiffness block matrix
    typedef linearsolver::EigenSparseMatrix<In,In>    SparseKMatrixEigen;
    //@}


    void resizeOut(); /// automatic resizing (of output model and jacobian blocks) when input samples have changed. Recomputes weights from shape function component.
    virtual void resizeOut(const vector<Coord>& position0, vector<vector<unsigned int> > index,vector<vector<Real> > w, vector<vector<defaulttype::Vec<spatial_dimensions,Real> > > dw, vector<vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > > ddw, vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > F0); /// resizing given custom positions and weights

    /** @name Mapping functions */
    //@{
    virtual void init();
    virtual void reinit();

    virtual void apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn);
    virtual void applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn);
    virtual void applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut);
    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId );
    virtual void applyJT(const core::ConstraintParams * /*cparams*/ , Data<InMatrixDeriv>& /*out*/, const Data<OutMatrixDeriv>& /*in*/);

    const defaulttype::BaseMatrix* getJ(const core::MechanicalParams * /*mparams*/)
    {
        if(!this->assemble.getValue() || !BlockType::constant) updateJ();
        else if( this->maskTo && this->maskTo->isInUse() )
        {
            if( previousMask!=this->maskTo->getEntries() )
            {
                previousMask = this->maskTo->getEntries();
                updateJ();
            }
        }
        else if( !eigenJacobian.compressedMatrix.nonZeros() ) updateJ();

        return &eigenJacobian;
    }

    // Compliant plugin experimental API
    virtual const vector<sofa::defaulttype::BaseMatrix*>* getJs()
    {
        if(!this->assemble.getValue() || !BlockType::constant) updateJ();
        else if( this->maskTo && this->maskTo->isInUse() )
        {
            if( previousMask!=this->maskTo->getEntries()  )
            {
                previousMask = this->maskTo->getEntries();
                updateJ();
            }

//            typedef helper::ParticleMask ParticleMask;
//            const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();
//            for (ParticleMask::InternalStorage::const_iterator  it=indices.begin(); it!=indices.end(); it++ )
//                std::cerr<<*it<<" ";
//            std::cerr<<std::endl;
        }
        else if( !eigenJacobian.compressedMatrix.nonZeros() ) updateJ();

        return &baseMatrices;
    }

    virtual const vector<defaulttype::BaseMatrix*>* getKs()
    {
        updateK(this->toModel->readForces().ref());
        return &stiffnessBaseMatrices;
    }

    void draw(const core::visual::VisualParams* vparams);

    //@}

    ///@brief Get parent state size
    virtual size_t getFromSize() const { return this->fromModel->getSize(); }
    ///@brief Get child state size
    virtual size_t getToSize()  const { return this->toModel->getSize(); }
    ///@brief Get child to parent indices as a const reference
    virtual const vector<VRef>& getChildToParentIndex() { return  f_index.getValue(); }
    ///@brief Get a structure storing parent to child indices as a const reference
    ///@see f_index_parentToChild to know how to properly use it
    virtual const vector<VRef>& getParentToChildIndex() { return f_index_parentToChild; }
    ///@brief Get a pointer to the shape function where the weights are computed
    virtual BaseShapeFunction* getShapeFunction() { return _shapeFunction; }
    ///@brief Get parent's influence weights on each child
    virtual vector<VReal> getWeights(){ return f_w.getValue(); }
    ///@brief Get parent's influence weights gradient on each child
    virtual vector<VGradient> getWeightsGradient(){ return f_dw.getValue(); }
    ///@brief Get parent's influence weights hessian on each child
    virtual vector<VHessian> getWeightsHessian(){ return f_ddw.getValue(); }

    /** @name PointMapper functions */
    //@{
    virtual void ForwardMapping(Coord& p,const Coord& p0);
    virtual void BackwardMapping(Coord& p0,const Coord& p,const Real Thresh=1e-5, const size_t NbMaxIt=10);
    virtual unsigned int getClosestMappedPoint(const Coord& p, Coord& x0,Coord& x, bool useKdTree=false);

    virtual void mapPosition(Coord& p,const Coord &p0, const VRef& ref, const VReal& w)=0;
    virtual void mapDeformationGradient(MaterialToSpatial& F, const Coord &p0, const MaterialToSpatial& M, const VRef& ref, const VReal& w, const VGradient& dw)=0;
    //@}

    SparseMatrix& getJacobianBlocks()
    {
        if(!this->assemble.getValue() || !BlockType::constant) updateJ();
        return jacobian;
    }

    void setWeights(const vector<VReal>& weights, const vector<VRef>& indices)
    {
        f_index = indices;
        f_w = weights;
    }

    Data<std::string> f_shapeFunction_name; ///< Name of the shape function component (optional: if not specified, will searchup)
    BaseShapeFunction* _shapeFunction;      ///< Where the weights are computed
    Data<vector<VRef> > f_index;            ///< Store child to parent relationship. index[i][j] is the index of the j-th parent influencing child i.
    vector<VRef> f_index_parentToChild;     ///< Store parent to child relationship.
                                            /**< @warning For each parent i, child index <b>and parent index (again)</b> are stored.
                                                 @warning Therefore to get access to parent's child index only you have to perform a loop over index[i] with an offset of size 2.
                                             */
    Data<vector<VReal> >       f_w;         ///< Influence weights of the parents for each child
    Data<vector<VGradient> >   f_dw;        ///< Influence weight gradients
    Data<vector<VHessian> >    f_ddw;       ///< Influence weight hessians
    Data<VMaterialToSpatial>    f_F0;
    Data< vector<int> > f_cell;    ///< indices required by shape function in case of overlapping elements


    Data<bool> assemble;

protected:
    BaseDeformationMappingT (core::State<In>* from = NULL, core::State<Out>* to= NULL);
    virtual ~BaseDeformationMappingT()     { }

    Data<VecCoord >    f_pos0; ///< initial spatial positions of children

    VecCoord f_pos;
    void mapPositions() ///< map initial spatial positions stored in f_pos0 to f_pos (used for visualization)
    {
        this->f_pos.resize(this->f_pos0.getValue().size());
        for(size_t i=0; i<this->f_pos.size(); i++ ) mapPosition(f_pos[i],this->f_pos0.getValue()[i],this->f_index.getValue()[i],this->f_w.getValue()[i]);
    }
    KDT f_KdTree;

    VMaterialToSpatial f_F;
    void mapDeformationGradients() ///< map initial deform  gradients stored in f_F0 to f_F      (used for visualization)
    {
        this->f_F.resize(this->f_pos0.getValue().size());
        for(size_t i=0; i<this->f_F.size(); i++ ) mapDeformationGradient(f_F[i],this->f_pos0.getValue()[i],this->f_F0.getValue()[i],this->f_index.getValue()[i],this->f_w.getValue()[i],this->f_dw.getValue()[i]);
    }

    bool missingInformationDirty;  ///< tells if pos or F need to be updated (to speed up visualization)
    bool KdTreeDirty;              ///< tells if kdtree need to be updated (to speed up closest point search)

    SparseMatrix jacobian;   ///< Jacobian of the mapping
    virtual void initJacobianBlocks()=0;

//    helper::ParticleMask* maskFrom;  ///< Subset of master DOF, to cull out computations involving null forces or displacements
    helper::ParticleMask* maskTo;    ///< Subset of slave DOF, to cull out computations involving null forces or displacements


    SparseMatrixEigen eigenJacobian;  ///< Assembled Jacobian matrix
    vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Vector of jacobian matrices, for the Compliant plugin API
    bool Jdirty; ///< Does J needs to be updated?
    void updateJ();
    helper::ParticleMask::InternalStorage previousMask; ///< storing previous dof maskTo to check if it changed from last time step to updateJ in consequence (TODO add such a mechanism directly in ParticleMask?)

    SparseKMatrixEigen K;  ///< Assembled geometric stiffness matrix
    vector<defaulttype::BaseMatrix*> stiffnessBaseMatrices;      ///< Vector of geometric stiffness matrices, for the Compliant plugin API
    void updateK(const OutVecDeriv& childForce);

    const core::topology::BaseMeshTopology::SeqTriangles *triangles; // Used for visualization
    const defaulttype::ResizableExtVector<core::topology::BaseMeshTopology::Triangle> *extTriangles;
    const defaulttype::ResizableExtVector<int> *extvertPosIdx;
    Data< float > showDeformationGradientScale;
    Data< helper::OptionsGroup > showDeformationGradientStyle;
    Data< helper::OptionsGroup > showColorOnTopology;
    Data< float > showColorScale;
};


} // namespace mapping
} // namespace component
} // namespace sofa

#endif
