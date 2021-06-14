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

#ifndef SOFA_CORE_BEHAVIOR_BaseShapeFunction_H
#define SOFA_CORE_BEHAVIOR_BaseShapeFunction_H

#include <Flexible/config.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Mat.h>
#include <sofa/type/MatSym.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>
#include <sofa/type/SVector.h>
#include <sofa/core/behavior/BaseMechanicalState.h>


namespace sofa
{
namespace core
{
namespace behavior
{


template<typename TShapeFunctionTypes>
struct ShapeFunctionInternalData
{
};

///Compute interpolation weights, their spatial derivatives, and a local orthonormal frame to map material to spatial coordinates.
/** Interpolation is defined across space as \f$ x_j = \sum_i w_{ij} x_i \f$
  It is used to map displacements/velocities/forces, but any other quantities in general.
  Shape function \f$ w_{ij}(x_j) \f$ encodes the influence of a parent node at \f$ x_i \f$ an a child node at \f$ x_j \f$.
  For efficiency, child nodes depend at most on nbRef parents.
  In general, it is a partition of unity : \f$ sum_i w_{ij}(x)=1 \f$ everywhere.
  When \f$ w_i(x_j)=0, i!=j \f$ and  \f$ w_i(x_i)=1 \f$, shape functions are called interpolating. Otherwise they are called approximating.
  In first order finite elements, the shape functions are the barycentric coordinates.
  */

template <class TShapeFunctionTypes>
class BaseShapeFunction : virtual public core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(BaseShapeFunction, TShapeFunctionTypes) , objectmodel::BaseObject);

    typedef TShapeFunctionTypes ShapeFunctionTypes;
    typedef typename ShapeFunctionTypes::Real Real;
    static const std::size_t spatial_dimensions=ShapeFunctionTypes::spatial_dimensions;

    /** @name types */
    //@{
	typedef typename ShapeFunctionTypes::VRef VRef;
	typedef typename ShapeFunctionTypes::VReal VReal;
    typedef typename ShapeFunctionTypes::Coord Coord;                          ///< Spatial coordinates in world space
	typedef typename ShapeFunctionTypes::VCoord VCoord;
    typedef typename ShapeFunctionTypes::Gradient Gradient;                       ///< Gradient of a scalar value in world space
	typedef typename ShapeFunctionTypes::VGradient VGradient;
    typedef typename ShapeFunctionTypes::Hessian Hessian;                       ///< Hessian (second derivative) of a scalar value in world space
	typedef typename ShapeFunctionTypes::VHessian VHessian;
	typedef typename ShapeFunctionTypes::Cell Cell;
	typedef typename ShapeFunctionTypes::VCell VCell;

	typedef typename ShapeFunctionTypes::VecVRef VecVRef;
	typedef typename ShapeFunctionTypes::VecVReal VecVReal;
	typedef typename ShapeFunctionTypes::VecVGradient VecVGradient;
	typedef typename ShapeFunctionTypes::VecVHessian VecVHessian;
	typedef ShapeFunctionInternalData<TShapeFunctionTypes> InternalData;
    //@}

    /** @name data */
    //@{
    Data<unsigned int > f_nbRef;      ///< maximum number of parents per child
    Data< VCoord > f_position;  ///< spatial coordinates of the parent nodes
	InternalData m_internalData;
    //@}

    BaseMechanicalState* _state;

    void init() override
    {
        if(!f_position.isSet())   // node positions are not given, so we retrieve them from the local mechanical state
        {
            if(!_state) this->getContext()->get(_state,core::objectmodel::BaseContext::Local);
            if(!_state) { serr<<"state not found"<< sendl; return; }
            else
            {
                helper::WriteOnlyAccessor<Data<VCoord > > pos(this->f_position);
                pos.resize(_state->getSize());
                for(unsigned int i=0; i<pos.size(); ++i)
                {
                    defaulttype::StdVectorTypes<Coord,Coord>::set( pos[i], _state->getPX(i),_state->getPY(i),_state->getPZ(i) );
//                    pos[i]=Coord(_state->getPX(i),_state->getPY(i),_state->getPZ(i));
				}
            }
        }

    }

    //Pierre-Luc : I added these two functions to fill indices, weights and derivatives from an external component. I also wanted to make a difference between gauss points and mesh vertices.
    virtual void fillWithMeshQuery( sofa::type::vector< VRef >& /*index*/, sofa::type::vector< VReal >& /*w*/,
                                    sofa::type::vector< VGradient >& /*dw*/, sofa::type::vector< VHessian >& /*ddw */){std::cout << SOFA_CLASS_METHOD << " : Do nothing" << std::endl;}

    virtual void fillWithGaussQuery( sofa::type::vector< VRef >& /*index*/, sofa::type::vector< VReal >& /*w*/,
                                     sofa::type::vector< VGradient >& /*dw*/, sofa::type::vector< VHessian >& /*ddw */){std::cout << SOFA_CLASS_METHOD << " : Do nothing" << std::endl;}

    /// interpolate shape function values (and their first and second derivatives) at a given child position
    /// 'cell' might be used to target a specific element/voxel in case of overlapping elements/voxels.
    /// this function is typically used for collision and visual points
    virtual void computeShapeFunction(const Coord& childPosition, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL, const Cell cell=-1)=0;

    /// wrappers
    virtual void computeShapeFunction(const VCoord& childPosition, VecVRef& ref, VecVReal& w, VecVGradient& dw,VecVHessian& ddw)
    {
        std::size_t nb=childPosition.size();
        ref.resize(nb);        w.resize(nb);   dw.resize(nb);  ddw.resize(nb);
        for(std::size_t i=0; i<nb; i++)            computeShapeFunction(childPosition[i],ref[i],w[i],&dw[i],&ddw[i]);
	}

    virtual void computeShapeFunction(const VCoord& childPosition, VecVRef& ref, VecVReal& w, VecVGradient& dw,VecVHessian& ddw,  const VCell& cells)
    {
        std::size_t nb=childPosition.size();
        ref.resize(nb);        w.resize(nb);   dw.resize(nb);  ddw.resize(nb);
        for(std::size_t i=0; i<nb; i++)            computeShapeFunction(childPosition[i],ref[i],w[i],&dw[i],&ddw[i],cells[i]);
    }

    /// used to make a partition of unity: $sum_i w_i(x)=1$ and adjust derivatives accordingly
    void normalize(VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL)
    {
        std::size_t nbRef=w.size();
        Real sum_w=0;
        Gradient sum_dw;
        Hessian sum_ddw;

        // Compute norm
        for (std::size_t j = 0; j < nbRef; j++) sum_w += w[j];
        if(dw)
        {
            for (std::size_t j = 0; j < nbRef; j++) sum_dw += (*dw)[j];
            if(ddw) for (std::size_t j = 0; j < nbRef; j++) sum_ddw += (*ddw)[j];
        }

        // Normalize
        if(sum_w)
            for (std::size_t j = 0; j < nbRef; j++)
            {
                Real wn=w[j]/sum_w;
                if(dw)
                {
                    Gradient dwn=((*dw)[j] - sum_dw*wn)/sum_w;
                    if(ddw) for(std::size_t o=0; o<Hessian::nbLines; o++) for(std::size_t p=0; p<Hessian::nbCols; p++) (*ddw)[j](o,p)=((*ddw)[j](o,p) - wn*sum_ddw(o,p) - sum_dw[o]*dwn[p] - sum_dw[p]*dwn[o])/sum_w;
                    (*dw)[j]=dwn;
                }
                w[j]=wn;
            }
    }


protected:
    BaseShapeFunction()
        : f_nbRef(initData(&f_nbRef,(unsigned int)4,"nbRef", "maximum number of parents per child"))
        , f_position(initData(&f_position,"position", "position of parent nodes"))
        , _state( NULL )
    {
    }

    ~BaseShapeFunction() override {}

};


template <std::size_t spatial_dimensions_, class Real_>
struct ShapeFunctionTypes
{
    typedef Real_ Real;
    typedef type::vector<unsigned int> VRef;
    typedef type::vector<Real> VReal;
    typedef type::Vec<spatial_dimensions_,Real> Coord;                          ///< Spatial coordinates in world space
    typedef type::vector<Coord> VCoord;
    typedef type::Vec<spatial_dimensions_,Real> Gradient;                       ///< Gradient of a scalar value in world space
    typedef type::vector<Gradient> VGradient;
    typedef type::Mat<spatial_dimensions_,spatial_dimensions_,Real> Hessian;    ///< Hessian (second derivative) of a scalar value in world space
    typedef type::vector<Hessian> VHessian;
	typedef int Cell;
    typedef type::vector<Cell> VCell;

    typedef type::vector< type::SVector<unsigned int> > VecVRef;
    typedef type::vector< type::SVector<Real> > VecVReal;
    typedef type::vector< type::SVector<Gradient> > VecVGradient;
    typedef type::vector< type::SVector<Hessian> > VecVHessian;

    static const std::size_t spatial_dimensions=spatial_dimensions_ ;
    static const char* Name();
};

typedef ShapeFunctionTypes<3,float>  ShapeFunction3f;
typedef ShapeFunctionTypes<2,float>  ShapeFunction2f;
template<> inline const char* ShapeFunction3f::Name() { return "ShapeFunction3f"; }
template<> inline const char* ShapeFunction2f::Name() { return "ShapeFunction2f"; }

typedef ShapeFunctionTypes<3,double> ShapeFunction3d;
typedef ShapeFunctionTypes<2,double> ShapeFunction2d;
template<> inline const char* ShapeFunction3d::Name() { return "ShapeFunction3d"; }
template<> inline const char* ShapeFunction2d::Name() { return "ShapeFunction2d"; }

typedef ShapeFunctionTypes<3,SReal> ShapeFunction3;
typedef ShapeFunctionTypes<2,SReal> ShapeFunction2;

}
}
}


#endif
