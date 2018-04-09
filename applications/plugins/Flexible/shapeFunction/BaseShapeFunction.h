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

#ifndef SOFA_CORE_BEHAVIOR_BaseShapeFunction_H
#define SOFA_CORE_BEHAVIOR_BaseShapeFunction_H

#include <Flexible/config.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/MatSym.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/SVector.h>
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
	static const unsigned int spatial_dimensions=ShapeFunctionTypes::spatial_dimensions;

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

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const BaseShapeFunction<ShapeFunctionTypes>* = NULL) { return ShapeFunctionTypes::Name(); }

    BaseMechanicalState* _state;

    virtual void init()
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
    virtual void fillWithMeshQuery( sofa::helper::vector< VRef >& /*index*/, sofa::helper::vector< VReal >& /*w*/,
                                    sofa::helper::vector< VGradient >& /*dw*/, sofa::helper::vector< VHessian >& /*ddw */){std::cout << SOFA_CLASS_METHOD << " : Do nothing" << std::endl;}

    virtual void fillWithGaussQuery( sofa::helper::vector< VRef >& /*index*/, sofa::helper::vector< VReal >& /*w*/,
                                     sofa::helper::vector< VGradient >& /*dw*/, sofa::helper::vector< VHessian >& /*ddw */){std::cout << SOFA_CLASS_METHOD << " : Do nothing" << std::endl;}

    /// interpolate shape function values (and their first and second derivatives) at a given child position
    /// 'cell' might be used to target a specific element/voxel in case of overlapping elements/voxels.
    /// this function is typically used for collision and visual points
    virtual void computeShapeFunction(const Coord& childPosition, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL, const Cell cell=-1)=0;

    /// wrappers
    virtual void computeShapeFunction(const VCoord& childPosition, VecVRef& ref, VecVReal& w, VecVGradient& dw,VecVHessian& ddw)
    {
		unsigned int nb=childPosition.size();
        ref.resize(nb);        w.resize(nb);   dw.resize(nb);  ddw.resize(nb);
        for(unsigned i=0; i<nb; i++)            computeShapeFunction(childPosition[i],ref[i],w[i],&dw[i],&ddw[i]);
	}

    virtual void computeShapeFunction(const VCoord& childPosition, VecVRef& ref, VecVReal& w, VecVGradient& dw,VecVHessian& ddw,  const VCell& cells)
    {
        unsigned int nb=childPosition.size();
        ref.resize(nb);        w.resize(nb);   dw.resize(nb);  ddw.resize(nb);
        for(unsigned i=0; i<nb; i++)            computeShapeFunction(childPosition[i],ref[i],w[i],&dw[i],&ddw[i],cells[i]);
    }

    /// used to make a partition of unity: $sum_i w_i(x)=1$ and adjust derivatives accordingly
    void normalize(VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL)
    {
        unsigned int nbRef=w.size();
        Real sum_w=0;
        Gradient sum_dw;
        Hessian sum_ddw;

        // Compute norm
        for (unsigned int j = 0; j < nbRef; j++) sum_w += w[j];
        if(dw)
        {
            for (unsigned int j = 0; j < nbRef; j++) sum_dw += (*dw)[j];
            if(ddw) for (unsigned int j = 0; j < nbRef; j++) sum_ddw += (*ddw)[j];
        }

        // Normalize
        if(sum_w)
            for (unsigned int j = 0; j < nbRef; j++)
            {
                Real wn=w[j]/sum_w;
                if(dw)
                {
                    Gradient dwn=((*dw)[j] - sum_dw*wn)/sum_w;
                    if(ddw) for(int o=0; o<Hessian::nbLines; o++) for(int p=0; p<Hessian::nbCols; p++) (*ddw)[j](o,p)=((*ddw)[j](o,p) - wn*sum_ddw(o,p) - sum_dw[o]*dwn[p] - sum_dw[p]*dwn[o])/sum_w;
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

    virtual ~BaseShapeFunction() {}

};


template <int spatial_dimensions_, class Real_>
struct ShapeFunctionTypes
{
    typedef Real_ Real;
    typedef helper::vector<unsigned int> VRef;
    typedef helper::vector<Real> VReal;
    typedef defaulttype::Vec<spatial_dimensions_,Real> Coord;                          ///< Spatial coordinates in world space
    typedef helper::vector<Coord> VCoord;
    typedef defaulttype::Vec<spatial_dimensions_,Real> Gradient;                       ///< Gradient of a scalar value in world space
    typedef helper::vector<Gradient> VGradient;
    typedef defaulttype::Mat<spatial_dimensions_,spatial_dimensions_,Real> Hessian;    ///< Hessian (second derivative) of a scalar value in world space
    typedef helper::vector<Hessian> VHessian;
	typedef int Cell;
    typedef helper::vector<Cell> VCell;

    typedef helper::vector< helper::SVector<unsigned int> > VecVRef;
    typedef helper::vector< helper::SVector<Real> > VecVReal;
    typedef helper::vector< helper::SVector<Gradient> > VecVGradient;
    typedef helper::vector< helper::SVector<Hessian> > VecVHessian;

    static const int spatial_dimensions=spatial_dimensions_ ;
    static const char* Name();
};

#ifndef SOFA_FLOAT
typedef ShapeFunctionTypes<3,double> ShapeFunctiond;
typedef ShapeFunctionTypes<2,double> ShapeFunction2d;
template<> inline const char* ShapeFunctiond::Name() { return "ShapeFunctiond"; }
template<> inline const char* ShapeFunction2d::Name() { return "ShapeFunction2d"; }
#endif
#ifndef SOFA_DOUBLE
typedef ShapeFunctionTypes<3,float>  ShapeFunctionf;
typedef ShapeFunctionTypes<2,float>  ShapeFunction2f;
template<> inline const char* ShapeFunctionf::Name() { return "ShapeFunctionf"; }
template<> inline const char* ShapeFunction2f::Name() { return "ShapeFunction2f"; }
#endif

#ifdef SOFA_FLOAT
typedef ShapeFunctionf ShapeFunction;
typedef ShapeFunction2f ShapeFunction2;
#else
typedef ShapeFunctiond ShapeFunction;
typedef ShapeFunction2d ShapeFunction2;
#endif


}
}
}


#endif
