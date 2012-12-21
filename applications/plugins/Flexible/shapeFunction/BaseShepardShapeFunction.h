#ifndef BASESHEPARDSHAPEFUNCTION_H
#define BASESHEPARDSHAPEFUNCTION_H

#include <initFlexible.h>
#include <shapeFunction/BaseShapeFunction.h>
#include <limits>

namespace sofa
{
namespace component
{
namespace shapefunction
{

using core::behavior::BaseShapeFunction;
using defaulttype::Mat;
/**
Shepard shape function (=inverse distance weights) is defined as w_i(x)=1/d(x,x_i)^power followed by normalization
http://en.wikipedia.org/wiki/Inverse_distance_weighting
  */

template<typename TShapeFunctionTypes>
struct ShepardShapeFunctionInternalData
{
};


template <class ShapeFunctionTypes_>
class BaseShepardShapeFunction : public BaseShapeFunction<ShapeFunctionTypes_>
{
public:
	SOFA_CLASS(SOFA_TEMPLATE(BaseShepardShapeFunction, ShapeFunctionTypes_) , SOFA_TEMPLATE(BaseShapeFunction, ShapeFunctionTypes_));
	typedef BaseShapeFunction<ShapeFunctionTypes_> Inherit;

	typedef typename Inherit::Real Real;
	typedef typename Inherit::Coord Coord;
	typedef typename Inherit::VCoord VCoord;
	enum {material_dimensions=Inherit::material_dimensions};
	typedef typename Inherit::VReal VReal;
	typedef typename Inherit::VGradient VGradient;
	typedef typename Inherit::VHessian VHessian;
	typedef typename Inherit::VRef VRef;
	typedef typename Inherit::MaterialToSpatial MaterialToSpatial;
	typedef typename Inherit::VMaterialToSpatial VMaterialToSpatial;
	typedef typename Inherit::Hessian Hessian;
	typedef typename Inherit::VecVRef VecVRef;
	typedef typename Inherit::VecVReal VecVReal;
	typedef typename Inherit::VecVGradient VecVGradient;
	typedef typename Inherit::VecVHessian VecVHessian;
	typedef ShepardShapeFunctionInternalData<ShapeFunctionTypes_> InternalData;

public :

	Data<Real> power;

	virtual void init()
	{
		Inherit::init();
	}

	virtual void computeShapeFunction(const VCoord& childPosition, VMaterialToSpatial& M, VecVRef& ref, VecVReal& w, VecVGradient& dw, VecVHessian& ddw)
	{
		Inherit::computeShapeFunction(childPosition, M, ref, w, dw, ddw);
	}

	virtual void computeShapeFunction(const Coord& childPosition, MaterialToSpatial& M, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL)
	{
	}

protected :
	BaseShepardShapeFunction() :
		Inherit(),
		power(initData(&power,(Real)2.0, "power", "power of the inverse distance"))
	{
	}

	~BaseShepardShapeFunction()
	{
	}

};

}
}
}


#endif // BASESHEPARDSHAPEFUNCTION_H
