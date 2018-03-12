#pragma once
#include "quadrature.h"

namespace ozp
{ 
    
namespace quadrature
{

    template<unsigned int N> struct Gaussian {};
  
    template<> struct Gaussian<1> : public Quadrature
    {
        Gaussian() : Quadrature(1)
        {
            points[0] = 0.0;
            weights[0] = 2.0;
        }
    };

    template<> struct Gaussian<2> : public Quadrature
     {
         Gaussian() : Quadrature(2)
         {
             points[0] =  0.57735026919;
             points[1] = -0.57735026919;

             weights[0] = 1;
             weights[1] = 1;
         }
    };

  template<> struct Gaussian<3> : public Quadrature
  {
    Gaussian() : Quadrature(3)
    {
      points[0] =  0.77459666924;
      points[1] = -0.77459666924;
      points[2] = 0.0;

      weights[0] = 0.55555555555;
      weights[1] = 0.55555555555;
      weights[2] = 0.88888888888;
    }
  };

  template<> struct Gaussian<4>  : public Quadrature
  {
    Gaussian() : Quadrature(4)
    {
      points[0] = -0.3399810435848563;
      points[1] =  0.3399810435848563;
      points[2] = -0.8611363115940526;
      points[3] =  0.8611363115940526;

      weights[0] = 0.6521451548625461;
      weights[1] = 0.6521451548625461;
      weights[2] = 0.3478548451374538;
      weights[3] = 0.3478548451374538;
    }
  };

  template<> struct Gaussian<5>  : public Quadrature
  {
    Gaussian() : Quadrature(5)
    {
      points[0] =  0.0;
      points[1] = -0.5384693101056831;
      points[2] =  0.5384693101056831;
      points[3] = -0.9061798459386640;
      points[4] =  0.9061798459386640;

      weights[0] = 0.5688888888888889;
      weights[1] = 0.4786286704993665;
      weights[2] = 0.4786286704993665;
      weights[3] = 0.2369268850561891;
      weights[4] = 0.2369268850561891;
    }
  };

  template<> struct Gaussian<6>  : public Quadrature
  {
    Gaussian() : Quadrature(6)
    {
      points[0] =  0.6612093864662645;
      points[1] = -0.6612093864662645;
      points[2] = -0.2386191860831969;
      points[3] =  0.2386191860831969;
      points[4] = -0.9324695142031521;
      points[5] =  0.9324695142031521;

      weights[0] = 0.3607615730481386;
      weights[1] = 0.3607615730481386;
      weights[2] = 0.4679139345726910;
      weights[3] = 0.4679139345726910;
      weights[4] = 0.1713244923791704;
      weights[5] = 0.1713244923791704;
    }
  };


} // namespace quadrature

} // namespace ozp
