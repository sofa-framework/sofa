/******************************************************************************/
/*  Template library implementing Gaussian quadrature procedures for simple   */
/*                            1D to 3D scenarios.                             */
/*                                                                            */
/* This library is composed of two header files (quadrature.h, gaussian.h).   */
/* It was created by OneZeroPlus (http://www.onezeroplus.com) and distributed */
/* under the Boost Software License 1.0 (BSL-1.0), which details are          */
/* presented at : https://www.boost.org/users/license.html. Accordingly, both */
/* quadrature.h and gaussian.h files are subject to the following conditions. */
/*                                                                            */
/* Permission is hereby granted, free of charge, to any person or             */
/* organization obtaining a copy of the software and accompanying             */
/* documentation covered by this license (the "Software") to use, reproduce,  */
/* display, distribute, execute, and transmit the Software, and to prepare    */
/* derivative works of the Software, and to permit third-parties to whom the  */
/* Software is furnished to do so, all subject to the following:              */
/*                                                                            */
/* The copyright notices in the Software and this entire statement, including */
/* the above license grant, this restriction and the following disclaimer,    */
/* must be included in all copies of the Software, in whole or in part, and   */
/* all derivative works of the Software, unless such copies or derivative     */
/* works are solely in the form of machine-executable object code generated   */
/* by a source language processor.                                            */
/*                                                                            */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS    */
/* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF                 */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND               */
/* NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE        */
/* DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY,    */
/* WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN         */
/* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */
/******************************************************************************/

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
