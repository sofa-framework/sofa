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
#include <vector>

#define OZP_QUADRATURE_VERSION_MAJOR 0
#define OZP_QUADRATURE_VERSION_MINOR 3
#define OZP_QUADRATURE_VERSION_SUBMINOR 0

namespace ozp
{ 
    
namespace quadrature
{

namespace detail 
{

template <unsigned int N> struct Interval {};
template <> struct Interval < 1 > { double a1; double b1; };
template <> struct Interval < 2 > { double a1; double b1; double a2; double b2; };
template <> struct Interval < 3 > { double a1; double b1; double a2; double b2; double a3; double b3; };

inline double change_point(double a, double b, double x) { return 0.5 * ((b - a)*x + a + b); }
inline double change_weight(double a, double b, double w) { return 0.5 * (b - a)*w ; }

template <typename LambdaType> void change_interval(double a1, double b1, double x, double w1, LambdaType fun)
{
    fun(change_point(a1, b1, x), change_weight(a1, b1, w1));
}

template <typename LambdaType> void change_interval(double a1, double a2, double b1, double b2, double x, double y, double w1, double w2, LambdaType fun)
{
    fun(change_point(a1, b1, x), change_point(a2, b2, y), change_weight(a1, b1, w1), change_weight(a2, b2, w2));
}

template <typename LambdaType> void change_interval(double a1, double a2, double a3, double b1, double b2, double b3, double x, double y, double z, double w1, double w2, double w3, LambdaType fun)
{
    fun(change_point(a1, b1, x), change_point(a2, b2, y), change_point(a3, b3, z), change_weight(a1, b1, w1), change_weight(a2, b2, w2), change_weight(a3, b3, w3));
}

template <typename Quadrature, unsigned int Dim> struct QuadratureHelper {};

template <typename Quadrature> struct QuadratureHelper<Quadrature, 1>
{
    template <typename LambdaType> void integrate_interval(const Quadrature& q, LambdaType fun, const detail::Interval<1>& interval)
    {
        for (unsigned int i = 0; i < q.n(); ++i) change_interval(interval.a1, interval.b1, q.points[i], q.weights[i], fun);
    }
    template <typename LambdaType> void integrate(const Quadrature& q, LambdaType fun)
    {
        for (unsigned int i = 0; i < q.n(); ++i) {fun(q.points[i], q.weights[i]);}
    }
};

template <typename Quadrature> struct QuadratureHelper<Quadrature, 2>
{
    template <typename LambdaType> void integrate_interval(const Quadrature& q, LambdaType fun, const detail::Interval<2>& interval)
    {
        const auto n = q.n();
        for (unsigned int i = 0; i < n; ++i) for (unsigned int j = 0; j < n; ++j)
        {
            change_interval(interval.a1, interval.a2, interval.b1, interval.b2, q.points[i], q.points[j], q.weights[i], q.weights[j], fun);
        }
    }
    template <typename LambdaType> void integrate(const Quadrature& q, LambdaType fun)
    {
        const auto n = q.n();
        for (unsigned int i = 0; i < n; ++i) for (unsigned int j = 0; j < n; ++j)
        {
            fun(q.points[i], q.points[j], q.weights[i], q.weights[j]);
        }
    }
};

template <typename Quadrature> struct QuadratureHelper < Quadrature, 3 >
{
    template <typename LambdaType> void integrate_interval(const Quadrature& q, LambdaType fun, const detail::Interval<3>& interval)
    {
        const auto n = q.n();
        for (unsigned int i = 0; i < n; ++i) for (unsigned int j = 0; j < n; ++j) for (unsigned int k = 0; k < n; ++k)
        {
            change_interval(interval.a1, interval.a2, interval.a3, interval.b1, interval.b2, interval.b3, q.points[i], q.points[j], q.points[k], q.weights[i], q.weights[j], q.weights[k], fun);
        }
    }

    template <typename LambdaType> void integrate(const Quadrature& q, LambdaType fun)
    {
        const auto n = q.n();
        for (unsigned int i = 0; i < n; ++i) for (unsigned int j = 0; j < n; ++j) for (unsigned int k = 0; k < n; ++k)
        {
            fun(q.points[i], q.points[j], q.points[k], q.weights[i], q.weights[j], q.weights[k]);
        }
    }
};

} // namespace detail

inline detail::Interval<1> make_interval(double a1, double b1) 
{
    detail::Interval<1> i;
    i.a1 = a1; i.b1 = b1;
    return i;
}

inline detail::Interval<2> make_interval(double a1, double a2, double b1, double b2)
{
    detail::Interval<2> i;
    i.a1 = a1; i.b1 = b1;
    i.a2 = a2; i.b2 = b2;
    return i;
}

inline detail::Interval<3> make_interval(double a1, double a2, double a3, double b1, double b2, double b3)
{
    detail::Interval<3> i;
    i.a1 = a1; i.b1 = b1;
    i.a2 = a2; i.b2 = b2;
    i.a3 = a3; i.b3 = b3;
    return i;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> Quadrature base class. </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
class Quadrature
{
public:
    Quadrature(unsigned int n)  : _n(n)
    {
        this->points.resize(_n);
        this->weights.resize(_n);
    } 
    
    std::vector<double> points;
    std::vector<double> weights;

    unsigned int n() const {return _n;}
    private:
    unsigned int _n;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> Integrates the given function. </summary>
///
/// <typeparam name="QuadratureType"> Type of the quadrature type. </typeparam>
/// <typeparam name="N">              Number of dimentions  </typeparam>
/// <typeparam name="LambdaType">     Type of the integration function. </typeparam>
/// <param name="fun"> the integration function. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename QuadratureType, unsigned int N, typename LambdaType> void integrate(LambdaType fun)
{
    QuadratureType q;
    detail::QuadratureHelper<QuadratureType, N> helper;
    helper.integrate(q, fun);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> Integrates the given function. </summary>
///
/// <typeparam name="QuadratureType"> Type of the quadrature type. </typeparam>
/// <typeparam name="N">              Number of dimentions  </typeparam>
/// <typeparam name="LambdaType">     Type of the integration function. </typeparam>
/// <param name="fun"> the integration function. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename QuadratureType, unsigned int N, typename LambdaType> void integrate(const detail::Interval<N>& interval, LambdaType fun )
{
    QuadratureType q;
    detail::QuadratureHelper<QuadratureType, N> helper;
    helper.integrate_interval(q, fun, interval);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>Integrates the given function with the given quadrature variable </summary>
///
/// <typeparam name="QuadratureType">Type of the quadrature type.</typeparam>
/// <typeparam name="N">             Number of dimentions</typeparam>
/// <typeparam name="LambdaType">    Type of the integration function.</typeparam>
/// <param name="q">  Quadrature .</param>
/// <param name="fun">the integration function.</param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template <unsigned int N, typename QuadratureType, typename LambdaType> void integrate(const QuadratureType& q, LambdaType fun)
{
    detail::QuadratureHelper<QuadratureType, N> helper;
    helper.integrate(q, fun);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>Integrates the given function with the given quadrature variable </summary>
///
/// <typeparam name="QuadratureType">Type of the quadrature type.</typeparam>
/// <typeparam name="N">             Number of dimentions</typeparam>
/// <typeparam name="LambdaType">    Type of the integration function.</typeparam>
/// <param name="q">  Quadrature .</param>
/// <param name="fun">the integration function.</param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template <unsigned int N, typename QuadratureType, typename LambdaType> void integrate(const QuadratureType& q, const detail::Interval<N>& interval, LambdaType fun )
{
    detail::QuadratureHelper<QuadratureType, N> helper;
    helper.integrate_interval(q, fun, interval);
}

} // namespace quadrature

} // namespace ozp