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
#pragma once
#include <limits>
#include <sofa/testing/config.h>

namespace sofa::testing
{

/**
 * @class LinearCongruentialRandomGenerator
 * @brief A simple deterministic and portable random number generator.
 *
 * This class implements a Linear Congruential Generator (LCG) algorithm to generate
 * pseudo-random numbers. It is designed to provide deterministic and portable random
 * number generation, making it well-suited for testing purposes.
 */
class SOFA_TESTING_API LinearCongruentialRandomGenerator
{
    unsigned int m_seed; ///< The current seed value for random number generation.

public:
    explicit LinearCongruentialRandomGenerator(unsigned int initialSeed);

    /**
     * @brief Generates the next pseudo-random number.
     * @return The generated pseudo-random number.
     *
     * This method uses a Linear Congruential Generator (LCG) algorithm to update
     * the seed and produce the next pseudo-random number.
     */
    unsigned int generateRandom();

    /**
     * @brief Generates a pseudo-random value within the unit interval [0, 1].
     *
     * This templated function generates a pseudo-random value of the specified scalar type
     * within the unit interval [0, 1]. It utilizes the underlying random number generator
     * to produce a normalized random value within the unit range.
     *
     * @tparam Scalar The scalar type for the generated value (e.g., float, double).
     * @return A pseudo-random value of the specified scalar type within the range [0, 1].
     *
     * Example usage:
     * @code
     * float randomFloat = generateInUnitRange<float>();
     * double randomDouble = generateInUnitRange<double>();
     * @endcode
     */
    template<class Scalar>
    Scalar generateInUnitRange()
    {
        return static_cast<Scalar>(generateRandom()) / static_cast<Scalar>(std::numeric_limits<unsigned int>::max());
    }

    /**
     * @brief Generates a pseudo-random double value within a specified range.
     *
     * This function generates a pseudo-random double value between the provided
     * minimum (`rmin`) and maximum (`rmax`) values.
     *
     * @param rmin The minimum value of the desired range (inclusive).
     * @param rmax The maximum value of the desired range (inclusive).
     * @return A pseudo-random double value in the specified range [rmin, rmax].
     *
     * Example usage:
     * @code
     * double randomValue = generateInRange(10.0, 20.0);
     * @endcode
     */
    double generateInRange(double rmin, double rmax);

    /**
     * @brief Generates a pseudo-random float value within a specified range.
     *
     * This function generates a pseudo-random float value between the provided
     * minimum (`rmin`) and maximum (`rmax`) values.
     *
     * @param rmin The minimum value of the desired range (inclusive).
     * @param rmax The maximum value of the desired range (inclusive).
     * @return A pseudo-random float value in the specified range [rmin, rmax].
     *
     * Example usage:
     * @code
     * float randomValue = generateInRange(10.f, 20.f);
     * @endcode
     */
    float generateInRange(float rmin, float rmax);
};

}
