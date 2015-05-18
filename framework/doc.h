/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

// This file is used to generate the main page of the doxygen documentation of SOFA.
// It should not be included by any external code.
#error doc.h is not meant to be included, it should be read only by doxygen.

/** \mainpage SOFA API Documentation
 *
 * You are on the main page of the SOFA API Documentation.  This is the starting
 * point of the documentation of the classes of the framework itself.
 *
 * A general introduction to SOFA is available on
 * https://www.sofa-framework.org/documentation/general-documentation/
 *
 *
 * <h4>Plugins</h4>
 *
 * The doxygen documentations of individual plugins are listed on the <b> \ref sofa_plugins "Plugins" </b> page.
 *
 *
 * <h4>Modules</h4>
 *
 * If you are looking for the documentation of a specific component that does not belong to a plugin, check out the <b> \ref sofa_modules_component_list "Component List" </b> page instead.
 *
 *
 * <h4>Generating this documentation</h4>
 *
 * To generate this documentation from the source of Sofa, create a build
 * directory and enable the SOFA-MISC_DOXYGEN CMake option.  This will create
 * various documentation targets:
 *
 * - Use <i><tt>make doc</tt></i> to generate all the documentation; the main page
 *   (this page) is <tt>doc/SOFA/index.html</tt>;
 *
 * - Use <i><tt>make doc-SOFA</tt></i> to re-generate only the documentation of SOFA
 *   (not the plugins);
 *
 * - Use <i><tt>make doc-Foo</tt></i> to generate the documentation of the Foo plugin, as well
 *   as the documentation of Foo's dependencies;
 *
 * - Use <i><tt>make doc-Foo/fast</tt></i> to re-generate only the documentation of the Foo
 *   plugin, without generating again the documentation for its dependencies.
 *
 */

/** \namespace sofa
 *  \brief Main SOFA namespace
 */

/** \defgroup sofa_framework sofa-framework package
 *  \{
 */

/** \namespace sofa::helper
 *  \brief Utility helper classes that we need, but that are not the core of the plateform.
 */

/** \namespace sofa::helper::io
 *  \brief Image and Mesh I/O.
 */

/** \namespace sofa::helper::gl
 *  \brief GL drawing helper classes, no actual visual models.
 */

/** \namespace sofa::helper::system
 *  \brief OS-specific classes
 */

/** \namespace sofa::helper::system::thread
 *  \brief Portable multithreading helper classes (thread, mutex, ...).
 */

/** \namespace sofa::defaulttype
 *  \brief Default data types for the most common cases (1D/2D/3D vectors, rigid frames).
 *
 *  Users can define other types, but it is best to try to use these when
 *  applicable, as many components are already instanciated with them.
 *  It is however not a requirement (nothing prevents a user to define his own
 *  Vec3 class for instance).
 *
 */

/** \namespace sofa::core
 *  \brief Base standardized classes.
 *
 *  Classes in the root sofa::core namespace define the 'high-level' sofa models.
 *
 */

/** \namespace sofa::core::objectmodel
 *  \brief SOFA Object Model.
 *
 *  Specifies how generic objects and context are handled, as well as the basic
 *  functionnalities of all objects (name, fields, ...).
 */

/** \namespace sofa::core::componentmodel
 *  \brief SOFA Component Model.
 *
 *  A composent is defined as an object with a specific role in a Sofa
 *  simulation.
 *
 */

/** \namespace sofa::core::behavior
 *  \brief Abstract API of components related to the behavior of simulated objects.
 *
 *  Simulated bodies in SOFA are split into components storing the current
 *  state (MechanicalState), handling mass properties (Mass), computing
 *  forces (ForceField) and constraints (Constraint), and managing the
 *  integration algorithm (AnimationLoop, OdeSolver).
 *
 *  Depending on the solvers used, two modes of computations are used :
 *  \li <i>vector</i> mode : computations are done directly in the vectors
 *    stored in MechanicalState (used for explicit schemes or iterative
 *    solvers such as conjugate gradient).
 *  \li <i>matrix</i> mode : matrices corresponding to the mechanical system of
 *    equations are constructed, and then inversed to compute the new state.
 *  Not all components support the matrix mode of computation, as it is rather
 *  new and not yet finalized.
 *
 */

/** \namespace sofa::core::collision
 *  \brief Abstract API of components related to collision handling.
 */

/** \namespace sofa::core::topology
 *  \brief Abstract API of components related to topologies.
 */

/** \} */

/** \defgroup sofa_modules sofa-modules package
 *  \{
 */

/** \namespace sofa::component
 *  \brief Implementation of components.
 */

/** \namespace sofa::simulation
 *  \brief Simulation data structure and scheduling (scene-graph, multithread support).
 */

/** \namespace sofa::simulation::tree
 *  \brief Default implementation of the simulation data structure using a tree.
 * @sa sofa::simulation::graph
 */

/** \namespace sofa::simulation::graph
 *  \brief New implementation of the simulation data structure using a directed acyclic graph.
 * This it necessary for node with multiple parents and MultiMappings.
 */

/** \namespace sofa::simulation::xml
 *  \brief XML I/O classes.
 */

#ifdef SOFA_DEV

/** \namespace sofa::simulation::automatescheduler
 *  \brief Automate-based multithread scheduler.
 */

#endif // SOFA_DEV

/** \} */

/** \defgroup sofa_applications sofa-applications package
 *  \{
 */

/** \namespace sofa::gui
 *  \brief User Interfaces
 */

/** \namespace sofa::gui::qt
 *  \brief Qt-based User Interface.
 */

/** \} */
