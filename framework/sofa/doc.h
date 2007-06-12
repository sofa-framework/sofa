#ifndef SOFA_DOC_H

// This file is only used to provide contents to the doxygen-generated documentation
// It should not be included by any external code

// * \version SVN trunk HEAD

/** \mainpage SOFA API Documentation
 *
 * \TODO TODO: Write main intro page...
 *
 * A good starting point to browse this documentation is the <a href="namespaces.html">Namespace List</a>.
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

/** \namespace sofa::core::componentmodel::behavior
 *  \brief Abstract API of components related to the behavior of simulated objects.
 *
 *  Simulated bodies in SOFA are split into components storing the current
 *  state (MechanicalState), handling mass properties (Mass), computing
 *  forces (ForceField) and constraints (Constraint), and managing the
 *  integration algorithm (MasterSolver, OdeSolver).
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

/** \namespace sofa::core::componentmodel::collision
 *  \brief Abstract API of components related to collision handling.
 */

/** \namespace sofa::core::componentmodel::topology
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
 *  \brief Default (and currently only) implementation of the simulation data structure using a tree.
 */

/** \namespace sofa::simulation::tree::xml
 *  \brief XML I/O classes.
 */

/** \namespace sofa::simulation::automatescheduler
 *  \brief Automate-based multithread scheduler.
 */

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

/** \namespace sofa::gui::fltk
 *  \brief FLTK-based User Interface.
 */

/** \} */

#endif
