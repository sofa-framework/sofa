"""
Benchmark of SparseLDLSolver: factorization on 3x3 blocks vs factorization on scalars.

A beam made of hexahedra is simulated with SparseLDLSolver, either with the template
CompressedRowSparseMatrixMat3x3d (the factorization is performed on 3x3 blocks) or with the
template CompressedRowSparseMatrixd (the factorization is performed on scalars). The beam is
refined to vary the number of elements. For each size, the script reports the time per step of
both templates, the speedup, and the difference between the final positions.

Usage:
    python SparseLDLSolver_blocks_vs_scalar.py [--sizes 4 6 8 10 12] [--steps 10]

    The beam has size x size x 5*size nodes, i.e. (size-1)^2 * (5*size-1) hexahedra.
    It can also be opened in runSofa (single simulation). In this case, the size and the template
    are read from the environment variables 'size' (default 8) and 'template' (default blocks).

Note: build in Release or RelWithDebInfo for meaningful timings.
"""

import argparse
import os
import time

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation

SCALAR_TEMPLATE = "CompressedRowSparseMatrixd"
BLOCKS_TEMPLATE = "CompressedRowSparseMatrixMat3x3d"


def createBeam(root_node, size, template):
    """Beam of size x size x 5*size nodes, fixed at one end, falling under gravity"""
    root_node.dt = 0.02
    root_node.gravity = [0, -10, 0]

    root_node.addObject('RequiredPlugin', pluginName=[
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.Engine.Select',
        'Sofa.Component.IntegrationScheme.Backward',
        'Sofa.Component.LinearSolver.Direct',
        'Sofa.Component.LinearSolver.Ordering',
        'Sofa.Component.Mass',
        'Sofa.Component.SolidMechanics.FEM.Elastic',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Grid',
    ])

    root_node.addObject('DefaultAnimationLoop')

    beam = root_node.addChild('beam')
    beam.addObject('EulerImplicitIntegrationScheme', rayleighStiffness=0.1, rayleighMass=0.1)
    beam.addObject('AMDOrderingMethod')
    beam.addObject('SparseLDLSolver', name='solver', template=template)
    beam.addObject('MechanicalObject', name='DoFs')
    beam.addObject('UniformMass', totalMass=320)
    beam.addObject('RegularGridTopology', name='grid', nx=size, ny=size, nz=5 * size,
                   xmin=0, xmax=3, ymin=0, ymax=3, zmin=0, zmax=19)
    beam.addObject('BoxROI', name='box', box=[-1, -1, -0.0001, 4, 4, 0.0001])
    beam.addObject('FixedProjectiveConstraint', indices='@box.indices')
    beam.addObject('HexahedronFEMForceField', youngModulus=4000, poissonRatio=0.3, method='large')


def createScene(root_node):
    size = int(os.getenv('size', 8))
    template = os.getenv('template', BLOCKS_TEMPLATE)
    createBeam(root_node, size, template)
    return root_node


def simulate(size, template, nb_steps):
    """Run nb_steps steps. Return the time of the first step, the mean time of the following steps,
    the final positions and the number of non-zero values in the factor L"""
    root = Sofa.Core.Node('root')
    createBeam(root, size, template)
    Sofa.Simulation.initRoot(root, False)
    dt = root.dt.value

    # The first step includes the symbolic factorization, which is reused if the pattern
    # of the matrix does not change
    start = time.perf_counter()
    Sofa.Simulation.animate(root, dt)
    first_step = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(nb_steps - 1):
        Sofa.Simulation.animate(root, dt)
    other_steps = (time.perf_counter() - start) / max(nb_steps - 1, 1)

    positions = np.array(root.beam.DoFs.position.value, copy=True)
    L_nnz = root.beam.solver.L_nnz.value

    Sofa.Simulation.unload(root)
    return first_step, other_steps, positions, L_nnz


def main():
    parser = argparse.ArgumentParser(description='SparseLDLSolver: blocks vs scalar')
    parser.add_argument('--sizes', type=int, nargs='+', default=[4, 6, 8, 10, 12],
                        help='number of nodes along x and y (5*size along z)')
    parser.add_argument('--steps', type=int, default=10, help='number of time steps per simulation')
    args = parser.parse_args()

    header = (f"{'elements':>9} {'DOFs':>7} | {'scalar (ms/step)':>16} {'blocks (ms/step)':>16} {'speedup':>8} | "
              f"{'1st step scalar':>15} {'1st step blocks':>15} | {'L_nnz scalar':>12} {'L_nnz blocks':>12} | {'rel. diff':>9}")
    print(header)
    print('-' * len(header))

    for size in args.sizes:
        nb_elements = (size - 1) ** 2 * (5 * size - 1)
        nb_dofs = 3 * size * size * 5 * size

        first_s, step_s, x_s, nnz_s = simulate(size, SCALAR_TEMPLATE, args.steps)
        first_b, step_b, x_b, nnz_b = simulate(size, BLOCKS_TEMPLATE, args.steps)

        scale = np.max(np.abs(x_s))
        rel_diff = np.max(np.abs(x_s - x_b)) / scale if scale > 0 else 0.0

        print(f"{nb_elements:9d} {nb_dofs:7d} | {1e3 * step_s:16.2f} {1e3 * step_b:16.2f} {step_s / step_b:7.2f}x | "
              f"{1e3 * first_s:13.1f}ms {1e3 * first_b:13.1f}ms | {nnz_s:12d} {nnz_b:12d} | {rel_diff:9.1e}", flush=True)


if __name__ == '__main__':
    main()
