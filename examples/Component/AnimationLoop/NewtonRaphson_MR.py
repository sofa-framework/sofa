"""
Python version of NewtonRaphson_MR.scn that tracks the tip node and plots its
position and velocity over time.

Standalone (runs headless, then opens the plot):
    export SOFA_ROOT=~/dev/build/sofa
    export PYTHONPATH=$SOFA_ROOT/lib/python3/site-packages
    python3 NewtonRaphson_MR.py [--steps 500] [--dt 0.01] [--no-predictor] [--csv out.csv] [--png out.png]

With runSofa (GUI or batch), the controller writes the samples to tip.csv next
to this file (runSofa loads scenes from their own directory); plot it afterwards with:
    python3 NewtonRaphson_MR.py --plot tip.csv
"""
import argparse
import csv
import os
import sys

import Sofa
import Sofa.Core


class TipTracker(Sofa.Core.Controller):
    """Records time, tip position and velocity, and the Newton residual at each step."""

    def __init__(self, state, newton, tip_index, csv_path=None, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.newton = newton
        self.tip = tip_index
        self.samples = []  # (t, x, y, z, vx, vy, vz, residual)
        # absolute path: SOFA changes the working directory while loading a scene
        self.csv_path = os.path.abspath(csv_path) if csv_path else None
        if csv_path:
            with open(csv_path, "w") as f:
                f.write("t,x,y,z,vx,vy,vz,residual\n")

    def onAnimateEndEvent(self, _):
        t = self.getContext().time.value
        x = self.state.position.value[self.tip]
        v = self.state.velocity.value[self.tip]
        res = self.newton.currentResidual.value
        row = (t, *x, *v, res)
        self.samples.append(row)
        if self.csv_path:
            with open(self.csv_path, "a") as f:
                f.write(",".join(f"{c:.10e}" for c in row) + "\n")


def createScene(root, dt=0.01, use_predictor=True, csv_path="tip.csv"):
    root.dt = dt
    root.gravity = [0, -9.81, 0]

    for plugin in [
        "Sofa.Component.Engine.Select",
        "Sofa.Component.IntegrationScheme.Backward",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Ordering",
        "Sofa.Component.LinearSystem",
        "Sofa.Component.Mass",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.SolidMechanics.FEM.HyperElastic",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Visual",
    ]:
        root.addObject("RequiredPlugin", name=plugin.replace(".", "_"), pluginName=plugin)

    newton = root.addObject(
        "NewtonRaphson",
        maxNbIterationsNewton=50,
        maxNbIterationsLineSearch=30,
        lineSearchReductionRate=0.3,
    )
    root.addObject("VisualStyle", displayFlags="showBehaviorModels showForceFields")

    if use_predictor:
        root.addObject("EulerImplicitIntegrationScheme", name="backward_Euler", unknownPredictor="@UnknownPredictor")
        root.addObject("ConstantVelocityPredictor", name="UnknownPredictor")
    else:
        root.addObject("EulerImplicitIntegrationScheme", name="backward_Euler")

    root.addObject("ConstantSparsityPatternSystem", template="CompressedRowSparseMatrix", name="A", checkIndices=False)
    root.addObject("NaturalOrderingMethod")
    root.addObject("SparseLDLSolver", name="linear_solver", template="CompressedRowSparseMatrix")

    root.addObject("RegularGridTopology", name="grid", min=[-0.01, -0.01, 0], max=[0.01, 0.01, 0.2], n=[5, 5, 30])
    state = root.addObject("MechanicalObject", template="Vec3", name="state", showObject=True)

    root.addObject("MooneyRivlinMaterial", name="material", mu10=1e5, mu01=1e3, bulkModulus=1e7)
    root.addObject(
        "HyperelasticityFEMForceField",
        name="FEM",
        template="Vec3,Hexahedron",
        topology="@grid",
        computeForceStrategy="sequenced",
        computeForceDerivStrategy="sequenced",
    )

    root.addObject("NodalMassDensity", property=1100)
    root.addObject("FEMMass", template="Vec3,Hexahedron")

    root.addObject("BoxROI", template="Vec3", name="box_roi", box=[-0.011, -0.011, -0.0001, 0.011, 0.011, 0.0001], drawBoxes=True)
    root.addObject("RestShapeSpringsForceField", stiffness=100000, points="@box_roi.indices")

    tip_index = 5 * 5 * 30 - 1  # last node of the grid: corner at z = 0.2
    tracker = root.addObject(TipTracker(state, newton, tip_index, csv_path=csv_path, name="tipTracker"))
    return tracker


def plot(samples, title, png=None):
    import matplotlib.pyplot as plt

    t = [s[0] for s in samples]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    for i, lab in enumerate("xyz"):
        axes[0].plot(t, [s[1 + i] for s in samples], label=lab)
        axes[1].plot(t, [s[4 + i] for s in samples], label="v" + lab)
    axes[2].semilogy(t, [max(s[7], 1e-30) for s in samples], color="k")
    axes[0].set_ylabel("tip position [m]")
    axes[1].set_ylabel("tip velocity [m/s]")
    axes[2].set_ylabel("Newton residual (squared norm)")
    axes[2].set_xlabel("time [s]")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    if png:
        fig.savefig(png, dpi=120)
        print(f"saved {png}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--no-predictor", action="store_true")
    parser.add_argument("--csv", default="tip.csv", help="where the samples are written")
    parser.add_argument("--png", default=None, help="save the figure instead of showing it")
    parser.add_argument("--plot", metavar="CSV", help="only plot an existing csv (e.g. written from runSofa)")
    args = parser.parse_args()

    if args.plot:
        with open(args.plot) as f:
            samples = [tuple(float(c) for c in row) for row in csv.reader(f) if row and row[0] != "t"]
        plot(samples, f"NewtonRaphson_MR ({args.plot})", args.png)
        return

    import SofaRuntime
    import Sofa.Simulation

    SofaRuntime.importPlugin("Sofa.Component")
    root = Sofa.Core.Node("root")
    tracker = createScene(root, dt=args.dt, use_predictor=not args.no_predictor, csv_path=args.csv)
    Sofa.Simulation.initRoot(root)

    for i in range(args.steps):
        Sofa.Simulation.animate(root, root.dt.value)
        if (i + 1) % 100 == 0:
            print(f"step {i + 1}/{args.steps}  t={root.time.value:.2f}", flush=True)

    title = f"NewtonRaphson_MR  dt={args.dt}  predictor={'on' if not args.no_predictor else 'off'}"
    plot(tracker.samples, title, args.png)


if __name__ == "__main__":
    main()
