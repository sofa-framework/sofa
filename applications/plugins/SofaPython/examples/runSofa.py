#!/usr/bin/env python2

import argparse
import os
from libSofaPython import Sofa


def main():
    """
    Equivalent to the cpp runSofa application, this script will launch a Sofa's GUI or run a simulation in batch mode

    To run this file, make sure the directory containing the file "libSofaPython.so" is in
    your PYTHONPATH environment variable.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--start",
                        action='store_true',
                        help="Start the animation loop.")

    parser.add_argument("-c", "--computation_time_sampling",
                        type=int,
                        metavar='N',
                        help="Frequency of display of the computation time statistics in number of animation steps.")

    supported_gui = Sofa.GUIManager.listSupportedGUI()
    parser.add_argument("-g", "--gui",
                        choices=supported_gui,
                        default="qglviewer",
                        help="Graphical interface to use.")

    parser.add_argument("-l", "--load",
                        nargs="*",
                        metavar='file',
                        default=[],
                        help="Load given plugins.")

    parser.add_argument("-n", "--nb_iterations",
                        type=int,
                        metavar='N',
                        default=1,
                        help="(only batch) Number of iterations of the simulation.")

    parser.add_argument("--prefix",
                        type=str,
                        metavar='dir',
                        default=os.environ.get('SOFA_ROOT', os.path.curdir),
                        help="Prefix path where the GUI's config and screenshots directories are or should be created.")

    parser.add_argument('scenefile', nargs='?', type=str, default="")

    o = parser.parse_args()

    # Load the plugins
    plugins = o.load
    for plugin in plugins:
        Sofa.loadPlugin(plugin)

    # Set the simulation
    Sofa.setSimulation(Sofa.createSimulation("DAG"))
    if o.scenefile:
        root = Sofa.loadScene(o.scenefile)
    else:
        root = Sofa.createNode("root")

    # todo(jnbrunet2000@gmail.com): load default plugins from ini file

    # Set the GUI
    if o.gui == "batch":
        Sofa.GUIManager.AddGUIOption("nbIterations={}".format(o.nb_iterations))
    else:
        Sofa.GUIManager.setSofaPrefix(o.prefix)
        Sofa.GUIManager.setConfigDirectoryPath(os.path.join(o.prefix, "config"))
        Sofa.GUIManager.setScreenshotDirectoryPath(os.path.join(o.prefix, "screenshots"))
        Sofa.GUIManager.setDimension(800, 600)

    Sofa.GUIManager.init(o.gui)
    Sofa.GUIManager.createGUI()
    Sofa.GUIManager.setScene(root)

    root.init()

    if o.start:
        root.animate = True

    if o.computation_time_sampling > 0:
        Sofa.timerSetEnabled("Animate", True)
        Sofa.timerSetInterval("Animate", o.computation_time_sampling)

    Sofa.GUIManager.MainLoop(root)
    Sofa.GUIManager.closeGUI()


if __name__ == "__main__":
    main()
