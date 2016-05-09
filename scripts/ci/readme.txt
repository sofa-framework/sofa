Continuous Integration Scripts
------------------------------


This directory contains the scripts used on Jenkins, the continuous integration
system used for Sofa.  Each time commits are pushed in a branch, those scripts
are executed on various machines to build the tip of the branch and run the
automatic tests that are available.

Those are bash scripts that run on Linux, OS X and Windows.  They are tailored
to the machines used behind Jenkins, but can reasonnably be expected to run on
any Linux or OS X system.

Obviously, you should only commit changes to those scripts that make perfect
sense to you.  Few people are expected to modify those scripts; if you are not
sure you are one of them, then you aren't.


The main build script is main.sh, and is responsible for calling the other ones
and communicating with the dashboard (http://www.sofa-framework.org/dash/).

  main.sh
  ├── init-build.sh
  ├── configure.sh
  ├── compile.sh
  ├── tests.sh
  └── scene-tests.sh


Miscellaneous notes / peculiarities:

- There is already a 'find' command on Windows, that is found before the unix
  "find", so we use the absolute path "/usr/bin/find".

- There is a "timeout" command on Windows, that has nothing to do with the
  regular "timeout" command, so we implement our own timeout in timeout.sh.

- BSD wc has a tendency to print spurious spaces/tabs, so we usually follow it a
  tr command to remove blanks.

- BSD sed has no \+ metacharacter.
