
This directory contains the scripts used on Jenkins, the continuous integration
system used for Sofa.

Those are bash scripts that run on Linux, OS X and Windows.  They are tailored
to the machines used behind Jenkins, but can reasonnably be expected to run on
any Linux or OS X system.

Obviously, you should not commit any change to those scripts that does not make
perfect sense to you.  Few people are expected to modify those scripts; if you
are not sure you are amongst them, then you aren't.




Miscellaneous notes / peculiarities:

- There is already a 'find' command on Windows, that is found before the unix
  "find", so we use the absolute path "/usr/bin/find".

- There is a "timeout" command on Windows, that has nothing to do with the
  regular "timeout" command, so we implement our own timeout in timeout.sh.

- BSD wc has a tendency to print spurious spaces/tabs, so we usually follow it a
  tr command to remove blanks.

- BSD sed has no \+ metacharacter.
