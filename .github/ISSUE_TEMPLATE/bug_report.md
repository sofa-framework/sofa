---
name: Bug report
about: Create a report to help us improve SOFA
title: ''
labels: 'issue: bug (minor)'
assignees: ''

---

### Problem

**Description**
< DESCRIBE YOUR PROBLEM HERE >

**Steps to reproduce**
< LIST THE STEPS TO REPRODUCE YOUR PROBLEM HERE >

**Expected behavior**
< DESCRIBE WHAT YOU THINK SHOULD HAVE HAPPEN INSTEAD >

---------------------------------------------

### Environment

**Context**

- System: < NAME AND VERSION - e.g: "Windows 10", "Ubuntu 20.04", ... >
- Version of SOFA: < INFOS ABOUT THE BRANCH OR BINARIES - e.g: "master branch at commit 70bb123", "v21.06.00 binaries", ... >
- State: < BUILD OR INSTALL DIRECTORY - e.g: "Build directory", "Install directory" >

**Command called**

```txt

< COPY-PASTE YOUR COMMAND HERE >

```

**Env vars**

```bash
python -c "exec( \"import os, sys\nprint('#################')\nprint('--- sys.version ---')\nprint(sys.version)\nprint('--- PATH ---')\ntry:\n  print(os.environ['PATH'])\nexcept Exception:\n  pass\nprint('--- SOFA_ROOT ---')\ntry:\n  print(os.environ['SOFA_ROOT'])\nexcept Exception:\n  pass\nprint('--- PYTHONPATH ---')\ntry:\n  print(os.environ['PYTHONPATH'])\nexcept Exception:\n  pass\nprint('--- sys.path ---')\ntry:\n   print(str(sys.path))\nexcept Exception:\n   pass\nprint('#################')\" )"
```

```txt

< COPY-PASTE HERE THE RESULT OF THE COMMAND ABOVE >

```

---------------------------------------------

### Logs

**Full output**

```txt

< COPY-PASTE YOUR OUTPUT HERE >

```

**Content of build_dir/CMakeCache.txt**

< DRAG AND DROP YOUR CMAKECACHE.TXT HERE >

---------------------------------------------

Thank you for your report.
