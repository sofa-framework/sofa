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
bash -c '
  echo "SOFA_ROOT = $SOFA_ROOT"
  echo "PYTHONPATH = $PYTHONPATH"
  echo "python -V = $(python -V 2>&1)"
  echo "python3 -V = $(python3 -V 2>&1)"
  '
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
