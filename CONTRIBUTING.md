# Getting started

Please **read carefully** [the GitHub guide on Contributing to Open Source](https://guides.github.com/activities/contributing-to-open-source/). We also recommend you to check this [more detailed documentation on issues and pull requests](https://help.github.com/categories/collaborating-with-issues-and-pull-requests/).


# Forum

For any **question** (help needed, problem of understanding SOFA, announcements), create a topic on [the SOFA forum](https://www.sofa-framework.org/community/forum/) and benefit from the feedback of the community.

When creating a new topic, pay attention to some tips:

- **Check existing topics** using the Search bar. Your question may have been answered already.
- **Be clear** about what your problem is: what was the expected outcome, what happened instead? Detail how someone else can recreate the problem.
- **Additional infos**: link to demos, screenshots or code showing the problem.


# Issues

For **bug tracking**, **feature proposals** and **task management**, create a [SOFA issue](https://github.com/sofa-framework/sofa/issues)! There is nothing to it and whatever issue you are having, you are likely not the only one, so others will find your issue helpful, too. Issues labeled "discussion" are also used for larger topics: architecture, future of SOFA, long term dev, etc.

Please **DO NOT create an issue for questions or support**. Use [the SOFA forum](https://www.sofa-framework.org/community/forum/) instead.

When creating an issue, pay attention to the following tips:

- **Check existing issues**. What you are running into may have been addressed already.
- **Set the right label** to your issue among our label list or propose them in the description.
- **Be clear** about what your problem is: what was the expected outcome, what happened instead? Detail how someone else can recreate the problem.

For more information on issues, check out [this GitHub guide](https://guides.github.com/features/issues/).  


# Pull-requests

If you are able to patch the bug or add the feature yourself – fantastic, make a pull request with the code! Be sure you have read any documents on contributing and you understand [the SOFA license](https://github.com/sofa-framework/sofa/blob/master/LICENCE.txt). Once you have submitted a pull request the maintainer(s) can compare your branch to the existing one and decide whether or not to incorporate (pull in) your changes.

### Reminder - How to pull request (from GitHub documentation)

- **[Fork](http://guides.github.com/activities/forking/)** the repository and clone it locally.
- [Connect your clone](https://help.github.com/articles/configuring-a-remote-for-a-fork/) to [the original **upstream** repository](https://github.com/sofa-framework/sofa/) by adding it as a remote.
- **Create a branch** for your changes.
- Make your changes.
- Pull in changes from upstream often to [**sync your fork**](https://help.github.com/articles/syncing-a-fork/) so that merge conflicts will be less likely in your pull request.
- [**Create a pull-request**](https://help.github.com/articles/creating-a-pull-request-from-a-fork/) when you are ready to propose your changes into the main project.

### Rules for SOFA pull requests

- Description must explain the **issue solved** or the **feature added**, and this must be reported in the **[CHANGELOG.md](https://github.com/sofa-framework/sofa/blob/master/CHANGELOG.md)** file.
- Code must follow **[our guidelines](https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md)**.
- Commit must build **successfully** on Jenkins for all steps (compilation + tests + examples).
- **Unit Tests** are required for each new component or if an issue is fixed.
- **Examples** (at least one) must be provided showing the new feature.

### Reviewing (for the reviewers team)

- Make sure the pull request is **labelized** and well assigned.
- Control that it follows **our rules** (defined above).
- You can **add commits** in a pull request: see [GitHub documentation](https://help.github.com/articles/committing-changes-to-a-pull-request-branch-created-from-a-fork/).
- If the pull request contains out of scope commits (from a previous merge with master), **consider rebasing it**.
- **Control the builds**: Dashboard > Details in the pull request checks.
- Merge method: **prefer "rebase"** or "squash" over "merge" to keep linear history.


For more information on forks and pull request, check out [this GitHub guide](https://guides.github.com/activities/forking/).
