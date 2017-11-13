# Welcome

Welcome to the SOFA Community! Here is a brief summary of how it is structured:
- SOFA Users: people using SOFA by writing scenes or using the SOFA API.
- SOFA Developpers: people programming into SOFA, modifying the API, writing plugins.
- SOFA Contributors: (awesome) people proposing their changes in SOFA code via pull-requests.
- SOFA Reviewers: people reviewing and merging the pull-requests. This group is validated by the Scientific and Technical Committee (STC).
- SOFA Consortium: research centers and companies willing to share the cost of development and maintenance of SOFA, hosted by the Inria Foundation.
- SOFA Consortium Staff: administrators of SOFA and its ecosystem. This group is directed by the Executive Committee (EC).

All SOFA Developpers are gladly invited to the SOFA-dev meetings.  
They take place remotely every Wednesday at 10 a.m. CET and are animated by the SOFA Reviewers + the Consortium Staff.  
[Subscribe to SOFA-dev mailing-list](https://sympa.inria.fr/sympa/subscribe/sofa-dev) to get the agenda, reports and conference room url.

About the steering committees:
- SOFA Scientific and Technical Committee (STC): defines the technical roadmap twice a year, validate the contribution rules, the Reviewers team and discuss every technical point in SOFA.
- SOFA Executive Committee (EC): decides on evolutions of the membership contract, the communication policy and the priorities of the Consortium.

A more detailed definition of the committees is available [in the SOFA website](https://www.sofa-framework.org/consortium/presentation/).


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
- If your issue reports a bug or any abnormal behavior in SOFA, a **test highlighting the issue** should be written and pull requested.

For more information on issues, check out [this GitHub guide](https://guides.github.com/features/issues/).  


# Pull requests

If you are able to patch the bug or add the feature yourself – fantastic, make a pull request with the code! Be sure you have read any documents on contributing and you understand [the SOFA license](https://github.com/sofa-framework/sofa/blob/master/LICENCE.txt). Once you have submitted a pull request the maintainer(s) can compare your branch to the existing one and decide whether or not to incorporate (pull in) your changes.

### Reminder - How to pull request (from GitHub documentation)

- **[Fork](http://guides.github.com/activities/forking/)** the repository and clone it locally.
- [Connect your clone](https://help.github.com/articles/configuring-a-remote-for-a-fork/) to [the original **upstream** repository](https://github.com/sofa-framework/sofa/) by adding it as a remote.
- **Create a branch** for your changes.
- Make your changes.
- Pull in changes from upstream often to [**sync your fork**](https://help.github.com/articles/syncing-a-fork/) so that merge conflicts will be less likely in your pull request.
- [**Create a pull request**](https://help.github.com/articles/creating-a-pull-request-from-a-fork/) when you are ready to propose your changes into the main project.

### Rules

- Description must explain the **issue solved** or the **feature added**, and this must be reported in the **[CHANGELOG.md](https://github.com/sofa-framework/sofa/blob/master/CHANGELOG.md)** file.
- Code must follow **[our guidelines](https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md)**.
- Commit must build **successfully** on Jenkins for all steps (compilation + tests + examples).
- **Unit Tests** are required for each new component or if an issue is fixed.
- **Examples** (at least one) must be provided showing the new feature.

### Lifecycle

Standard pull-requests are reviewed and approved by the "Reviewers" team.  
Major pull-requests (BREAKING, major features) are reviewed by the "Reviewers" team and approved by the "STC members" team through a vote within a maximum period of 2 weeks.

Reviewing:

- Make sure the pull request is **labelized** and well assigned.
- Control that it follows **our rules** (defined above).
- Track the **status of each pull request** using the dedicated labels:
  - "pr: wip" must be set if the PR has been created for a team work or if some fixes are needed (discussed in the comments).
  - "pr: to review" must be set if the PR is ready to be reviewed. 
  - "pr: ready" must be set **instead of merge** if another PR merge is being built on the [Dashboard](https://www.sofa-framework.org/dash/?branch=origin/master). It is used to delay the merge and avoid CI overflow.
- **Control the builds**: Dashboard > Details in the pull request checks.
- Merge method: **prefer "merge"** or "squash" over "rebase".

Remember that:

- You can **add commits** in a pull request: see [GitHub documentation](https://help.github.com/articles/committing-changes-to-a-pull-request-branch-created-from-a-fork/).
- If the pull request contains out of scope commits (from a previous merge with master), **consider rebasing it**.

For more information on forks and pull request, check out [this GitHub guide](https://guides.github.com/activities/forking/).

### SOFA Contributor License Agreement ("Agreement"), v1.0

Note that by submitting a "Contribution" (shall mean the code, documentation or other original works of authorship) to SOFA, through the Pull Request mechanism in github or by other means, "You" (the entity that is making this Contribution) hereby accept and agree to the following terms and conditions for Your present and future Contributions:

- You certify that You have authored 100% of the content, or in the event part or all of the content has been authored by a third party, you have appropriate permissions to include their work in your Contribution,
- You certify You have the necessary rights to license the Contribution (i.e. if You are employed You have received the necessary permissions from Your employer to make the Contribution),
- You grant a perpetual, irrevocable, non-exclusive, royalty-free, worldwide license to distribute this Contribution under the license agreement commonly known as the [LGPL License](https://opensource.org/licenses/LGPL-2.1).
- You will remain the copyright holder of your Contribution.
