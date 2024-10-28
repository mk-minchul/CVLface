
# Research Code Structure and Workflow

## Introduction

This README outlines the structured approach we use in managing and tracking experiments in our research projects. Each folder under `code/` is an independent project, designed to facilitate easy tracking of changes and experiment variations.

## Recommended Workflow: Copy-Folder Method

While not mandatory, the copy-folder method is highly recommended for conducting research as it provides a straightforward way to manage versions and track modifications semantically.

### Steps to Follow

1. **Initialize Your Experiment:**
   - Start by copying the `run_v1` folder to create a new project folder, such as `new_loss_v1`. This new folder should reflect the specific aspect of the experiment you are modifying (e.g., testing a new loss function).

2. **Modify the Code:**
   - Make necessary changes in the newly created folder, adjusting the code as per your experimental requirements.

3. **Semantic Versioning:**
   - Label each modification semantically, which aids in easily tracking and referencing different experiments. For example, after further modifications, rename the folder to `new_loss_v2`.

4. **Iterate:**
   - Repeat this process for each new idea or modification, incrementing the version number with each significant change.

### Example Folder Structure

```
code/
    run_v1 (default)/
        ...
    new_loss_v1/
        ...
    new_loss_v2/
        ...
    new_architecture_v1/
        ...
    ...
```

## Managing Experiment Results

All experimental results are organized in the `experiments/` folder, stored in a structured format to facilitate easy navigation and comparison.

### Results Saving Format

Results are saved using the following naming convention:
```
{folder_name}/{optional_prefix}_{Month}-{Date}_{Trial}
```

### Example Experiments Structure

```
experiments/
    run_v1/
        run_02-15_0/
    new_loss_v1/
        optim_sgd_02-15_0/
    new_loss_v1/
        optim_adam_02-15_1/
    new_loss_v1/
        bugfix_02-16_0/
    new_loss_v2/
        batchsize100_03-12_0/
    new_architecture_v1_04-14_0/
    ...
```

## FAQ

**Why not use Git for version control and history tracking?**
- While Git is an excellent tool for version control, the copy-folder method offers simplicity in tracking and comparing results directly through folder structures. IDEs such as VSCode or PyCharm enhance this process by enabling easy folder comparisons.
- This method allows each experiment to run independently, which is particularly useful for conducting multiple parallel experiments.
- You can combine Git for detailed file changes tracking with the copy-folder method for managing different experimental projects, leveraging the strengths of both approaches.

## Conclusion

Our structured approach to code and experiment management notifies how we prioritize clarity, ease of access, and robust tracking in our research environment. Follow these guidelines to ensure a consistent and efficient workflow in your research activities.