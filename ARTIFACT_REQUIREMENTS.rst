## **Hardware requirements:**

-   **GPU requirements**: Because our approach is based on DeepLearning,
    having a GPU would be beneficial if the user intends to retrain the
    models. Using only a CPU would result in extremely slow performance.
    Our paper describes the GPU we used in detail, but other GPU
    generations can be used. Some of our scripts make the assumption
    that the GPU is from NVIDIA. This can be changed by searching across
    the code for the string \"cuda:0\" and replacing it with the right
    GPU identifier.

-   **Disk Space**: We have multiple data files that sums up to 45 GB +
    12 GB for docker image.

-   **RAM free space**: some data preprocessing requires large RAM
    space, in our experiments the usage might go up to 80 GBs of RAM.
    However, we provide in our instructions a toy example that does not
    consume much resources.

-   **Internet**: when installing or using our package there is some
    data that will be fetched from online sources.

-   **CPU cores**: to speedup some of the pre-processing and data
    preparation tasks we used multiprocessing (we used 48 cpus).

## **Software requirements:**

-   **Docker and OS**: Docker should be installed to be able to use our
    Docker image. If the user wants to install the package on the host
    machine without using Docker, we recommend Ubuntu/Debian (we provide
    instructions in the README file).

-   **Python version**: The implementation and used packages are
    compatible with Python3.8 (recommended) and above.

-   **Jupyter-Lab**: We also provide one Jupyter-Notebook to make RoC
    curves that we put in the paper.

-   **OpenAI API token**: One of our modules use a generative model
    (Codex) from OpenAI. We provide the code for that module but it
    would require a Token to access the OpenAI API.

-   **GitHub user name and Token**: to collect data from GitHub
    repositories, CMI-Finder requires a username and a token in some
    steps (data collection steps).
