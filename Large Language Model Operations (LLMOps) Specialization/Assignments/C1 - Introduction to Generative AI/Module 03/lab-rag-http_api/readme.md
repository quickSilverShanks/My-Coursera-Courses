The webapp has two codes:
- main_openai.py : This uses openai library to connect with llamafile based LLM. Currently this does not work due to openai unable to connect with locally running LLM application. This needs to be fixed, probably version change and minor code changes. To work with this code, uncomment openai from requirements file and install the packages.
- main.py : This uses requests library to interact with  llamafile based LLM. This is ready to use and is  served using FastAPI.

