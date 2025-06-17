## External lab: Create a RAG with LLM using your own data

**Environment Setup**

```bash
conda create -n coursera_i2genai python=3.12
conda activate coursera_i2genai
cd "d:/..../.../..."        # move to the project folder with requirements.txt file
pip install -r requirements.txt
```

- LLM (llamafile) Used: [google_gemma-3-4b-it-Q6_K.llamafile](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#other-example-llamafiles)
- Data Used: [TMDB Top Movies Dataset](https://www.kaggle.com/datasets/spypsc07/imdb-top-movies-dataset)

Note: For windows, llamafile can be executed only if its < 4GB. To run larger files, there is some complicated [workaround](https://github.com/Mozilla-Ocho/llamafile/issues/98#issuecomment-1857551456).

**Steps to get the lab_RAG notebook working**

- For Windows, rename the llamafile by adding ".exe" on the end.
- Run the LLM executable directly or as below (9999 means "offload as many layers as possible to the GPU")
```bash
google_gemma-3-4b-it-Q6_K.llamafile -ngl 9999
```
- Go to the url 'http://localhost:8080' to display a chat interface for above LLM.
- Run lab_RAG with custom queries as desired.
