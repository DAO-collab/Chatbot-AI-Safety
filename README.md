# Chatbot AI Safety

# How to run locally

### Clone the repository

### Create a virtual environment with Python 3.10

## install requirement via pip

```
pip3 install -r requirements.txt
```

### Configure environment variables:

copy `.env.example` to `.env` and set the variables

### RUN

```
uvicorn main:app
```

or

```
python3 main.py
```

# Task description

- clean up the code in the langchain-upgrade notebook (remove unused imports etc.)
- load prompts from text files
- implement the new chatbot chain inside main.py, in the /ws endpoint
  (currently, main.py is using an older LangChain version so when you run it with python3 main.py, it will not work)
- run and test the chatbot: https://share.cleanshot.com/b3HhfB4Y
- deploy the chatbot to a server using Github Actions
- suggest improvements and next steps for the project

Note: you can use GPT-3.5-turbo or GPT-4. We will cover any reasonable costs for the API.
