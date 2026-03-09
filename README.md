# FINed (Financial Education)
A Haystack pipeline application focusing on financial education. It uses Ollama, Milvus and Streamlit.


## Clone the repo
```bash
git clone https://github.com/initdream/fined
```
## Install Dependencies
**Arch Linux:**
```bash
yay -S ollama-for-amd
```
Create a virtual environment and install the required packages:
```bash
cd fined/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Run the Application
Open a terminal and start the Ollama server:
```bash
ollama serve
```
Open a **second** terminal (ensure the virtual environment is activated) and run the Streamlit app:
```bash
source venv/bin/activate
streamlit run app_streamlit.py
```