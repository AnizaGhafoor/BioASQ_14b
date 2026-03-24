# BioASQ_14b
BioASQ 14b QA system with BM25 retrieval, query expansion, reranking, and LLM-based answer generation for biomedical question answering.
## Setup and Installation
Follow these steps to set up the environment and prepare the necessary data and models.
### 1. Prerequisites
Works well with Python 3.8 – 3.11 (we used 3.11.6)
A system with sufficient RAM and a modern NVIDIA GPU (for the reranker and generation phases).
### 2. Clone the Repository
git clone https://github.com/AnizaGhafoor/BioASQ_14b
cd BioASQ_14b
### 3. Install Dependencies
Create and activate a virtual environment, then install the required packages.
----sh----
python -m venv pt_env
source pt_env/bin/activate
pip install -r requirements.txt
### 4. Install Java_SDK
 "OpenJDK17U-jdk_x64_windows_hotspot_17.0.18_8.msi"
**--------------------------------------permanent setup--------------------------------------**
Press Win + S, search “Environment Variables” → Edit the system environment variables
Click Environment Variables…
Under System variables → New:
Variable name: JAVA_HOME
Variable value: C:\Program Files\Adoptium\jdk-17.0.8.1
click ok---> Apply ----> restart vscode
**----------------------------------temporary for session---------------------------------------**
----sh-----
$env:JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
$env:JPYPE_JVM_DLL="$env:JAVA_HOME\bin\server\jvm.dll"

