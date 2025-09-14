# Financial Filings RAG System

This project allows you to store financial filings into **ChromaDB** and ask questions about your data using an LLM-powered RAG pipeline.

---

## 🚀 Setup Instructions (macOS)

### 1. Create Python Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python3 main.py
```

---

## 📌 Usage

When you run `main.py`, you will see options in the terminal.

- **Option 1** → Store new data into **ChromaDB**
- After data is stored, you can ask any questions related to your data.

---

## ✅ Example

```bash
python3 main.py
```
```
Select an option:
1. Store new files to vector db
2. Chat
3. Exit1
```

If you select **1**, it will process and embed your data.  
Then, you can select **2** to query your dataset.

---

