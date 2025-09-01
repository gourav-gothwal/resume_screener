# 📄 Resume Screener & Job Match Engine

An **AI-powered resume screening and job recommendation system** that leverages **Natural Language Processing (NLP)** to match candidates with suitable job openings.  
Simply upload your resume, and the model analyzes your skills & experience to provide **personalized career recommendations**.

---

## 🚀 Features

- 🔍 **Resume Parsing** – Extracts skills, experience, and education from resumes.  
- 🤖 **NLP-Powered Matching** – Uses machine learning to analyze text and match candidates with jobs.  
- 🎯 **Job Recommendation** – Suggests the most relevant job opportunities based on resume insights.  
- 📊 **Evaluation & Testing** – Includes scripts for training, evaluation, and testing models.  

---

## 🛠️ Tech Stack

- **Language:** Python 3  
- **Libraries:** NLP & ML libraries (e.g., scikit-learn, spaCy, PyTorch/TensorFlow)  
- **Notebooks:** For experimentation and model development  
- **Scripts:** Training, evaluation, and testing pipelines  

---

## 📂 Project Structure

```
resume_screener/
│-- notebooks/         # Jupyter notebooks for experiments & EDA
│-- training.py        # Script to train the ML/NLP model
│-- evaluation.py      # Script to evaluate trained models
│-- test.py            # Script to test resumes against job data
│-- .gitignore
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/gourav-gothwal/resume_screener.git
cd resume_screener

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
```

*(If `requirements.txt` is missing, install core NLP/ML libraries like `scikit-learn`, `pandas`, `numpy`, `torch` or `tensorflow`, and `spacy`.)*

---

## ▶️ Usage

### 1. Train the model
```bash
python training.py
```

### 2. Evaluate performance
```bash
python evaluation.py
```

### 3. Test on a resume
```bash
python test.py --resume path/to/resume.pdf
```

---

## 📊 Example Workflow

1. Upload a resume (PDF or text).  
2. The system extracts features like skills, experience, and education.  
3. The model compares extracted features against job postings.  
4. Generates a ranked list of **recommended jobs**.  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork the repo, create a new branch, and submit a pull request.  

---

## 📜 License

This project is licensed under the **MIT License**.  
You’re free to use, modify, and distribute it with attribution.

---

## 🙌 Acknowledgements

Inspired by real-world HRTech systems that streamline hiring by combining **data science, NLP, and machine learning**.
