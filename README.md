```markdown
# Cyber Epidemic ML

A Flask web application that applies *Machine Learning models* to defend against epidemic-style cybersecurity threats such as *XSS attacks, SQL Injection, and Malicious URLs*.

---

## Features

- 🛡️ **XSS Detection** — Uses a deep learning model (`xss_model.h5`)  
- 🔍 **SQL Injection Detection** — Uses trained ML model (`saved_model.pkl`)  
- 🌐 **Malicious URL Detection** — Uses ensemble model (`BestModel_ExtraTreesClassifier.joblib`)  
- 📊 Web interface for uploading URLs and queries  
- ⚡ Real-time results directly in your browser  

---

## Project Structure

```

project/
│── app.py                # Main Flask app
│── web\_vul.py            # Web vulnerability scanning logic
│── requirements.txt       # Python dependencies
│── models/               # Pretrained ML models (.h5, .pkl, .joblib, etc.)
│── templates/            # HTML templates
│── static/               # CSS, JS, assets

````

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/CHAbhiram/cyber-epidemic-ml.git
   cd cyber-epidemic-ml/project
````

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

Start the Flask server with:

```bash
python app.py
```

By default, the app runs on:

```
http://127.0.0.1:5000
```

Open this link in your browser.

---

## Usage

1. Navigate to the homepage.
2. Select the **section/module** you want:

   * **XSS Detection** → Paste a query or payload to test
   * **SQL Injection Detection** → Enter a query to check for injection
   * **URL Threat Detection** → Upload or paste URLs for classification
3. Submit your input.
4. View results in the browser (classified as safe/threat).

---

## Example Workflow

* Go to `http://127.0.0.1:5000`

* Select *URL Threat Detection*

* Paste:

  ```
  http://malicious-example.com/abc?id=1
  ```

* Output:

  ```
  Prediction: Malicious URL
  ```

---

## Screenshots / Demo

### Homepage

<img width="1877" height="970" alt="Screenshot 2025-09-20 212619" src="https://github.com/user-attachments/assets/ac8f90b9-f48b-4eb0-83c6-e72a6f294890" />


### XSS Detection Module

<img width="1875" height="971" alt="Screenshot 2025-09-20 212631" src="https://github.com/user-attachments/assets/d8e6c5ad-dfad-474f-8fb0-2417c028450e" />


### SQL Injection Detection

<img width="1880" height="973" alt="Screenshot 2025-09-20 212641" src="https://github.com/user-attachments/assets/95510572-538a-4714-bc0f-06afdf08eecd" />


### URL Threat Detection

<img width="1874" height="967" alt="Screenshot 2025-09-20 212659" src="https://github.com/user-attachments/assets/55b118a3-26bb-4050-a21e-b1a1d7cf9e5b" />


---

## Troubleshooting

* *Model not found* → Ensure the `models/` folder contains required `.h5`, `.pkl`, `.joblib` files.
* *Dependency errors* → Reinstall with `pip install -r requirements.txt`.
* *Port conflicts* → Run with a different port:

  ```bash
  flask run --port 8080
  ```

---

## License

MIT License

Copyright (c) 2025 Chada Abhiram Sharma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE..

```

---
