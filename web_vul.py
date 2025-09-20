from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Helper Function to Check for SQL Injection
def check_sql_injection(url):
    test_payload = "' OR '1'='1"
    try:
        response = requests.get(f"{url}?id={test_payload}", timeout=5)
        if "syntax error" in response.text.lower() or "database" in response.text.lower():
            return True
    except requests.exceptions.RequestException:
        pass
    return False

# Helper Function to Check for Missing Security Headers
def check_security_headers(response):
    vulnerabilities = []
    headers = response.headers
    
    if "Content-Security-Policy" not in headers:
        vulnerabilities.append("Missing Content-Security-Policy header.")
    if "X-Content-Type-Options" not in headers:
        vulnerabilities.append("Missing X-Content-Type-Options header.")
    if "Strict-Transport-Security" not in headers:
        vulnerabilities.append("Missing Strict-Transport-Security header.")
    if "X-Frame-Options" not in headers:
        vulnerabilities.append("Missing X-Frame-Options header.")
    
    return vulnerabilities

# Helper Function to Check for XSS Vulnerabilities
def check_xss(url):
    vulnerabilities = []
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        forms = soup.find_all("form")
        for form in forms:
            action = form.get("action") or ""
            inputs = form.find_all("input")
            for input_tag in inputs:
                name = input_tag.get("name")
                if name:
                    xss_payload = "<script>alert('XSS')</script>"
                    target_url = url + action
                    xss_test = requests.post(target_url, data={name: xss_payload}, timeout=5)
                    if xss_payload in xss_test.text:
                        vulnerabilities.append(f"Potential XSS vulnerability in form: {action}")
    except requests.exceptions.RequestException:
        pass
    
    return vulnerabilities

# Endpoint to Scan Website
@app.route('/scan', methods=['POST'])
def scan_website():
    data = request.json
    url = data.get("url")
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    results = {"url": url, "vulnerabilities": []}
    
    # Check for SQL Injection Vulnerability
    if check_sql_injection(url):
        results["vulnerabilities"].append("Potential SQL Injection detected.")
    
    # Check for Missing Security Headers
    try:
        response = requests.get(url, timeout=5)
        header_vulnerabilities = check_security_headers(response)
        results["vulnerabilities"].extend(header_vulnerabilities)
    except requests.exceptions.RequestException:
        return jsonify({"error": "Failed to connect to the URL"}), 500
    
    # Check for XSS Vulnerabilities
    xss_vulnerabilities = check_xss(url)
    results["vulnerabilities"].extend(xss_vulnerabilities)
    
    return jsonify(results)

# Home Endpoint
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Web Vulnerability Detection Tool!",
        "instructions": "Use the /scan endpoint with a POST request and provide a 'url' in the JSON body to scan for vulnerabilities."
    })

if __name__ == '__main__':
    app.run(debug=True)
