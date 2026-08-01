# 🧠 Universal Document AI - Data Extractor 


## 📌 One-Sentence Overview
My agent takes messy financial documents (receipts, invoices) or handwritten notes and produces validated, structured JSON financial data or clean text summaries with extracted action items.

---

## 🧾 Real-World Bill Anomaly & Agent Validation (`bill.jpg`)

### 🔍 The Problem in the Bill
* **Pricing Discrepancy:** The restaurant bill (`bill.jpg`) contains a real-world vendor pricing inconsistency for the item **"Gulab Jamun"**.  
* **Details:** The bill lists a **Quantity of 2** at a **Rate of ₹100.00** each. Mathematically, 2 units at ₹100 should total ₹200.00, but the printed line-item amount on the bill is listed as **₹100.00**.

### 🤖 What the Agent Did
* **Data Extraction:** The agent successfully scanned the layout and extracted all line items, document ID (`1304`), and date (`2024-04-24`).
* **Math Handling & Sanity Check:** Rather than crashing or blindly overriding the text, the agent parsed the printed line-item amounts as written (calculating a subtotal of ₹490.00, adding CGST/SGST taxes totaling ₹24.50, and matching the final printed total of **₹514.50**). Because the subtotal plus tax matched the printed total within the accepted tolerance, the agent successfully passed the financial validation check.

---

## 🚀 Setup & Installation

Follow these steps to run the agent locally in minutes:

**1. Clone the repository:**
```bash
git clone [https://github.com/Adan-2128/Universal-Document-AI-Agent.git](https://github.com/Adan-2128/Universal-Document-AI-Agent.git)
cd Universal-Document-AI-Agent
