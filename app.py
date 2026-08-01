import base64
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize Groq client
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"), 
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="Universal Document AI", page_icon="🧠", layout="centered")

st.title("🧠 Universal Document AI")
st.write("Upload any document (handwritten notes, invoices, receipts). The AI will auto-detect what it is and extract the right data.")

input_mode = st.radio(
    "Select Input Method", ("Upload Document Image", "Paste Raw Text")
)

def get_universal_prompt() -> str:
    return """You are a highly intelligent Universal Document AI. Look at the document and determine if it is a "financial" document (invoice, receipt, purchase order) OR a "notes" document (handwritten notes, meeting transcripts, study notes).

Follow these rules based on what you see:
1. If FINANCIAL: Extract the ID, date, line items, subtotal, tax, and total. Strip currency symbols from numbers (e.g., 15.99). Identify the primary currency symbol used (e.g., ₹, $, €, £).
2. If NOTES: Transcribe the handwritten text cleanly and accurately, organized line by line or paragraph by paragraph. Provide a well-structured summary, and extract any action items.

CRITICAL INSTRUCTION: Do NOT use <think> tags. Do NOT explain your reasoning outside of JSON. 
Output ONLY valid JSON matching this exact schema, starting with { and ending with }:
{
  "detected_type": "financial" or "notes" or "unknown",
  
  "notes_data": {
    "raw_transcription": "string (or null)",
    "summary": "string (or null)",
    "action_items": [
      {"task": "string", "owner": "string", "due_date": "string"}
    ]
  },
  
  "financial_data": {
    "document_id": "string (or null)",
    "date": "YYYY-MM-DD (or null)",
    "currency": "string (e.g., ₹, $, €) (or null)",
    "line_items": [
      {"description": "string", "quantity": float, "unit_price": float, "total_price": float}
    ],
    "subtotal": float,
    "tax": float,
    "total_amount": float
  }
}
Leave irrelevant fields as null or empty lists depending on the detected_type."""

def safely_extract_json(text_response: str) -> dict:
    try:
        # Automatically delete any <think> blocks the AI tries to sneak in
        text_response = re.sub(r'<think>.*?</think>', '', text_response, flags=re.DOTALL).strip()
        
        # Look for JSON markdown block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text_response, re.DOTALL)
        if match:
            return json.loads(match.group(1))
            
        # Fallback: Extract from first '{' to last '}'
        start_idx = text_response.find('{')
        end_idx = text_response.rfind('}')
        if start_idx != -1 and end_idx != -1:
            return json.loads(text_response[start_idx:end_idx + 1])
            
        return json.loads(text_response)
    except Exception:
        return {"error": "Failed to parse JSON.", "raw_text": text_response}

def perform_financial_sanity_checks(data: dict) -> dict:
    fin_data = data.get("financial_data", {})
    if not fin_data:
        return data

    flags = []
    
    # 1. Date Validation
    date_str = fin_data.get("date")
    if date_str:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            flags.append(f"Date format invalid: '{date_str}' (Expected YYYY-MM-DD)")

    # 2. Line Items Math Validation
    calculated_subtotal = 0.0
    for item in fin_data.get("line_items", []):
        qty = float(item.get("quantity", 0.0) or 1.0)
        unit = float(item.get("unit_price", 0.0) or 0.0)
        total = float(item.get("total_price", 0.0) or 0.0)
        
        expected_total = round(qty * unit, 2)
        calculated_subtotal += total
        
        if abs(expected_total - total) > 0.05:
            flags.append(f"Line item math error: {qty} x {unit} != {total}")

    # 3. Document Totals Validation
    stated_subtotal = float(fin_data.get("subtotal", 0.0) or 0.0)
    stated_tax = float(fin_data.get("tax", 0.0) or 0.0)
    stated_total = float(fin_data.get("total_amount", 0.0) or 0.0)

    if abs(calculated_subtotal - stated_subtotal) > 0.05:
        flags.append(f"Subtotal mismatch: Calculated ({calculated_subtotal:.2f}) != Stated ({stated_subtotal:.2f})")

    if abs((stated_subtotal + stated_tax) - stated_total) > 0.05:
        flags.append(f"Total mismatch: Subtotal + Tax != Total Amount")

    fin_data["validation_flags"] = flags
    fin_data["is_valid"] = len(flags) == 0
    data["financial_data"] = fin_data
    return data

def process_image(image_bytes) -> dict:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b", 
        messages=[
            {"role": "system", "content": get_universal_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this document and extract the JSON. Start response immediately with {"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ],
        temperature=0.0,
    )
    raw_data = safely_extract_json(response.choices[0].message.content)
    if raw_data.get("detected_type") == "financial":
        raw_data = perform_financial_sanity_checks(raw_data)
    return raw_data

def process_text(text: str) -> dict:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": get_universal_prompt()},
            {"role": "user", "content": f"Here is the document text:\n\n{text}\n\nOutput ONLY JSON."},
        ],
        temperature=0.0,
    )
    raw_data = safely_extract_json(response.choices[0].message.content)
    if raw_data.get("detected_type") == "financial":
        raw_data = perform_financial_sanity_checks(raw_data)
    return raw_data

# UI Setup
uploaded_file = None
text_input = ""

if "Image" in input_mode:
    uploaded_file = st.file_uploader("Upload any document...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Document", width="stretch")
else:
    text_input = st.text_area("Paste document text here:", height=200)

if st.button("🚀 Process Document"):
    if "Image" in input_mode and not uploaded_file:
        st.warning("Please upload an image first.")
    elif "Text" in input_mode and not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing document type and extracting data..."):
            if "Image" in input_mode:
                result = process_image(uploaded_file.getvalue())
            else:
                result = process_text(text_input)

            if "error" in result:
                st.error("Extraction Failed.")
                st.json(result)
            else:
                # Save Output
                with open("extracted_data.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

                doc_type = result.get("detected_type", "unknown")
                st.success(f"✅ AI successfully identified this as a **{doc_type.upper()}** document.")

                # --- DISPLAY FOR NOTES ---
                if doc_type == "notes":
                    notes = result.get("notes_data", {})
                    
                    st.subheader("📝 Exact Text Transcription")
                    st.text(notes.get("raw_transcription", "No text could be read."))
                    
                    st.subheader("📋 Summary")
                    st.write(notes.get("summary", "No summary provided."))
                    
                    st.subheader("🎯 Action Items")
                    items = notes.get("action_items", [])
                    if not items:
                        st.write("No action items found.")
                    else:
                        for idx, item in enumerate(items, 1):
                            st.markdown(f"**{idx}. {item.get('task')}** (Owner: {item.get('owner')})")

                # --- DISPLAY FOR INVOICES/RECEIPTS ---
                elif doc_type == "financial":
                    fin = result.get("financial_data", {})
                    
                    if fin.get("is_valid"):
                        st.success("✅ Sanity Checks Passed: Document math is perfectly valid.")
                    else:
                        st.error("⚠️ Validation Errors Detected (Sanity Check Failed)")
                        for flag in fin.get("validation_flags", []):
                            st.write(f"- {flag}")

                    st.subheader("📑 Financial Details")
                    col1, col2 = st.columns(2)
                    col1.metric("Document ID", fin.get("document_id", "N/A"))
                    col2.metric("Date", fin.get("date", "N/A"))

                    st.write("**Line Items:**")
                    st.table(fin.get("line_items", []))

                    cur = fin.get("currency") or ""
                    
                    col3, col4, col5 = st.columns(3)
                    col3.metric("Subtotal", f"{cur}{fin.get('subtotal', 0.0)}")
                    col4.metric("Tax", f"{cur}{fin.get('tax', 0.0)}")
                    col5.metric("Total", f"{cur}{fin.get('total_amount', 0.0)}")

                else:
                    st.warning("Could not determine document type.")
                    st.json(result)

                st.download_button(
                    label="📥 Download Extracted JSON",
                    data=json.dumps(result, indent=2),
                    file_name="extracted_data.json",
                    mime="application/json",
                )