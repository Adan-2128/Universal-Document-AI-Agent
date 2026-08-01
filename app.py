import base64
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import fitz  # PyMuPDF for PDF handling

# Load environment variables
load_dotenv()

# Initialize Groq client
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"), 
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="Universal Document AI", page_icon="🧠", layout="centered")

st.title("🧠 Universal Document AI")
st.write("Upload any document (PDFs, handwritten notes, invoices, receipts). The AI will auto-detect what it is and extract the right data.")

input_mode = st.radio(
    "Select Input Method", ("Upload Document", "Paste Raw Text")
)

def get_universal_prompt() -> str:
    return """You are a highly intelligent Universal Document AI. Look at the document and determine if it is a "financial" document (invoice, receipt, purchase order) OR a "notes" document (handwritten notes, meeting transcripts, study notes).

Follow these rules based on what you see:
1. If FINANCIAL: Extract the ID, date, line items, subtotal, tax, and total. Strip currency symbols from numbers. Identify the primary currency symbol used (e.g., ₹, $, €, £).
2. If NOTES: Transcribe the text cleanly, provide a structured summary, and extract action items if any exist.

CRITICAL INSTRUCTION: Output ONLY a valid JSON object. Do not include any <think> tags, conversational filler, or Markdown code blocks outside the JSON. Start your response with { and end with }.

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
    "currency": "string (or null)",
    "line_items": [
      {"description": "string", "quantity": float, "unit_price": float, "total_price": float}
    ],
    "subtotal": float,
    "tax": float,
    "total_amount": float
  }
}"""

def safely_extract_json(text_response: str) -> dict:
    try:
        text_response = re.sub(r'<think>.*?</think>', '', text_response, flags=re.DOTALL).strip()
        start_idx = text_response.find('{')
        end_idx = text_response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            clean_json = text_response[start_idx:end_idx + 1]
            return json.loads(clean_json)
            
        return json.loads(text_response)
    except Exception as e:
        return {"error": f"Failed to parse JSON: {str(e)}", "raw_text": text_response}

def perform_financial_sanity_checks(data: dict) -> dict:
    fin_data = data.get("financial_data", {})
    if not fin_data:
        return data

    flags = []
    
    date_str = fin_data.get("date")
    if date_str:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            flags.append(f"Date format invalid: '{date_str}' (Expected YYYY-MM-DD)")

    calculated_subtotal = 0.0
    for item in fin_data.get("line_items", []):
        qty = float(item.get("quantity", 0.0) or 1.0)
        unit = float(item.get("unit_price", 0.0) or 0.0)
        total = float(item.get("total_price", 0.0) or 0.0)
        
        expected_total = round(qty * unit, 2)
        calculated_subtotal += total
        
        if abs(expected_total - total) > 0.05:
            flags.append(f"Line item math error: {qty} x {unit} != {total}")

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
                    {"type": "text", "text": "Analyze this document and extract the JSON. Output MUST start with {"},
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

def process_pdf(pdf_bytes) -> dict:
    # Convert first page of PDF to image bytes using PyMuPDF (fitz)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    image_bytes = pix.tobytes("jpeg")
    return process_image(image_bytes)

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

if "Upload" in input_mode:
    uploaded_file = st.file_uploader("Upload document image or PDF...", type=["jpg", "jpeg", "png", "pdf"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            st.info("📄 PDF Uploaded successfully.")
        else:
            st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)
else:
    text_input = st.text_area("Paste document text here:", height=200)

if "result_data" not in st.session_state:
    st.session_state.result_data = None

if st.button("🚀 Process Document"):
    if "Upload" in input_mode and not uploaded_file:
        st.warning("Please upload a file first.")
    elif "Text" in input_mode and not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing document type and extracting data..."):
            if "Upload" in input_mode:
                file_bytes = uploaded_file.getvalue()
                if uploaded_file.type == "application/pdf":
                    st.session_state.result_data = process_pdf(file_bytes)
                else:
                    st.session_state.result_data = process_image(file_bytes)
            else:
                st.session_state.result_data = process_text(text_input)

if st.session_state.result_data is not None:
    result = st.session_state.result_data
    
    if "error" in result:
        st.error("Extraction Failed.")
        st.json(result)
    else:
        doc_type = result.get("detected_type", "unknown")
        st.success(f"✅ AI successfully identified this as a **{doc_type.upper()}** document.")

        if doc_type == "notes":
            notes = result.get("notes_data", {})
            
            st.subheader("📝 Exact Text Transcription")
            st.text_area("Transcription Output", notes.get("raw_transcription", "No text could be read."), height=150)
            
            st.subheader("📋 Summary")
            st.write(notes.get("summary", "No summary provided."))
            
            st.subheader("🎯 Action Items")
            items = notes.get("action_items", [])
            if not items:
                st.write("No action items found.")
            else:
                for idx, item in enumerate(items, 1):
                    st.markdown(f"**{idx}. {item.get('task')}** (Owner: {item.get('owner')})")

            text_content = f"--- EXTRACTED NOTES ---\n\n"
            text_content += f"SUMMARY:\n{notes.get('summary', 'No summary')}\n\n"
            text_content += "ACTION ITEMS:\n"
            if not items:
                text_content += "None\n"
            else:
                for idx, item in enumerate(items, 1):
                    text_content += f"{idx}. {item.get('task')} (Owner: {item.get('owner')}, Due: {item.get('due_date')})\n"
            text_content += f"\nRAW TRANSCRIPTION:\n{notes.get('raw_transcription', 'No transcription')}"

            st.download_button(
                label="📥 Download Notes as Text File (.txt)",
                data=text_content.encode("utf-8"),
                file_name="extracted_notes.txt",
                mime="text/plain",
            )

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

            st.download_button(
                label="📥 Download Invoice Data as JSON",
                data=json.dumps(result, indent=2).encode("utf-8"),
                file_name="extracted_invoice.json",
                mime="application/json",
            )

        else:
            st.warning("Could not determine document type.")
            st.json(result)