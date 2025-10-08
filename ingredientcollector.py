#!/usr/bin/env python3
"""
Robust Ingredient Collector with Advanced Lab Report Analysis

Features:
- Literal test result extraction from COA reports
- SambaNova LLM summarization
- Cross-verification with public APIs (OpenFoodFacts, USDA)
- Reasoning for inconsistencies and risks
- Structured JSON reports with trust/health scores
- Consumer health guidance and recommendations
"""

import os
import re
import json
import requests
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class IngredientCollector:
    """
    Advanced Ingredient Collector with Lab Report Analysis

    Processes Certificate of Analysis (COA) reports with:
    - Literal test result extraction
    - SambaNova LLM summarization
    - Cross-verification with public APIs
    - Reasoning for inconsistencies and risks
    - Structured JSON reports with scores and recommendations
    """

    def __init__(self):
        """Initialize IngredientCollector with API configurations."""
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY")
        )

        # SambaNova API configuration
        self.llm_base_url = os.getenv("LLM_BASE_URL") + "/chat/completions"
        self.llm_api_key = os.getenv("AGENT2_API_KEY")
        self.llm_model = os.getenv("AGENT2_MODEL")

        # Trusted labs list
        self.trusted_labs = [
            "fssai", "agmark", "iso", "nabl", "fao", "who",
            "usda", "fda", "efsa", "codex", "bis", "bipm"
        ]

        # Batch ID cache for duplicate detection
        self.batch_cache = set()

        # Safe ranges for common parameters
        self.safe_ranges = {
            "ph": {"min": 3.0, "max": 8.0, "typical_food": {"pickles": [3.0, 4.0], "dairy": [4.5, 7.0]}},
            "moisture": {"max": 15.0},  # percentage
            "protein": {"min": 0.0, "max": 100.0},
            "fat": {"min": 0.0, "max": 100.0},
            "carbohydrates": {"min": 0.0, "max": 100.0},
            "ash": {"max": 10.0},  # percentage
            "acidity": {"max": 5.0},  # percentage
            "total_plate_count": {"max": 10000},  # cfu/g
            "yeast_mold": {"max": 100},  # cfu/g
            "e_coli": {"max": 0},  # cfu/g
            "salmonella": {"max": 0},  # presence
            "lead": {"max": 1.0},  # ppm
            "arsenic": {"max": 0.1},  # ppm
            "mercury": {"max": 0.05},  # ppm
        }

    def process_coa_report(self, coa_path: str) -> Dict:
        """
        Main entry point for processing COA reports.

        Args:
            coa_path: Path to COA PDF/image/text file

        Returns:
            Structured JSON report with all analysis
        """
        print(f"Processing COA report: {coa_path}")

        # Step 1: Extract literal data
        raw_data = self.extract_literal_data(coa_path)
        if "error" in raw_data:
            return raw_data

        # Step 2: Call SambaNova for summarization
        llm_summary = self.generate_llm_summary(raw_data)

        # Step 3: Cross-verify with public APIs
        verification_data = self.cross_verify_with_apis(raw_data)

        # Step 4: Reasoning and risk analysis
        reasoning_results = self.perform_reasoning_analysis(raw_data, verification_data)

        # Step 5: Calculate scores and recommendations
        scores_and_recommendations = self.calculate_scores_and_recommendations(
            raw_data, reasoning_results, verification_data
        )

        # Step 6: Generate consumer health summary
        health_summary = self.generate_health_summary(raw_data, reasoning_results)

        # Compile final report
        final_report = {
            "processing_timestamp": datetime.now().isoformat(),
            "coa_file": os.path.basename(coa_path),
            "raw_extracted_data": raw_data,
            "llm_summary": llm_summary,
            "verification_data": verification_data,
            "reasoning_analysis": reasoning_results,
            "scores_and_recommendations": scores_and_recommendations,
            "consumer_health_summary": health_summary
        }

        return final_report

    def extract_literal_data(self, coa_path: str) -> Dict:
        """
        Extract literal test results, ingredients, and metadata from COA.

        Returns structured dictionary of key->value pairs.
        """
        # Extract text from file
        coa_text = self._extract_text_from_file(coa_path)
        if coa_text.startswith("Error"):
            return {"error": coa_text}

        extracted_data = {
            "batch_metadata": {},
            "test_results": {},
            "ingredients": [],
            "compliance_status": "unknown",
            "extraction_confidence": 0.0
        }

        # Extract batch metadata
        extracted_data["batch_metadata"] = self._extract_batch_metadata(coa_text)

        # Extract test results (parameter-value pairs)
        extracted_data["test_results"] = self._extract_test_results(coa_text)

        # Extract ingredients if present
        extracted_data["ingredients"] = self._extract_ingredients_from_coa(coa_text)

        # Determine compliance status
        extracted_data["compliance_status"] = self._determine_compliance_status(coa_text)

        # Calculate extraction confidence
        extracted_data["extraction_confidence"] = self._calculate_extraction_confidence(extracted_data)

        return extracted_data

    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF, image, or text file."""
        if not os.path.exists(file_path):
            return "Error: File not found."

        if file_path.lower().endswith('.pdf'):
            return self._extract_pdf_text(file_path)
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            return self._extract_image_text(file_path)
        else:
            # Assume text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        text = ""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except ImportError:
            return "Error: PyPDF2 not installed"
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"

        return text.strip() or "Error: Could not extract text from PDF"

    def _extract_image_text(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except ImportError:
            return "Error: PIL/pytesseract not installed for OCR"
        except Exception as e:
            return f"Error extracting image text: {str(e)}"

    def _extract_batch_metadata(self, coa_text: str) -> Dict:
        """Extract batch ID, sample date, lab name, signature."""
        metadata = {
            "batch_id": None,
            "sample_date": None,
            "analysis_date": None,
            "lab_name": None,
            "analyst_signature": None,
            "accreditation": None
        }

        # Batch ID patterns
        batch_patterns = [
            r'batch\s*(?:no\.?|number|id)?\s*[:\-]?\s*([A-Z0-9\-]+)',
            r'lot\s*(?:no\.?|number)?\s*[:\-]?\s*([A-Z0-9\-]+)',
            r'sample\s*(?:no\.?|number|id)?\s*[:\-]?\s*([A-Z0-9\-]+)'
        ]

        for pattern in batch_patterns:
            match = re.search(pattern, coa_text, re.IGNORECASE)
            if match:
                metadata["batch_id"] = match.group(1).strip()
                break

        # Date patterns
        date_patterns = [
            r'sample\s*date\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'analysis\s*date\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'date\s*of\s*analysis\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'report\s*date\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, coa_text, re.IGNORECASE)
            if match and not metadata["analysis_date"]:
                metadata["analysis_date"] = match.group(1).strip()

        # Lab name
        lab_patterns = [
            r'lab(?:oratory)?\s*[:\-]?\s*([^\n\r]+)',
            r'test(?:ing)?\s*lab(?:oratory)?\s*[:\-]?\s*([^\n\r]+)',
            r'conducted\s*(?:by|at)\s*([^\n\r]+)',
            r'analys(?:is|ed)\s*(?:by|at)\s*([^\n\r]+)'
        ]

        for pattern in lab_patterns:
            match = re.search(pattern, coa_text, re.IGNORECASE)
            if match:
                lab_name = match.group(1).strip()
                # Clean up common artifacts
                lab_name = re.sub(r'[^\w\s&\-\.]', '', lab_name).strip()
                if len(lab_name) > 3:
                    metadata["lab_name"] = lab_name
                    break

        # Analyst signature
        signature_patterns = [
            r'analyst\s*[:\-]?\s*([^\n\r]+)',
            r'signature\s*[:\-]?\s*([^\n\r]+)',
            r'approved\s*(?:by)?\s*[:\-]?\s*([^\n\r]+)',
            r'checked\s*(?:by)?\s*[:\-]?\s*([^\n\r]+)'
        ]

        for pattern in signature_patterns:
            match = re.search(pattern, coa_text, re.IGNORECASE)
            if match:
                signature = match.group(1).strip()
                if len(signature) > 2:
                    metadata["analyst_signature"] = signature
                    break

        return metadata

    def _extract_test_results(self, coa_text: str) -> Dict:
        """Extract parameter-value pairs from test results."""
        test_results = {}

        # Common test parameter patterns
        param_patterns = {
            # Physical parameters
            "moisture_content": r'moisture\s*(?:content)?\s*[:\-]?\s*([\d.]+)\s*%?',
            "ph": r'pH\s*[:\-]?\s*([\d.]+)',
            "acidity": r'acidity\s*[:\-]?\s*([\d.]+)\s*%?',
            "total_solids": r'total\s*sol(?:uble)?\s*solids?\s*[:\-]?\s*([\d.]+)\s*%?',
            "ash_content": r'ash\s*(?:content)?\s*[:\-]?\s*([\d.]+)\s*%?',

            # Nutritional parameters
            "protein": r'protein\s*[:\-]?\s*([\d.]+)\s*%?',
            "fat": r'fat\s*[:\-]?\s*([\d.]+)\s*%?',
            "carbohydrates": r'carbohydrates?\s*[:\-]?\s*([\d.]+)\s*%?',
            "fiber": r'fiber\s*(?:content)?\s*[:\-]?\s*([\d.]+)\s*%?',
            "calories": r'calor(?:ies?|ific\s*value)\s*[:\-]?\s*([\d.]+)',

            # Microbiological parameters
            "total_plate_count": r'total\s*plate\s*count\s*[:\-]?\s*([\d.]+\s*(?:x\s*10\^\d+)?)\s*cfu/?g',
            "yeast_mold": r'yeast\s*(?:and|&)\s*mold\s*[:\-]?\s*([\d.]+\s*(?:x\s*10\^\d+)?)\s*cfu/?g',
            "e_coli": r'e\.?\s*coli\s*[:\-]?\s*([\d.]+\s*(?:x\s*10\^\d+)?)\s*cfu/?g',
            "salmonella": r'salmonella\s*[:\-]?\s*(negative|positive|<[^>]+>|absent|present)',
            "staphylococcus": r'staph(?:ylococcus)?\s*[:\-]?\s*([\d.]+\s*(?:x\s*10\^\d+)?)\s*cfu/?g',

            # Heavy metals
            "lead": r'lead\s*[:\-]?\s*([\d.]+)\s*(?:ppm|mg/kg|µg/g)',
            "arsenic": r'arsenic\s*[:\-]?\s*([\d.]+)\s*(?:ppm|mg/kg|µg/g)',
            "mercury": r'mercury\s*[:\-]?\s*([\d.]+)\s*(?:ppm|mg/kg|µg/g)',
            "cadmium": r'cadmium\s*[:\-]?\s*([\d.]+)\s*(?:ppm|mg/kg|µg/g)',

            # Pesticides (simplified)
            "pesticide_residues": r'pesticide\s*residues?\s*[:\-]?\s*(within\s*limits|exceeded|compliant|non[\-\s]compliant)',

            # Preservatives
            "sodium_benzoate": r'sodium\s*benzoate\s*[:\-]?\s*([\d.]+)\s*(?:ppm|mg/kg)',
            "potassium_sorbate": r'potassium\s*sorbate\s*[:\-]?\s*([\d.]+)\s*(?:ppm|mg/kg)',
            "bha": r'bha\s*[:\-]?\s*([\d.]+)\s*(?:ppm|mg/kg)',
            "bht": r'bht\s*[:\-]?\s*([\d.]+)\s*(?:ppm|mg/kg)'
        }

        for param, pattern in param_patterns.items():
            match = re.search(pattern, coa_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Convert scientific notation if present
                if 'x10^' in value.lower():
                    try:
                        # Simple conversion (e.g., "3.2 x 10^3" -> 3200)
                        parts = re.split(r'\s*x\s*10\^\s*', value, flags=re.IGNORECASE)
                        if len(parts) == 2:
                            base = float(parts[0])
                            exponent = int(parts[1])
                            value = str(base * (10 ** exponent))
                    except:
                        pass  # Keep original if conversion fails

                test_results[param] = value

        return test_results

    def _extract_ingredients_from_coa(self, coa_text: str) -> List[str]:
        """Extract ingredients list from COA if present."""
        ingredients = []

        # Look for ingredients section
        ingredients_match = re.search(
            r'ingredients?\s*[:\-]?\s*(.*?)(?:\n\s*(?:batch|sample|test|analysis|$))',
            coa_text,
            re.IGNORECASE | re.DOTALL
        )

        if ingredients_match:
            ingredients_text = ingredients_match.group(1).strip()
            # Split by common separators
            ingredients = re.split(r'[,;]\s*|\sand\s|\sor\s', ingredients_text)
            ingredients = [ing.strip() for ing in ingredients if ing.strip() and len(ing.strip()) > 2]

        return ingredients

    def _determine_compliance_status(self, coa_text: str) -> str:
        """Determine overall compliance status from COA text."""
        compliance_indicators = {
            "compliant": ["pass", "complies", "satisfactory", "within limits", "acceptable", "conforms"],
            "non_compliant": ["fail", "not comply", "out of limits", "exceeded", "unacceptable", "non-conforming"]
        }

        coa_lower = coa_text.lower()

        # Check for non-compliant indicators first
        for indicator in compliance_indicators["non_compliant"]:
            if indicator in coa_lower:
                return "non_compliant"

        # Check for compliant indicators
        for indicator in compliance_indicators["compliant"]:
            if indicator in coa_lower:
                return "compliant"

        return "unknown"

    def _calculate_extraction_confidence(self, extracted_data: Dict) -> float:
        """Calculate confidence score for data extraction."""
        confidence = 0.0
        max_confidence = 5.0

        # Batch metadata completeness
        metadata = extracted_data.get("batch_metadata", {})
        if metadata.get("batch_id"): confidence += 1.0
        if metadata.get("analysis_date"): confidence += 0.8
        if metadata.get("lab_name"): confidence += 0.7
        if metadata.get("analyst_signature"): confidence += 0.5

        # Test results presence
        test_results = extracted_data.get("test_results", {})
        if len(test_results) > 0: confidence += 1.0
        if len(test_results) >= 3: confidence += 0.5

        # Compliance status
        if extracted_data.get("compliance_status") != "unknown": confidence += 0.5

        return min(confidence / max_confidence, 1.0)

    def generate_llm_summary(self, raw_data: Dict) -> Dict:
        """Generate concise summary using SambaNova LLM."""
        try:
            # Prepare prompt
            prompt = self._build_summary_prompt(raw_data)

            # Call SambaNova API
            response = self._call_sambanova_api(prompt)

            if "error" in response:
                return {"error": response["error"]}

            # Parse response
            summary_text = response.get("content", "")

            # Split into factual summary and interpretation
            parts = summary_text.split("\n\n", 1)
            factual_summary = parts[0].strip() if parts else summary_text
            interpretation = parts[1].strip() if len(parts) > 1 else ""

            return {
                "factual_summary": factual_summary,
                "producer_interpretation": interpretation,
                "raw_llm_response": summary_text
            }

        except Exception as e:
            return {"error": f"LLM summarization failed: {str(e)}"}

    def _build_summary_prompt(self, raw_data: Dict) -> str:
        """Build prompt for LLM summarization."""
        metadata = raw_data.get("batch_metadata", {})
        test_results = raw_data.get("test_results", {})

        prompt = f"""Analyze this Certificate of Analysis (COA) data and provide a summary in two parts:

PART 1 - FACTUAL SUMMARY (40-100 words):
- Batch ID: {metadata.get('batch_id', 'N/A')}
- Lab: {metadata.get('lab_name', 'N/A')}
- Analysis Date: {metadata.get('analysis_date', 'N/A')}
- Overall Status: {raw_data.get('compliance_status', 'unknown')}
- Key numeric results: {', '.join([f'{k}: {v}' for k, v in list(test_results.items())[:5]])}

PART 2 - PRODUCER/BUYER INTERPRETATION:
Explain what this means for food producers and buyers in plain language.

COA DATA:
{json.dumps(raw_data, indent=2)}
"""

        return prompt

    def _call_sambanova_api(self, prompt: str) -> Dict:
        """Call SambaNova API for LLM processing."""
        try:
            headers = {
                "Authorization": f"Bearer {self.llm_api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.llm
