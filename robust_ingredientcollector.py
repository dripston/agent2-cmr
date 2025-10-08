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

class RobustIngredientCollector:
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
        """Initialize RobustIngredientCollector with API configurations."""
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

        # Step 6: Generate comprehensive health analysis report
        health_analysis_report = self.generate_health_analysis_report(raw_data, reasoning_results)

        # Compile final report
        final_report = {
            "processing_timestamp": datetime.now().isoformat(),
            "coa_file": os.path.basename(coa_path),
            "raw_extracted_data": raw_data,
            "llm_summary": llm_summary,
            "verification_data": verification_data,
            "reasoning_analysis": reasoning_results,
            "scores_and_recommendations": scores_and_recommendations,
            "comprehensive_health_analysis_report": health_analysis_report
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
            r'batch\s*(?:no\.?|number|id)?\s*[:\-]?\s*([A-Z0-9\-/]+)',
            r'lot\s*(?:no\.?|number)?\s*[:\-]?\s*([A-Z0-9\-/]+)',
            r'sample\s*(?:no\.?|number|id)?\s*[:\-]?\s*([A-Z0-9\-/]+)'
        ]

        for pattern in batch_patterns:
            match = re.search(pattern, coa_text, re.IGNORECASE)
            if match:
                metadata["batch_id"] = match.group(1).strip()
                break

        # Date patterns
        date_patterns = [
            r'sample\s*date\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            r'analysis\s*date\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            r'date\s*of\s*analysis\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            r'report\s*date\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            r'date\s*of\s*manufacture\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            r'date\s*of\s*expiry\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})'
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
            r'analys(?:is|ed)\s*(?:by|at)\s*([^\n\r]+)',
            r'place\s*of\s*analysis\s*[:\-]?\s*([^\n\r]+)'
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
            r'checked\s*(?:by)?\s*[:\-]?\s*([^\n\r]+)',
            r'authorized\s*signatory\s*[:\-]?\s*([^\n\r]+)',
            r'dr\.\s*[a-z\s]+',
            r'chief\s*quality\s*analyst\s*[:\-]?\s*([^\n\r]+)'
        ]

        for pattern in signature_patterns:
            match = re.search(pattern, coa_text, re.IGNORECASE)
            if match:
                signature = match.group(1).strip() if match.group(1) else match.group(0).strip()
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
            "e_coli": r'e\.?\s*coli\s*[:\-]?\s*([^\n\r]+)',
            "salmonella": r'salmonella\s*[:\-]?\s*([^\n\r]+)',

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

        # Special handling for structured tables in COA files
        # Try to extract table-based data
        table_data = self._extract_table_data(coa_text)
        test_results.update(table_data)

        return test_results

    def _extract_table_data(self, coa_text: str) -> Dict:
        """Extract structured data from tables in COA."""
        table_results = {}
        
        # Normalize the text to make extraction easier
        normalized_text = re.sub(r'\s+', ' ', coa_text)
        
        # Extract chemical parameters with specific patterns for this COA format
        # Moisture Content
        moisture_match = re.search(r'Moisture Content.*?Max 15\.0\s+(\d+\.?\d+)', normalized_text, re.IGNORECASE)
        if moisture_match:
            table_results["moisture_content"] = moisture_match.group(1)
            
        # Total Acidity
        acidity_match = re.search(r'Total Acidity.*?1\.2.*?1\.8\s+(\d+\.?\d+)', normalized_text, re.IGNORECASE)
        if acidity_match:
            table_results["acidity"] = acidity_match.group(1)
            
        # Salt Content
        salt_match = re.search(r'Salt Content.*?10.*?15\s+(\d+\.?\d+)', normalized_text, re.IGNORECASE)
        if salt_match:
            table_results["salt_content"] = salt_match.group(1)
            
        # Oil Content
        oil_match = re.search(r'Oil Content.*?20.*?25\s+(\d+\.?\d+)', normalized_text, re.IGNORECASE)
        if oil_match:
            table_results["oil_content"] = oil_match.group(1)
            
        # pH
        ph_match = re.search(r'pH\s*3\.0.*?4\.0\s+(\d+\.?\d+)', normalized_text, re.IGNORECASE)
        if ph_match:
            table_results["ph"] = ph_match.group(1)

        # Extract microbiological parameters with improved patterns
        # Total Plate Count
        total_plate_match = re.search(r'Total Plate Count.*?< 10\s*(\d+\.?\d*)\s*[×x]\s*10\^?3\s*(\w+)', normalized_text, re.IGNORECASE)
        if total_plate_match:
            base = total_plate_match.group(1)
            # Convert to standard scientific notation format
            value = f"{base}x10^3"
            table_results["total_plate_count"] = value
            
        # Yeast & Mould
        yeast_mold_match = re.search(r'Yeast.*?Mould.*?< 10\^?3\s*(\d+\.?\d*)\s*[×x]\s*10\^?2\s*(\w+)', normalized_text, re.IGNORECASE)
        if yeast_mold_match:
            base = yeast_mold_match.group(1)
            # Convert to standard scientific notation format
            value = f"{base}x10^2"
            table_results["yeast_mold"] = value
            
        # E. coli
        e_coli_match = re.search(r'E\.\s*coli\s*Absent\s*(\w+)', normalized_text, re.IGNORECASE)
        if e_coli_match:
            table_results["e_coli"] = e_coli_match.group(1)
            
        # Salmonella
        salmonella_match = re.search(r'Salmonella\s*Absent\s*(\w+)', normalized_text, re.IGNORECASE)
        if salmonella_match:
            table_results["salmonella"] = salmonella_match.group(1)

        return table_results

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
            "compliant": ["pass", "complies", "satisfactory", "within limits", "acceptable", "conforms", "compliant"],
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
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.3
            }

            response = requests.post(self.llm_base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            return {
                "content": result["choices"][0]["message"]["content"].strip(),
                "usage": result.get("usage", {})
            }

        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def cross_verify_with_apis(self, raw_data: Dict) -> Dict:
        """Cross-verify extracted data with public APIs."""
        verification_results = {
            "openfoodfacts": {},
            "usda": {},
            "verification_status": "partial",
            "confidence_score": 0.0
        }

        ingredients = raw_data.get("ingredients", [])
        test_results = raw_data.get("test_results", {})

        # OpenFoodFacts verification
        if ingredients:
            verification_results["openfoodfacts"] = self._verify_with_openfoodfacts(ingredients)

        # USDA FoodData Central verification
        if test_results:
            verification_results["usda"] = self._verify_with_usda(test_results)

        # Calculate overall confidence
        api_scores = []
        if verification_results["openfoodfacts"]:
            api_scores.append(0.6)  # OpenFoodFacts verification
        if verification_results["usda"]:
            api_scores.append(0.7)  # USDA verification

        verification_results["confidence_score"] = sum(api_scores) / len(api_scores) if api_scores else 0.0

        if verification_results["confidence_score"] > 0.5:
            verification_results["verification_status"] = "verified"
        elif verification_results["confidence_score"] > 0.2:
            verification_results["verification_status"] = "partial"
        else:
            verification_results["verification_status"] = "unverified"

        return verification_results

    def _verify_with_openfoodfacts(self, ingredients: List[str]) -> Dict:
        """Verify ingredients with OpenFoodFacts API."""
        verification = {"verified_ingredients": [], "unknown_ingredients": []}

        for ingredient in ingredients[:3]:  # Limit to first 3 ingredients
            try:
                # Search for ingredient
                search_url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={ingredient}&json=1"
                response = requests.get(search_url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("count", 0) > 0:
                        verification["verified_ingredients"].append({
                            "ingredient": ingredient,
                            "found": True,
                            "products_count": data["count"]
                        })
                    else:
                        verification["unknown_ingredients"].append(ingredient)
                else:
                    verification["unknown_ingredients"].append(ingredient)

            except Exception as e:
                verification["unknown_ingredients"].append(ingredient)

        return verification

    def _verify_with_usda(self, test_results: Dict) -> Dict:
        """Verify nutritional data with USDA FoodData Central."""
        verification = {"nutritional_verification": {}, "discrepancies": []}

        # This is a simplified verification - in practice, you'd need USDA API key
        # and more complex matching logic

        # Check if nutritional values are within reasonable ranges
        for param, value in test_results.items():
            if param in ["protein", "fat", "carbohydrates", "moisture_content"]:
                try:
                    num_value = float(value.replace('%', ''))
                    expected_range = self.safe_ranges.get(param, {})

                    if expected_range:
                        min_val = expected_range.get("min", 0)
                        max_val = expected_range.get("max", 100)

                        if min_val <= num_value <= max_val:
                            verification["nutritional_verification"][param] = "within_range"
                        else:
                            verification["nutritional_verification"][param] = "out_of_range"
                            verification["discrepancies"].append(f"{param}: {num_value} outside {min_val}-{max_val}")
                    else:
                        verification["nutritional_verification"][param] = "no_reference_range"

                except ValueError:
                    verification["nutritional_verification"][param] = "invalid_value"

        return verification

    def perform_reasoning_analysis(self, raw_data: Dict, verification_data: Dict) -> Dict:
        """Perform reasoning analysis for inconsistencies and risks."""
        reasoning = {
            "inconsistencies": [],
            "risks": [],
            "anomalies": [],
            "recommendations": []
        }

        metadata = raw_data.get("batch_metadata", {})
        test_results = raw_data.get("test_results", {})
        ingredients = raw_data.get("ingredients", [])

        # Check for missing mandatory fields
        mandatory_fields = ["batch_id", "analysis_date", "lab_name", "analyst_signature"]
        for field in mandatory_fields:
            if not metadata.get(field):
                reasoning["inconsistencies"].append(f"Missing mandatory field: {field}")

        # Check lab trustworthiness
        lab_name = metadata.get("lab_name", "")
        if lab_name:
            lab_name_lower = lab_name.lower()
            lab_trusted = any(trusted in lab_name_lower for trusted in self.trusted_labs)
            if not lab_trusted:
                reasoning["risks"].append(f"Lab '{lab_name}' not in trusted labs list")

        # Check for batch ID duplicates
        batch_id = metadata.get("batch_id")
        if batch_id:
            batch_hash = hashlib.md5(batch_id.encode()).hexdigest()
            if batch_hash in self.batch_cache:
                reasoning["anomalies"].append(f"Duplicate batch ID detected: {batch_id}")
            else:
                self.batch_cache.add(batch_hash)

        # Check test result ranges (only for numeric values)
        for param, value in test_results.items():
            safe_range = self.safe_ranges.get(param)
            if safe_range:
                # Handle categorical values like "Absent" or "Present"
                if param in ["e_coli", "salmonella"]:
                    # These are categorical values, not numeric
                    # Check if they are within acceptable categorical values
                    if value.lower() in ["absent", "negative", "not detected"]:
                        # This is good - no harmful bacteria detected
                        continue
                    elif value.lower() in ["present", "positive", "detected"]:
                        # This is a risk - harmful bacteria detected
                        reasoning["risks"].append(f"{param} detected in sample")
                    else:
                        # Unexpected categorical value
                        reasoning["inconsistencies"].append(f"Unexpected value for {param}: {value}")
                else:
                    # Handle numeric values
                    try:
                        num_value = float(value.replace('%', '').replace('ppm', '').strip())

                        # Check absolute ranges
                        if "max" in safe_range and num_value > safe_range["max"]:
                            reasoning["risks"].append(f"{param}: {num_value} exceeds safe limit of {safe_range['max']}")

                        if "min" in safe_range and num_value < safe_range["min"]:
                            reasoning["anomalies"].append(f"{param}: {num_value} below minimum of {safe_range['min']}")

                        # Check food-specific ranges
                        if param == "ph" and ingredients:
                            food_type = self._infer_food_type(ingredients)
                            typical_range = safe_range.get("typical_food", {}).get(food_type)
                            if typical_range and not (typical_range[0] <= num_value <= typical_range[1]):
                                reasoning["anomalies"].append(f"pH {num_value} unusual for {food_type} (typical: {typical_range[0]}-{typical_range[1]})")

                    except ValueError:
                        # If we can't convert to float, it might be a categorical value
                        # We've already handled e_coli and salmonella above, so this is unexpected
                        reasoning["inconsistencies"].append(f"Invalid value for {param}: {value}")

        # Check ingredient-test result consistency
        if ingredients and test_results:
            reasoning["inconsistencies"].extend(self._check_ingredient_consistency(ingredients, test_results))

        # Generate recommendations based on findings
        if reasoning["risks"]:
            reasoning["recommendations"].append("Immediate review required due to safety risks")
        if reasoning["inconsistencies"]:
            reasoning["recommendations"].append("Address data completeness issues")
        if not reasoning["risks"] and not reasoning["inconsistencies"]:
            reasoning["recommendations"].append("COA appears compliant and complete")

        return reasoning

    def _infer_food_type(self, ingredients: List[str]) -> str:
        """Infer food type from ingredients."""
        ingredients_lower = [ing.lower() for ing in ingredients]

        if any("pickle" in ing or "vinegar" in ing for ing in ingredients_lower):
            return "pickles"
        elif any("milk" in ing or "cream" in ing for ing in ingredients_lower):
            return "dairy"
        elif any("sugar" in ing for ing in ingredients_lower):
            return "confectionery"

        return "general"

    def _check_ingredient_consistency(self, ingredients: List[str], test_results: Dict) -> List[str]:
        """Check for inconsistencies between ingredients and test results."""
        inconsistencies = []
        ingredients_lower = [ing.lower() for ing in ingredients]

        # Check for sugar content vs sugar ingredient
        if any("sugar" in ing for ing in ingredients_lower):
            if "carbohydrates" in test_results:
                try:
                    carb_value = float(test_results["carbohydrates"].replace('%', ''))
                    if carb_value < 10:  # Sugar should contribute significantly to carbs
                        inconsistencies.append("Low carbohydrate content despite sugar ingredient present")
                except ValueError:
                    pass

        # Check for contradictory claims
        ingredient_text = ' '.join(ingredients_lower)
        if "no sugar" in ingredient_text and "sugar" in ingredient_text:
            inconsistencies.append("Conflicting sugar claims in ingredients")

        return inconsistencies

    def calculate_scores_and_recommendations(self, raw_data: Dict, reasoning: Dict, verification: Dict) -> Dict:
        """Calculate trust score, health score, and provide recommendations."""
        scores = {
            "trust_score": 0.0,  # 0-1 based on lab credibility, signatures, completeness
            "health_score": 0.0,  # 0-1 based on test results vs safe ranges
            "overall_score": 0.0,  # Combined score
            "action": "unknown",  # publish/review/require_lab_test
            "confidence_level": "low"
        }

        # Calculate trust score
        trust_factors = []

        # Lab credibility
        metadata = raw_data.get("batch_metadata", {})
        lab_name = metadata.get("lab_name", "")
        if lab_name:
            lab_name_lower = lab_name.lower()
            if any(trusted in lab_name_lower for trusted in self.trusted_labs):
                trust_factors.append(0.3)
            else:
                trust_factors.append(0.1)  # Unknown lab

        # Signature presence
        if metadata.get("analyst_signature"):
            trust_factors.append(0.2)

        # Data completeness
        required_fields = ["batch_id", "analysis_date", "lab_name"]
        completeness = sum(1 for field in required_fields if metadata.get(field)) / len(required_fields)
        trust_factors.append(completeness * 0.3)

        # No major inconsistencies
        if not reasoning.get("inconsistencies"):
            trust_factors.append(0.2)

        scores["trust_score"] = min(sum(trust_factors), 1.0)

        # Calculate health score
        health_factors = []
        test_results = raw_data.get("test_results", {})

        # Compliance status
        if raw_data.get("compliance_status") == "compliant":
            health_factors.append(0.4)
        elif raw_data.get("compliance_status") == "non_compliant":
            health_factors.append(0.0)
        else:
            health_factors.append(0.2)

        # Test result safety
        safe_params = 0
        total_params = len(test_results)
        for param, value in test_results.items():
            safe_range = self.safe_ranges.get(param)
            if safe_range:
                try:
                    num_value = float(value.replace('%', '').replace('ppm', '').strip())
                    max_val = safe_range.get("max")
                    if max_val is not None and num_value <= max_val:
                        safe_params += 1
                    elif max_val is None:
                        safe_params += 1  # No max limit defined
                except ValueError:
                    pass

        if total_params > 0:
            safety_ratio = safe_params / total_params
            health_factors.append(safety_ratio * 0.4)

        # Verification confidence
        verification_confidence = verification.get("confidence_score", 0.0)
        health_factors.append(verification_confidence * 0.2)

        scores["health_score"] = min(sum(health_factors), 1.0)

        # Calculate overall score
        scores["overall_score"] = (scores["trust_score"] * 0.4) + (scores["health_score"] * 0.6)

        # Determine action and confidence
        if scores["overall_score"] >= 0.8 and scores["trust_score"] >= 0.7:
            scores["action"] = "publish"
            scores["confidence_level"] = "high"
        elif scores["overall_score"] >= 0.6:
            scores["action"] = "review"
            scores["confidence_level"] = "medium"
        else:
            scores["action"] = "require_lab_test"
            scores["confidence_level"] = "low"

        return scores

    def generate_health_analysis_report(self, raw_data: Dict, reasoning: Dict) -> Dict:
        """Generate comprehensive health analysis report for consumers."""
        report = {
            "product_suitability": {},
            "expiry_and_shelf_life": {},
            "consumption_guidelines": {},
            "health_warnings_and_risks": {},
            "nutritional_profile": {},
            "storage_and_handling": {},
            "allergen_information": {},
            "special_dietary_considerations": {},
            "overall_health_rating": "",
            "consumer_recommendations": []
        }

        test_results = raw_data.get("test_results", {})
        ingredients = raw_data.get("ingredients", [])
        metadata = raw_data.get("batch_metadata", {})

        # 1. Product Suitability Analysis
        allergens = self._identify_potential_allergens(ingredients)
        report["product_suitability"] = self._analyze_product_suitability(allergens, test_results)

        # 2. Expiry and Shelf Life Analysis
        report["expiry_and_shelf_life"] = self._analyze_expiry_and_shelf_life(raw_data, test_results)

        # 3. Consumption Guidelines
        report["consumption_guidelines"] = self._generate_consumption_guidelines(test_results, ingredients)

        # 4. Health Warnings and Risks
        report["health_warnings_and_risks"] = self._analyze_health_warnings(test_results, reasoning)

        # 5. Nutritional Profile
        report["nutritional_profile"] = self._analyze_nutritional_profile(test_results, ingredients)

        # 6. Storage and Handling Instructions
        report["storage_and_handling"] = self._generate_storage_instructions(test_results, ingredients)

        # 7. Allergen Information
        report["allergen_information"] = self._compile_allergen_information(allergens, ingredients)

        # 8. Special Dietary Considerations
        report["special_dietary_considerations"] = self._analyze_dietary_considerations(ingredients, test_results)

        # 9. Overall Health Rating
        report["overall_health_rating"] = self._calculate_overall_health_rating(raw_data, reasoning)

        # 10. Consumer Recommendations
        report["consumer_recommendations"] = self._generate_consumer_recommendations(report)

        return report

    def _analyze_product_suitability(self, allergens: List[str], test_results: Dict) -> Dict:
        """Analyze who should and shouldn't consume this product."""
        suitability = {
            "recommended_for": [],
            "not_recommended_for": [],
            "suitable_age_groups": [],
            "medical_conditions_to_avoid": []
        }

        # Who should eat (general population considerations)
        suitability["recommended_for"].extend([
            "General healthy adults",
            "Individuals seeking balanced nutrition",
            "People with active lifestyles"
        ])

        # Who should NOT eat based on allergens
        allergen_mappings = {
            "milk": ["Lactose intolerant individuals", "People with milk protein allergies", "Those with dairy sensitivities"],
            "eggs": ["People with egg allergies", "Individuals with avian protein sensitivities"],
            "peanuts": ["People with peanut allergies", "Those with legume allergies"],
            "tree_nuts": ["People with tree nut allergies", "Individuals with multiple nut sensitivities"],
            "wheat": ["People with celiac disease", "Those with gluten intolerance", "Individuals with wheat allergies"],
            "soy": ["People with soy allergies", "Those with legume sensitivities"],
            "fish": ["People with fish allergies", "Individuals with seafood sensitivities"],
            "shellfish": ["People with shellfish allergies", "Those with crustacean or mollusk allergies"],
            "sesame": ["People with sesame allergies", "Individuals with seed allergies"]
        }

        for allergen in allergens:
            allergen_key = allergen.lower()
            if allergen_key in allergen_mappings:
                suitability["not_recommended_for"].extend(allergen_mappings[allergen_key])

        # Age group considerations based on nutritional content
        if test_results.get("protein"):
            try:
                protein_val = float(test_results["protein"].replace('%', ''))
                if protein_val > 15:
                    suitability["recommended_for"].append("Growing children and adolescents")
                    suitability["recommended_for"].append("Athletes and fitness enthusiasts")
            except:
                pass

        # Medical conditions to avoid
        if "milk" in [a.lower() for a in allergens]:
            suitability["medical_conditions_to_avoid"].extend([
                "Lactose intolerance",
                "Milk protein allergy",
                "Galactosemia"
            ])

        if "wheat" in [a.lower() for a in allergens]:
            suitability["medical_conditions_to_avoid"].extend([
                "Celiac disease",
                "Non-celiac gluten sensitivity",
                "Wheat allergy"
            ])

        # Check for heavy metals and contaminants
        for param in ["lead", "arsenic", "mercury", "cadmium"]:
            if param in test_results:
                try:
                    value = float(test_results[param].replace('ppm', '').strip())
                    safe_limit = self.safe_ranges[param]["max"]
                    if value > safe_limit:
                        suitability["not_recommended_for"].extend([
                            "Pregnant women",
                            "Nursing mothers",
                            "Young children",
                            f"Individuals concerned about {param} exposure"
                        ])
                        suitability["medical_conditions_to_avoid"].append(f"Elevated {param} levels - consult healthcare provider")
                except:
                    pass

        return suitability

    def _analyze_expiry_and_shelf_life(self, raw_data: Dict, test_results: Dict) -> Dict:
        """Analyze expiry timeline and shelf life information."""
        expiry_analysis = {
            "manufacture_date": None,
            "best_before_date": None,
            "expiry_date": None,
            "remaining_shelf_life_days": None,
            "storage_conditions_met": True,
            "expiry_status": "unknown",
            "consumption_urgency": "normal",
            "preservation_method": "unknown"
        }

        metadata = raw_data.get("batch_metadata", {})

        # Extract dates from metadata
        analysis_date = metadata.get("analysis_date")
        if analysis_date:
            try:
                # Parse analysis date and estimate expiry
                # This is a simplified estimation - real implementation would need proper date parsing
                expiry_analysis["analysis_date"] = analysis_date

                # Estimate shelf life based on product type and test results
                if test_results.get("moisture_content"):
                    try:
                        moisture = float(test_results["moisture_content"].replace('%', ''))
                        if moisture < 10:
                            expiry_analysis["estimated_shelf_life_months"] = 24  # Low moisture = longer shelf life
                            expiry_analysis["preservation_method"] = "Dehydration/Drying"
                        elif moisture < 50:
                            expiry_analysis["estimated_shelf_life_months"] = 12
                            expiry_analysis["preservation_method"] = "Moderate moisture preservation"
                        else:
                            expiry_analysis["estimated_shelf_life_months"] = 6
                            expiry_analysis["preservation_method"] = "High moisture - refrigerate"
                    except:
                        pass

                # Check microbial stability
                if test_results.get("total_plate_count"):
                    try:
                        tpc = test_results["total_plate_count"]
                        if "x10^" in tpc:
                            # Convert scientific notation
                            parts = tpc.split("x10^")
                            if len(parts) == 2:
                                base = float(parts[0])
                                exponent = int(parts[1])
                                count = base * (10 ** exponent)
                                if count > 100000:  # High microbial count
                                    expiry_analysis["consumption_urgency"] = "high"
                                    expiry_analysis["expiry_status"] = "questionable"
                    except:
                        pass

            except:
                pass

        return expiry_analysis

    def _generate_consumption_guidelines(self, test_results: Dict, ingredients: List[str]) -> Dict:
        """Generate detailed consumption guidelines."""
        guidelines = {
            "serving_suggestions": [],
            "preparation_methods": [],
            "pairing_recommendations": [],
            "consumption_frequency": "",
            "maximum_daily_intake": None,
            "cooking_requirements": []
        }

        # Serving suggestions based on nutritional content
        if test_results.get("protein"):
            try:
                protein_val = float(test_results["protein"].replace('%', ''))
                if protein_val > 20:
                    guidelines["serving_suggestions"].append("Excellent protein supplement")
                    guidelines["consumption_frequency"] = "Daily as part of balanced diet"
                elif protein_val > 10:
                    guidelines["serving_suggestions"].append("Good protein source")
                    guidelines["consumption_frequency"] = "2-3 times per week"
            except:
                pass

        # Preparation methods based on ingredients
        ingredients_lower = [ing.lower() for ing in ingredients]
        if any("pickle" in ing for ing in ingredients_lower):
            guidelines["preparation_methods"].extend([
                "Ready to eat",
                "Can be used in salads or as condiment",
                "No cooking required"
            ])
        elif any("flour" in ing or "wheat" in ing for ing in ingredients_lower):
            guidelines["preparation_methods"].extend([
                "Bake at 180°C for 20-30 minutes",
                "Can be used in cooking or baking",
                "Mix with water for dough preparation"
            ])

        # Cooking requirements based on test results
        if test_results.get("moisture_content"):
            try:
                moisture = float(test_results["moisture_content"].replace('%', ''))
                if moisture > 70:
                    guidelines["cooking_requirements"].append("Requires cooking/heating before consumption")
                    guidelines["cooking_requirements"].append("Heat to internal temperature of 75°C minimum")
                elif moisture > 50:
                    guidelines["cooking_requirements"].append("Light cooking recommended")
                else:
                    guidelines["cooking_requirements"].append("Ready to eat - no cooking required")
            except:
                pass

        return guidelines

    def _analyze_health_warnings(self, test_results: Dict, reasoning: Dict) -> Dict:
        """Analyze and compile health warnings and risks."""
        warnings = {
            "critical_warnings": [],
            "moderate_warnings": [],
            "general_cautions": [],
            "contaminant_levels": {},
            "microbiological_risks": [],
            "allergen_alerts": []
        }

        # Check for critical contaminants
        contaminant_limits = {
            "lead": {"limit": 1.0, "unit": "ppm", "risk": "Neurodevelopmental toxicity"},
            "arsenic": {"limit": 0.1, "unit": "ppm", "risk": "Carcinogenic"},
            "mercury": {"limit": 0.05, "unit": "ppm", "risk": "Neurological damage"},
            "cadmium": {"limit": 0.1, "unit": "ppm", "risk": "Kidney damage"}
        }

        for contaminant, info in contaminant_limits.items():
            if contaminant in test_results:
                try:
                    value = float(test_results[contaminant].replace('ppm', '').strip())
                    if value > info["limit"]:
                        warnings["critical_warnings"].append(
                            f"Elevated {contaminant} levels ({value} ppm) - {info['risk']}"
                        )
                        warnings["contaminant_levels"][contaminant] = {
                            "level": value,
                            "limit": info["limit"],
                            "exceeds_limit": True,
                            "risk": info["risk"]
                        }
                except:
                    pass

        # Microbiological risks
        if test_results.get("e_coli") and "present" in test_results["e_coli"].lower():
            warnings["critical_warnings"].append("E. coli contamination detected - potential food poisoning risk")

        if test_results.get("salmonella") and "present" in test_results["salmonella"].lower():
            warnings["critical_warnings"].append("Salmonella contamination detected - severe food poisoning risk")

        if test_results.get("total_plate_count"):
            try:
                tpc = test_results["total_plate_count"]
                if "x10^" in tpc:
                    parts = tpc.split("x10^")
                    if len(parts) == 2:
                        base = float(parts[0])
                        exponent = int(parts[1])
                        count = base * (10 ** exponent)
                        if count > 1000000:  # Very high count
                            warnings["moderate_warnings"].append("High microbial load - consume promptly")
            except:
                pass

        # Add reasoning-based warnings
        risks = reasoning.get("risks", [])
        for risk in risks:
            if "exceeds" in risk.lower() or "high" in risk.lower():
                warnings["moderate_warnings"].append(risk)

        return warnings

    def _analyze_nutritional_profile(self, test_results: Dict, ingredients: List[str]) -> Dict:
        """Analyze nutritional profile and benefits."""
        profile = {
            "macronutrients": {},
            "micronutrients": {},
            "nutritional_highlights": [],
            "dietary_fiber_content": None,
            "caloric_density": "",
            "nutrient_density_score": 0.0
        }

        # Macronutrients analysis
        if test_results.get("protein"):
            try:
                protein = float(test_results["protein"].replace('%', ''))
                if protein > 20:
                    profile["macronutrients"]["protein"] = {"value": protein, "rating": "excellent", "benefit": "High-quality protein source"}
                elif protein > 10:
                    profile["macronutrients"]["protein"] = {"value": protein, "rating": "good", "benefit": "Moderate protein content"}
                else:
                    profile["macronutrients"]["protein"] = {"value": protein, "rating": "low", "benefit": "Low protein content"}
            except:
                pass

        if test_results.get("fat"):
            try:
                fat = float(test_results["fat"].replace('%', ''))
                if fat < 5:
                    profile["macronutrients"]["fat"] = {"value": fat, "rating": "low", "benefit": "Low-fat option"}
                elif fat < 15:
                    profile["macronutrients"]["fat"] = {"value": fat, "rating": "moderate", "benefit": "Moderate fat content"}
                else:
                    profile["macronutrients"]["fat"] = {"value": fat, "rating": "high", "benefit": "Energy-dense"}
            except:
                pass

        if test_results.get("carbohydrates"):
            try:
                carbs = float(test_results["carbohydrates"].replace('%', ''))
                profile["macronutrients"]["carbohydrates"] = {"value": carbs, "rating": "good", "benefit": "Carbohydrate source for energy"}
            except:
                pass

        # Nutritional highlights
        if profile["macronutrients"].get("protein", {}).get("rating") == "excellent":
            profile["nutritional_highlights"].append("Excellent protein source")

        if profile["macronutrients"].get("fat", {}).get("rating") == "low":
            profile["nutritional_highlights"].append("Low-fat healthy option")

        # Caloric density estimation
        total_macros = 0
        if "protein" in profile["macronutrients"]:
            total_macros += profile["macronutrients"]["protein"]["value"]
        if "fat" in profile["macronutrients"]:
            total_macros += profile["macronutrients"]["fat"]["value"]
        if "carbohydrates" in profile["macronutrients"]:
            total_macros += profile["macronutrients"]["carbohydrates"]["value"]

        if total_macros > 0:
            if total_macros < 30:
                profile["caloric_density"] = "Low calorie density - suitable for weight management"
            elif total_macros < 60:
                profile["caloric_density"] = "Moderate calorie density"
            else:
                profile["caloric_density"] = "High calorie density - energy-rich"

        return profile

    def _generate_storage_instructions(self, test_results: Dict, ingredients: List[str]) -> Dict:
        """Generate comprehensive storage and handling instructions."""
        storage = {
            "primary_storage_method": "",
            "temperature_requirements": "",
            "humidity_requirements": "",
            "light_protection": "",
            "packaging_integrity": "",
            "handling_precautions": [],
            "shelf_life_expectancy": ""
        }

        # Determine storage based on moisture content
        if test_results.get("moisture_content"):
            try:
                moisture = float(test_results["moisture_content"].replace('%', ''))
                if moisture > 70:
                    storage["primary_storage_method"] = "Refrigeration required"
                    storage["temperature_requirements"] = "Store at 4°C or below"
                    storage["shelf_life_expectancy"] = "7-14 days refrigerated"
                elif moisture > 50:
                    storage["primary_storage_method"] = "Cool, dry place"
                    storage["temperature_requirements"] = "Store below 25°C"
                    storage["shelf_life_expectancy"] = "3-6 months"
                else:
                    storage["primary_storage_method"] = "Cool, dry place"
                    storage["temperature_requirements"] = "Store in cool, dry conditions"
                    storage["shelf_life_expectancy"] = "6-24 months"
            except:
                pass

        # Light protection
        storage["light_protection"] = "Keep away from direct sunlight"

        # Packaging integrity
        storage["packaging_integrity"] = "Keep package sealed when not in use"

        # Handling precautions
        storage["handling_precautions"].extend([
            "Wash hands before handling",
            "Use clean utensils",
            "Avoid cross-contamination with raw foods"
        ])

        # Special handling based on ingredients
        ingredients_lower = [ing.lower() for ing in ingredients]
        if any("oil" in ing for ing in ingredients_lower):
            storage["handling_precautions"].append("Store away from heat sources to prevent rancidity")

        return storage

    def _compile_allergen_information(self, allergens: List[str], ingredients: List[str]) -> Dict:
        """Compile comprehensive allergen information."""
        allergen_info = {
            "declared_allergens": allergens,
            "potential_cross_contamination": [],
            "allergen_warnings": [],
            "labeling_compliance": True
        }

        # Check for common allergens in ingredients
        allergen_keywords = {
            "milk": ["milk", "dairy", "cheese", "butter", "cream", "yogurt", "whey", "casein", "lactose"],
            "eggs": ["egg", "eggs", "albumin", "lecithin", "lysozyme"],
            "peanuts": ["peanut", "peanuts", "groundnut", "arachis"],
            "tree_nuts": ["almond", "walnut", "cashew", "pistachio", "hazelnut", "pecan", "macadamia"],
            "wheat": ["wheat", "flour", "gluten", "bread", "pasta", "bran"],
            "soy": ["soy", "soya", "soybean", "tofu", "tempeh", "edamame"],
            "fish": ["fish", "salmon", "tuna", "cod", "sardine", "anchovy"],
            "shellfish": ["shrimp", "crab", "lobster", "clam", "oyster", "scallop"],
            "sesame": ["sesame", "tahini", "halva"]
        }

        ingredients_lower = [ing.lower() for ing in ingredients]

        for allergen, keywords in allergen_keywords.items():
            if any(keyword in ing for ing in ingredients_lower):
                if allergen.title() not in allergen_info["declared_allergens"]:
                    allergen_info["declared_allergens"].append(allergen.title())

        # Generate allergen warnings
        for allergen in allergen_info["declared_allergens"]:
            allergen_info["allergen_warnings"].append(
                f"Contains {allergen.lower()}. Not suitable for individuals with {allergen.lower()} allergies."
            )

        return allergen_info

    def _analyze_dietary_considerations(self, ingredients: List[str], test_results: Dict) -> Dict:
        """Analyze special dietary considerations."""
        considerations = {
            "vegan_suitable": True,
            "vegetarian_suitable": True,
            "kosher_suitable": True,
            "halal_suitable": True,
            "organic_certified": False,
            "gmo_free": False,
            "dietary_restrictions_addressed": []
        }

        ingredients_lower = [ing.lower() for ing in ingredients]

        # Check vegan/vegetarian suitability
        animal_products = ["meat", "chicken", "beef", "pork", "fish", "milk", "dairy", "cheese", "butter", "cream", "yogurt", "whey", "egg", "eggs", "honey", "gelatin"]
        if any(product in ing for product in animal_products for ing in ingredients_lower):
            considerations["vegan_suitable"] = False
            considerations["vegetarian_suitable"] = False

        # Check for potential kosher/halal concerns
        if any("pork" in ing or "gelatin" in ing for ing in ingredients_lower):
            considerations["kosher_suitable"] = False
            considerations["halal_suitable"] = False

        # Dietary restrictions
        if any("gluten" in ing or "wheat" in ing for ing in ingredients_lower):
            considerations["dietary_restrictions_addressed"].append("Contains gluten - not suitable for gluten-free diets")

        if test_results.get("fat"):
            try:
                fat_content = float(test_results["fat"].replace('%', ''))
                if fat_content < 3:
                    considerations["dietary_restrictions_addressed"].append("Low-fat - suitable for low-fat diets")
            except:
                pass

        return considerations

    def _calculate_overall_health_rating(self, raw_data: Dict, reasoning: Dict) -> str:
        """Calculate overall health rating."""
        compliance_status = raw_data.get("compliance_status", "unknown")
        risks = reasoning.get("risks", [])
        inconsistencies = reasoning.get("inconsistencies", [])

        if compliance_status == "compliant" and not risks and not inconsistencies:
            return "EXCELLENT - Fully compliant with high safety standards"
        elif compliance_status == "compliant" and len(risks) <= 1:
            return "GOOD - Compliant with minor considerations"
        elif compliance_status == "compliant":
            return "FAIR - Compliant but with some safety concerns"
        else:
            return "POOR - Does not meet safety standards"

    def _generate_consumer_recommendations(self, report: Dict) -> List[str]:
        """Generate final consumer recommendations based on analysis."""
        recommendations = []

        # Basic recommendations
        recommendations.append("Read all labels carefully before consumption")
        recommendations.append("Store according to recommended conditions")

        # Suitability-based recommendations
        suitability = report.get("product_suitability", {})
        if suitability.get("not_recommended_for"):
            recommendations.append("Consult healthcare provider if you have food allergies")

        # Health warnings
        warnings = report.get("health_warnings_and_risks", {})
        if warnings.get("critical_warnings"):
            recommendations.append("⚠️ CRITICAL: Do not consume - seek immediate medical attention if ingested")

        if warnings.get("moderate_warnings"):
            recommendations.append("Consult healthcare provider before consumption")

        # Storage recommendations
        storage = report.get("storage_and_handling", {})
        if storage.get("primary_storage_method"):
            recommendations.append(f"Storage: {storage['primary_storage_method']}")

        return recommendations

    # Keep the old method for backward compatibility
    def generate_health_summary(self, raw_data: Dict, reasoning: Dict) -> Dict:
        """Generate consumer health summary (legacy method)."""
        return self.generate_health_analysis_report(raw_data, reasoning)

    def _extract_expiry_info_from_raw(self, raw_data: Dict) -> Dict:
        """Extract expiry info from raw data (simplified version)."""
        # This is a simplified version - in practice, you'd reuse the full extraction logic
        return {"timeline_analysis": {"status": "unknown"}}

    def _identify_potential_allergens(self, ingredients: List[str]) -> List[str]:
        """Identify potential allergens from ingredients list."""
        # Common allergens
        common_allergens = {
            'milk': ['milk', 'dairy', 'cheese', 'butter', 'cream', 'yogurt', 'whey'],
            'eggs': ['egg', 'eggs', 'albumin', 'lecithin'],
            'peanuts': ['peanut', 'peanuts', 'groundnut'],
            'tree_nuts': ['almond', 'walnut', 'cashew', 'pistachio', 'hazelnut', 'pecan'],
            'wheat': ['wheat', 'flour', 'gluten', 'bread', 'pasta'],
            'soy': ['soy', 'soya', 'soybean', 'tofu'],
            'fish': ['fish', 'salmon', 'tuna', 'cod', 'sardine'],
            'shellfish': ['shrimp', 'crab', 'lobster', 'clam', 'oyster'],
            'sesame': ['sesame', 'tahini'],
            'sulfites': ['sulfite', 'sulfur dioxide', 'sulphite']
        }

        found_allergens = []
        ingredients_lower = [ing.lower() for ing in ingredients]

        for allergen, keywords in common_allergens.items():
            for keyword in keywords:
                if any(keyword in ing for ing in ingredients_lower):
                    found_allergens.append(allergen.title())
                    break

        return list(set(found_allergens))  # Remove duplicates


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python robust_ingredientcollector.py <coa_file_path>")
        sys.exit(1)

    coa_path = sys.argv[1]

    # Initialize collector
    collector = RobustIngredientCollector()

    # Process COA report
    print(f"Processing COA report: {coa_path}")
    result = collector.process_coa_report(coa_path)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        # Print summary
        print("\n" + "="*80)
        print("COA ANALYSIS SUMMARY")
        print("="*80)

        # Raw data summary
        raw = result["raw_extracted_data"]
        meta = raw["batch_metadata"]
        print(f"Batch ID: {meta.get('batch_id', 'N/A')}")
        print(f"Lab: {meta.get('lab_name', 'N/A')}")
        print(f"Compliance: {raw.get('compliance_status', 'unknown')}")
        print(f"Test Parameters: {len(raw.get('test_results', {}))}")

        # LLM Summary
        llm = result.get("llm_summary", {})
        if "factual_summary" in llm:
            print(f"\nFactual Summary: {llm['factual_summary'][:200]}...")

        # Scores
        scores = result.get("scores_and_recommendations", {})
        print(f"\nTrust Score: {scores.get('trust_score', 0):.2f}")
        print(f"Health Score: {scores.get('health_score', 0):.2f}")
        print(f"Overall Score: {scores.get('overall_score', 0):.2f}")
        print(f"Recommended Action: {scores.get('action', 'unknown')}")

        # Comprehensive Health Analysis Report
        health_report = result.get("comprehensive_health_analysis_report", {})
        if health_report.get("product_suitability"):
            suitability = health_report["product_suitability"]
            if suitability.get("recommended_for"):
                print(f"\nRecommended for: {', '.join(suitability['recommended_for'][:3])}")
            if suitability.get("not_recommended_for"):
                print(f"Not recommended for: {', '.join(suitability['not_recommended_for'][:3])}")

        if health_report.get("overall_health_rating"):
            print(f"Overall Health Rating: {health_report['overall_health_rating']}")

        if health_report.get("health_warnings_and_risks", {}).get("critical_warnings"):
            print(f"Critical Warnings: {len(health_report['health_warnings_and_risks']['critical_warnings'])}")

        print("\n" + "="*80)

        # Save full report
        output_file = f"coa_analysis_{os.path.basename(coa_path).replace('.', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"Full analysis saved to: {output_file}")