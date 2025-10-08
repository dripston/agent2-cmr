import os
import base64
import requests
from dotenv import load_dotenv
from PIL import Image
import io
import re
import json

# LangGraph imports for the biological health recommender agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator

load_dotenv()

# State definition for LangGraph
class HealthAnalysisState(TypedDict):
    ingredients: list
    health_recommendations: dict
    who_should_eat: list
    who_should_not_eat: list
    ideal_expiry: str
    storage_instructions: list
    consumption_tips: list
    allergen_warnings: list
    nutritional_insights: dict

class Agent2FoodLabel:
    """
    Agent 2 - Food Label Extraction
    Takes an image path, extracts text from the food label using AI vision.
    No summaries, just the text as on the label.
    """

    def __init__(self):
        """Initialize Agent2FoodLabel"""
        self.api_url = os.getenv("LLM_BASE_URL") + "/chat/completions"
        self.api_key = os.getenv("AGENT2_API_KEY")
        self.model = os.getenv("AGENT2_MODEL")
        self.health_recommender = self._create_health_recommender_graph()

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64, resizing if necessary."""
        with Image.open(image_path) as img:
            # Convert to RGB if image has alpha channel (RGBA) to avoid JPEG issues
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparent images
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if width or height > 1024
            max_size = 1024
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Save to bytes as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def extract_label_text(self, image_path: str) -> str:
        """Extract text from food label image using AI."""
        if not os.path.exists(image_path):
            return "Error: Image file not found."

        # Encode image
        base64_image = self.encode_image(image_path)

        # Prepare the message
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the text from this food label image. Only return the text as it appears on the label, no summaries or additional comments."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1000
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error extracting label: {str(e)}"

    def extract_ingredients(self, label_text: str) -> list:
        """Extract ingredients list from label text."""
        # Look for ingredients section
        ingredients_patterns = [
            r'Ingredients?\s*[:\-]?\s*(.*?)(?:\n\s*(?:Nutritional|Directions|Allergen|Contains|Net\s*W|Manufacturer|FSSAI|Best\s*Before|Expiry|$))',
            r'Ingredients?\s*[:\-]?\s*(.*?)(?:\.\s*(?:Nutritional|Directions|Allergen|Contains|Net\s*W|Manufacturer|FSSAI|Best\s*Before|Expiry|$))',
            r'Contains\s*[:\-]?\s*(.*?)(?:\n\s*(?:Nutritional|Directions|Allergen|Net\s*W|Manufacturer|FSSAI|Best\s*Before|Expiry|$))'
        ]
        
        for pattern in ingredients_patterns:
            match = re.search(pattern, label_text, re.IGNORECASE | re.DOTALL)
            if match:
                ingredients_text = match.group(1).strip()
                # Clean up the text - remove extra whitespace and newlines
                ingredients_text = re.sub(r'\s+', ' ', ingredients_text)
                # Split by common separators
                ingredients = re.split(r'[,;]\s*|\sand\s|\sor\s', ingredients_text)
                ingredients = [ing.strip() for ing in ingredients if ing.strip() and len(ing.strip()) > 1]
                # Filter out any remaining section headers or artifacts
                filtered_ingredients = []
                for ing in ingredients:
                    # Skip if it looks like a section header
                    if not re.match(r'^(Nutritional|Directions|Allergen|Contains|Net\s*W|Manufacturer|FSSAI|Best\s*Before|Expiry)', ing, re.IGNORECASE):
                        filtered_ingredients.append(ing)
                return filtered_ingredients
        
        return []

    def check_fssai_compliance(self, label_text: str) -> dict:
        """Check FSSAI compliance of the food label and create guide report for incomplete details."""
        # Prepare the message for compliance check
        messages = [
            {
                "role": "user",
                "content": f"""Analyze this food label text for FSSAI compliance in India. Check if it contains all required elements:

MANDATORY REQUIREMENTS:
1. Product identity/name - clear product name
2. List of ingredients - complete ingredient list with quantities
3. Manufacturer/importer details - complete address and contact
4. FSSAI license number - valid 14-digit number
5. Date marking - MFD/Best Before/Expiry dates
6. Net quantity/weight - clear quantity declaration
7. Country of origin - if imported

OPTIONAL BUT RECOMMENDED:
8. Nutritional information - if making nutritional claims
9. Allergen declarations - if allergens present
10. Storage instructions - how to store the product

Label Text:
{label_text}

First, determine if this label has COMPLETE details (contains ALL mandatory requirements) or INCOMPLETE details (missing some mandatory requirements).

If COMPLETE: Provide compliance score (80-100) and standard analysis.
If INCOMPLETE: Create a "GUIDE REPORT" explaining exactly what is missing and why it failed compliance. Include specific recommendations for what needs to be added.

Format as JSON with keys: compliance_status, score, compliant_items, missing_items, guide_report (if incomplete), recommendations."""
            }
        ]

        # API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1000
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            analysis = result["choices"][0]["message"]["content"].strip()

            # Try to parse as JSON
            try:
                import json
                compliance_data = json.loads(analysis)

                # Check if this is an incomplete label that needs a guide report
                if compliance_data.get("compliance_status") == "incomplete":
                    # Enhance the guide report with more specific guidance
                    compliance_data["guide_report"] = self._enhance_guide_report(
                        compliance_data.get("guide_report", ""),
                        compliance_data.get("missing_items", [])
                    )

                return compliance_data
            except:
                # If not JSON, return structured response
                return {
                    "compliance_status": "unknown",
                    "score": 50,  # Default
                    "analysis": analysis,
                    "raw_response": analysis
                }
        except Exception as e:
            return {"error": f"Error checking compliance: {str(e)}"}

    def _enhance_guide_report(self, basic_report: str, missing_items: list) -> str:
        """Enhance the guide report with specific actionable recommendations."""
        enhanced_report = f"""
FOOD LABEL COMPLIANCE FAILURE GUIDE
=====================================

{basic_report}

SPECIFIC CORRECTIONS NEEDED:
"""

        corrections_guide = {
            "Product identity/name": """
- Add a clear, prominent product name at the top of the label
- Use simple, understandable language
- Avoid misleading or confusing names
- Include brand name if applicable""",

            "List of ingredients": """
- List all ingredients in descending order of weight
- Include quantities for major ingredients
- Use proper chemical names for additives/preservatives
- Declare allergens clearly (e.g., "Contains: Milk, Wheat")
- Group ingredients logically if needed""",

            "Manufacturer/importer details": """
- Include complete business address
- Add contact phone number and email
- Specify if manufacturer or importer
- Include pincode and city for verification""",

            "FSSAI license number": """
- Must be exactly 14 digits
- Format: XXXXXXXXYYYYYY (where XXXXXXXX is establishment code, YYYYYY is year)
- Display prominently on label
- Ensure license is active and valid""",

            "Date marking": """
- Include Manufacturing Date (MFD)
- Include Best Before or Expiry Date
- Use clear date format (DD/MM/YYYY)
- Specify what the date refers to""",

            "Net quantity/weight": """
- Declare net weight/volume clearly
- Use metric units (grams, kilograms, milliliters, liters)
- Include drained weight for canned foods
- Be accurate and not misleading""",

            "Country of origin": """
- Required for imported products
- State "Country of Origin: [Country Name]"
- Include import license number if applicable
- Verify with customs documentation"""
        }

        for item in missing_items:
            item_key = item.lower().strip()
            for correction_key, guidance in corrections_guide.items():
                if correction_key.lower() in item_key or item_key in correction_key.lower():
                    enhanced_report += f"\n{correction_key.upper()}:{guidance}"
                    break

        enhanced_report += """

GENERAL COMPLIANCE REQUIREMENTS:
================================
- Label must be in English and Hindi (for products sold in India)
- Font size must be legible (minimum 1.5mm height for key information)
- All mandatory information must be clearly visible
- No misleading claims or false information
- Regular compliance audits recommended

NEXT STEPS:
===========
1. Review current label against FSSAI guidelines
2. Consult with FSSAI licensed designer/printer
3. Test new label with compliance checker
4. Submit for FSSAI approval if required
5. Update all packaging with compliant labels

For detailed FSSAI labeling guidelines, visit: https://fssai.gov.in/
"""

        return enhanced_report

    def verify_label_with_producer(self, label_text: str, producer_data: dict) -> dict:
        """Verify label information matches producer data."""
        if not producer_data or "data" not in producer_data:
            return {"error": "Invalid producer data"}

        producer = producer_data["data"]

        # Extract key information from label
        label_fssai = self.extract_fssai_from_label(label_text)
        label_manufacturer = self.extract_manufacturer_from_label(label_text)

        # Compare with producer data
        verification = {
            "fssai_match": False,
            "manufacturer_match": False,
            "verification_details": {}
        }

        # Check FSSAI match
        producer_fssai = producer.get("fssai_license_number")
        if producer_fssai and label_fssai:
            verification["fssai_match"] = producer_fssai in label_fssai or label_fssai in producer_fssai
            verification["verification_details"]["fssai"] = {
                "producer": producer_fssai,
                "label": label_fssai,
                "match": verification["fssai_match"]
            }

        # Check manufacturer match
        producer_name = producer.get("name", "").lower()
        if producer_name and label_manufacturer:
            # Simple string matching - could be enhanced
            verification["manufacturer_match"] = producer_name in label_manufacturer.lower() or label_manufacturer.lower() in producer_name
            verification["verification_details"]["manufacturer"] = {
                "producer": producer_name,
                "label": label_manufacturer,
                "match": verification["manufacturer_match"]
            }

        # Overall verification
        verification["overall_verified"] = verification["fssai_match"] or verification["manufacturer_match"]

        return verification

    def extract_fssai_from_label(self, label_text: str) -> str:
        """Extract FSSAI license number from label text."""
        import re
        # Look for FSSAI patterns
        patterns = [
            r'FSSAI\s*(?:No\.?|License|Lic\.?)?\s*[:\-]?\s*(\d{14})',
            r'FSSAI\s*(\d{14})',
            r'License\s*No\.?\s*[:\-]?\s*(\d{14})'
        ]

        for pattern in patterns:
            match = re.search(pattern, label_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def extract_manufacturer_from_label(self, label_text: str) -> str:
        """Extract manufacturer name from label text."""
        import re
        # Look for manufacturer patterns
        patterns = [
            r'(?:Mfd\.?|Manufactured|Made)\s*(?:By|by)\s*:?\s*([^\n\r]+)',
            r'Manufacturer\s*[:\-]?\s*([^\n\r]+)',
            r'By\s*([^\n\r]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, label_text, re.IGNORECASE)
            if match:
                manufacturer = match.group(1).strip()
                # Clean up common endings
                manufacturer = re.split(r'(?:FSSAI|License|Address|Phone|Email)', manufacturer, flags=re.IGNORECASE)[0].strip()
                return manufacturer
        return ""

    def _create_health_recommender_graph(self):
        """Create LangGraph-based health recommender agent."""
        # Define the graph
        workflow = StateGraph(HealthAnalysisState)
        
        # Add nodes
        workflow.add_node("extract_ingredients", self._extract_ingredients_node)
        workflow.add_node("analyze_nutritional_profile", self._analyze_nutritional_profile_node)
        workflow.add_node("identify_allergens", self._identify_allergens_node)
        workflow.add_node("determine_suitability", self._determine_suitability_node)
        workflow.add_node("generate_recommendations", self._generate_recommendations_node)
        
        # Add edges
        workflow.add_edge("extract_ingredients", "analyze_nutritional_profile")
        workflow.add_edge("analyze_nutritional_profile", "identify_allergens")
        workflow.add_edge("identify_allergens", "determine_suitability")
        workflow.add_edge("determine_suitability", "generate_recommendations")
        
        # Set entry point
        workflow.set_entry_point("extract_ingredients")
        
        # Set finish point
        workflow.add_edge("generate_recommendations", END)
        
        return workflow.compile()

    def _extract_ingredients_node(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Extract ingredients from the label text."""
        # This node would typically receive the label text as input
        # For now, we'll just pass through the ingredients that were already extracted
        return state

    def _analyze_nutritional_profile_node(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Analyze the nutritional profile of the product."""
        # In a real implementation, this would analyze nutritional data from the label
        # For now, we'll create a placeholder
        state["nutritional_insights"] = {
            "calorie_density": "moderate",
            "macronutrient_balance": "balanced",
            " micronutrient_content": "limited information"
        }
        return state

    def _identify_allergens_node(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Identify potential allergens in the ingredients."""
        ingredients = state.get("ingredients", [])
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

        state["allergen_warnings"] = list(set(found_allergens))
        return state

    def _determine_suitability_node(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Determine who should and shouldn't eat this product."""
        allergens = state.get("allergen_warnings", [])
        ingredients = state.get("ingredients", [])
        
        # Who should eat
        who_should_eat = ["General population"]
        if not allergens:
            who_should_eat.append("People with no known food allergies")
        who_should_eat.append("Individuals monitoring nutritional intake")
        
        # Who should not eat
        who_should_not_eat = []
        if "milk" in allergens:
            who_should_not_eat.extend(["Lactose intolerant individuals", "People with milk allergies"])
        if "eggs" in allergens:
            who_should_not_eat.append("People with egg allergies")
        if "peanuts" in allergens:
            who_should_not_eat.append("People with peanut allergies")
        if "tree_nuts" in allergens:
            who_should_not_eat.append("People with tree nut allergies")
        if "wheat" in allergens:
            who_should_not_eat.extend(["People with celiac disease", "Individuals with gluten intolerance"])
        if "soy" in allergens:
            who_should_not_eat.append("People with soy allergies")
        if "fish" in allergens:
            who_should_not_eat.append("People with fish allergies")
        if "shellfish" in allergens:
            who_should_not_eat.append("People with shellfish allergies")
        if "sesame" in allergens:
            who_should_not_eat.append("People with sesame allergies")
        if "sulfites" in allergens:
            who_should_not_eat.append("People sensitive to sulfites")
            
        state["who_should_eat"] = who_should_eat
        state["who_should_not_eat"] = who_should_not_eat
        return state

    def _generate_recommendations_node(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Generate health recommendations and consumption tips."""
        # Storage instructions
        state["storage_instructions"] = [
            "Store in a cool, dry place",
            "Keep away from direct sunlight",
            "Ensure package is properly sealed after opening",
            "Consume within recommended timeframe"
        ]
        
        # Consumption tips
        state["consumption_tips"] = [
            "Consume in moderation as part of a balanced diet",
            "Check expiry date before consumption",
            "If you have food allergies, verify ingredients before eating",
            "Consult a healthcare provider if you have specific dietary concerns"
        ]
        
        # Ideal expiry (placeholder - would be extracted from label in real implementation)
        state["ideal_expiry"] = "Refer to the 'Best Before' or 'Expiry' date printed on the package"
        
        return state

    def generate_health_recommendations(self, label_text: str, ingredients: list) -> dict:
        """Generate comprehensive health recommendations using the LangGraph agent."""
        # Initialize the state with extracted ingredients
        initial_state = {
            "ingredients": ingredients,
            "health_recommendations": {},
            "who_should_eat": [],
            "who_should_not_eat": [],
            "ideal_expiry": "",
            "storage_instructions": [],
            "consumption_tips": [],
            "allergen_warnings": [],
            "nutritional_insights": {}
        }
        
        # Run the LangGraph workflow
        final_state = self.health_recommender.invoke(initial_state)
        
        # Enhance with LLM-based analysis
        health_analysis = self._get_llm_health_analysis(label_text, ingredients)
        final_state["health_recommendations"] = health_analysis
        
        return final_state

    def _get_llm_health_analysis(self, label_text: str, ingredients: list) -> dict:
        """Get health analysis from LLM using external resources."""
        # Prepare the message for health analysis
        messages = [
            {
                "role": "user",
                "content": f"""As a biological health expert, analyze this food product and provide comprehensive health recommendations. 
                Use external freely available resources to inform your analysis.

                Product Label Text:
                {label_text}

                Extracted Ingredients:
                {', '.join(ingredients) if ingredients else 'Not available'}

                Please provide:
                1. Who should eat this product
                2. Who should avoid this product
                3. Ideal expiry timeline
                4. How to eat/consume this product
                5. Storage instructions
                6. Any health warnings
                7. Nutritional insights
                8. Consumption tips

                Format your response as JSON with appropriate keys."""
            }
        ]

        # API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1000
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            analysis = result["choices"][0]["message"]["content"].strip()

            # Try to parse as JSON
            try:
                health_data = json.loads(analysis)
                return health_data
            except:
                # If not JSON, return structured response
                return {
                    "raw_analysis": analysis
                }
        except Exception as e:
            return {"error": f"Error generating health analysis: {str(e)}"}

    def start_conversation(self):
        """Start interactive conversation for food label extraction."""
        print("Welcome to Sadapurne Agent 2 - Food Label Extraction")
        print("=" * 55)
        print("Upload an image of a food label to extract the text.")
        print()

        while True:
            # Get image path
            print("Please enter the path to the food label image:")
            image_path = input("> ").strip()

            if not image_path:
                print("Image path cannot be empty. Please try again.")
                continue

            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                retry = input("Would you like to try again? (y/yes to retry, any other key to exit): ").strip().lower()
                if retry in ['y', 'yes']:
                    continue
                else:
                    print("Thank you for using Sadapurne Agent 2. Goodbye!")
                    break

            # Extract label text
            print("\nExtracting text from the image. Please wait...")
            extracted_text = self.extract_label_text(image_path)

            if extracted_text.startswith("Error"):
                print(f"\n{extracted_text}")
                continue

            # Display extracted text
            print("\n" + "=" * 50)
            print("EXTRACTED LABEL TEXT:")
            print(extracted_text)
            print("=" * 50)

            # Extract ingredients
            ingredients = self.extract_ingredients(extracted_text)
            print(f"\nExtracted Ingredients: {', '.join(ingredients) if ingredients else 'None found'}")

            # Check FSSAI compliance
            print("\nChecking FSSAI compliance...")
            compliance_result = self.check_fssai_compliance(extracted_text)

            print("\n" + "=" * 50)
            print("FSSAI COMPLIANCE ANALYSIS:")
            if "error" in compliance_result:
                print(f"Error: {compliance_result['error']}")
            else:
                score = compliance_result.get('score', 'N/A')
                print(f"Compliance Score: {score}/100")

                if 'compliant_items' in compliance_result:
                    print("✓ Compliant Items:")
                    for item in compliance_result['compliant_items']:
                        print(f"  - {item}")

                if 'missing_items' in compliance_result:
                    print("✗ Missing Items:")
                    for item in compliance_result['missing_items']:
                        print(f"  - {item}")

                if 'recommendations' in compliance_result:
                    print("Recommendations:")
                    for rec in compliance_result['recommendations']:
                        print(f"  - {rec}")
            print("=" * 50)

            # Extract FSSAI and manufacturer for verification
            label_fssai = self.extract_fssai_from_label(extracted_text)
            label_manufacturer = self.extract_manufacturer_from_label(extracted_text)

            print("\nExtracted from Label:")
            print(f"  FSSAI License: {label_fssai or 'Not found'}")
            print(f"  Manufacturer: {label_manufacturer or 'Not found'}")

            # MCP verification simulation
            print("\n--- MCP VERIFICATION PROCESS ---")
            print("To verify this label against producer database, the system would:")

            if label_fssai:
                print(f"1. Call MCP tool 'get_producer_by_fssai' with fssai_number='{label_fssai}'")
            else:
                print("1. No FSSAI found on label - cannot verify via FSSAI")

            if label_manufacturer:
                print(f"2. Call MCP tool 'get_verified_producer_by_name' with name='{label_manufacturer}'")
            else:
                print("2. No manufacturer found on label - cannot verify via name")

            print("3. Compare label information with producer database records")
            print("4. Generate verification report")

            # Simulate verification with available data
            print("\n--- SIMULATED VERIFICATION ---")
            print("Note: In actual implementation, MCP would return producer data for comparison")

            # For demo, show what the verification would look like
            mock_producer_data = {
                "status": "found",
                "data": {
                    "name": "Kings Roll",
                    "fssai_license_number": "20819019000744",
                    "certificate_type": "registration",
                    "business_type": "Food Vending Establishment",
                    "annual_income": 11000.0,
                    "issue_date": None,
                    "expiry_date": "13/12/2025",
                    "address": "NEAR PUNJAB BUS STAND VPO AND TEH KALANWALI DISTT SIRSA HARYANA"
                }
            }

            verification = self.verify_label_with_producer(extracted_text, mock_producer_data)

            print("VERIFICATION RESULTS:")
            print(f"  FSSAI Match: {'✓' if verification.get('fssai_match') else '✗'}")
            print(f"  Manufacturer Match: {'✓' if verification.get('manufacturer_match') else '✗'}")
            print(f"  Overall Verified: {'✓' if verification.get('overall_verified') else '✗'}")

            if verification.get('verification_details'):
                print("  Details:")
                for key, detail in verification['verification_details'].items():
                    match_status = "✓" if detail.get('match') else "✗"
                    print(f"    {key.title()}: {match_status} (Label: '{detail.get('label')}' vs Producer: '{detail.get('producer')}')")

            # Generate health recommendations using the biological health recommender agent
            print("\n" + "=" * 50)
            print("BIOLOGICAL HEALTH RECOMMENDATIONS:")
            health_recommendations = self.generate_health_recommendations(extracted_text, ingredients)
            
            if "error" in health_recommendations:
                print(f"Error generating health recommendations: {health_recommendations['error']}")
            else:
                # Display health recommendations
                recs = health_recommendations.get("health_recommendations", {})
                
                if recs.get("who_should_eat"):
                    print("\n✅ WHO SHOULD EAT THIS PRODUCT:")
                    for item in recs.get("who_should_eat", []):
                        print(f"  • {item}")
                
                if recs.get("who_should_not_eat"):
                    print("\n❌ WHO SHOULD AVOID THIS PRODUCT:")
                    for item in recs.get("who_should_not_eat", []):
                        print(f"  • {item}")
                
                if recs.get("ideal_expiry"):
                    print(f"\n📅 IDEAL EXPIRY: {recs.get('ideal_expiry')}")
                
                if recs.get("consumption_tips"):
                    print("\n🍽️  HOW TO EAT/CONSUME:")
                    for tip in recs.get("consumption_tips", []):
                        print(f"  • {tip}")
                
                if recs.get("storage_instructions"):
                    print("\n📦 STORAGE INSTRUCTIONS:")
                    for instruction in recs.get("storage_instructions", []):
                        print(f"  • {instruction}")
                
                if recs.get("health_warnings"):
                    print("\n⚠️  HEALTH WARNINGS:")
                    for warning in recs.get("health_warnings", []):
                        print(f"  • {warning}")
                
                if recs.get("nutritional_insights"):
                    print("\n📊 NUTRITIONAL INSIGHTS:")
                    for key, value in recs.get("nutritional_insights", {}).items():
                        print(f"  • {key.replace('_', ' ').title()}: {value}")

            # Ask if user wants to extract another
            retry = input("\nWould you like to extract text from another image? (y/yes to continue, any other key to exit): ").strip().lower()
            if retry not in ['y', 'yes']:
                print("Thank you for using Sadapurne Agent 2. Goodbye!")
                break
            print()

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Test mode with image path
        image_path = sys.argv[1]
        agent = Agent2FoodLabel()
        print(f"Testing with image: {image_path}")

        # Extract text
        extracted_text = agent.extract_label_text(image_path)
        print("Extracted text:")
        print(extracted_text)

        if not extracted_text.startswith("Error"):
            # Extract ingredients
            ingredients = agent.extract_ingredients(extracted_text)
            print(f"\nExtracted Ingredients: {', '.join(ingredients) if ingredients else 'None found'}")
            
            # Check compliance
            print("\nChecking FSSAI compliance...")
            compliance = agent.check_fssai_compliance(extracted_text)
            print("Compliance result:", compliance)

            # Extract key info
            fssai = agent.extract_fssai_from_label(extracted_text)
            manufacturer = agent.extract_manufacturer_from_label(extracted_text)
            print(f"\nExtracted FSSAI: {fssai}")
            print(f"Extracted Manufacturer: {manufacturer}")

            # Simulate MCP verification
            print("\n--- MCP VERIFICATION SIMULATION ---")
            if fssai:
                print(f"Would call: get_producer_by_fssai('{fssai}')")
            if manufacturer:
                print(f"Would call: get_verified_producer_by_name('{manufacturer}')")

            # Mock verification
            mock_producer = {
                "status": "found",
                "data": {
                    "name": "MAHENDRA INDUSTRIES",
                    "fssai_license_number": "20819019000744",
                    "certificate_type": "registration"
                }
            }
            verification = agent.verify_label_with_producer(extracted_text, mock_producer)
            print("Verification result:", verification)
            
            # Generate health recommendations
            print("\n--- HEALTH RECOMMENDATIONS ---")
            health_recs = agent.generate_health_recommendations(extracted_text, ingredients)
            print("Health recommendations:", health_recs)
    else:
        # Interactive mode
        agent = Agent2FoodLabel()
        agent.start_conversation()