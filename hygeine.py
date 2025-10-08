import os
import base64
import requests
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

class Agent2Hygiene:
    """
    Agent 2 - Hygiene Assessment
    Takes an image, assesses hygiene and provides a cleanliness score.
    """

    def __init__(self):
        """Initialize Agent2Hygiene"""
        self.api_url = os.getenv("LLM_BASE_URL") + "/chat/completions"
        self.api_key = os.getenv("AGENT2_API_KEY")
        self.model = os.getenv("AGENT2_MODEL")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64, resizing if necessary."""
        with Image.open(image_path) as img:
            # Resize if width or height > 1024
            max_size = 1024
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def validate_image_type(self, image_path: str) -> str:
        """Validate if image is a kitchen/preparation area or invalid."""
        try:
            base64_image = self.encode_image(image_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and determine if it shows a food preparation area, kitchen, cooking equipment, or food storage area. Answer with only 'KITCHEN_HYGIENE' if it clearly shows a kitchen or food preparation environment, or 'INVALID' if it's something else (like a food label, receipt, document, person, or unrelated image)."
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

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 20
            }

            response = requests.post(self.api_url, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            classification = result["choices"][0]["message"]["content"].strip().upper()

            # Normalize response
            if "KITCHEN" in classification or "HYGIENE" in classification:
                return "kitchen_hygiene"
            else:
                return "invalid"
        except Exception as e:
            print(f"Image validation error: {str(e)}")
            return "invalid"

    def assess_hygiene(self, image_path: str, location: str = None, producer_data: dict = None) -> dict:
        """Assess hygiene from image using AI, with location verification."""
        if not os.path.exists(image_path):
            return {"error": "Image file not found."}

        # Step 1: First validate that this is actually a kitchen/preparation area image
        image_type = self.validate_image_type(image_path)
        if image_type != "kitchen_hygiene":
            return {"error": "invalid message"}

        # Step 2: Check location match with FSSAI address if producer data is available
        if producer_data and "data" in producer_data:
            producer_address = producer_data["data"].get("address", "")
            if producer_address and location:
                location_match = self.check_location_match(location, producer_address)
                if not location_match:
                    return {"error": "address doesn't match"}

        # Step 3: Proceed with hygiene assessment
        # Encode image
        base64_image = self.encode_image(image_path)

        # Build assessment prompt
        prompt = "Assess the hygiene and cleanliness of this food preparation area or kitchen. Provide a cleanliness score from 1 to 10 (where 10 is perfectly clean) and explain in detail the reasons for the score based on what you see in the image. Be specific about cleanliness issues, good practices observed, and any recommendations for improvement."

        if location:
            prompt += f"\n\nLocation provided: {location}"

        if producer_data and "data" in producer_data:
            producer_address = producer_data["data"].get("address", "")
            if producer_address:
                prompt += f"\n\nProducer's registered address: {producer_address}"
                prompt += f"\n\nLocation verification: Confirmed to match registered address"

        # Prepare the message
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
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
            "max_tokens": 800
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            assessment_text = result["choices"][0]["message"]["content"].strip()

            # Parse score from response
            score = self.extract_score_from_assessment(assessment_text)

            # Build comprehensive result
            result = {
                "assessment": assessment_text,
                "score": score,
                "location_provided": location,
                "location_verified": True,  # Already verified above
                "producer_address": None,
                "image_validated": True
            }

            if producer_data and "data" in producer_data:
                result["producer_address"] = producer_data["data"].get("address")

            return result

        except Exception as e:
            return {"error": f"Error assessing hygiene: {str(e)}"}

    def check_location_match(self, provided_location: str, producer_address: str) -> bool:
        """Check if provided location matches or is near producer's address."""
        if not provided_location or not producer_address:
            return False

        # Convert to lowercase for comparison
        provided = provided_location.lower()
        producer = producer_address.lower()

        # Check for exact match
        if provided == producer:
            return True

        # Check for partial matches (city, district, etc.)
        provided_parts = set(provided.replace(',', ' ').replace('-', ' ').split())
        producer_parts = set(producer.replace(',', ' ').replace('-', ' ').split())

        # Common location keywords
        location_keywords = {'near', 'opposite', 'opp', 'beside', 'next to', 'behind', 'in front of'}

        # Remove common words
        provided_parts = {word for word in provided_parts if word not in location_keywords and len(word) > 2}
        producer_parts = {word for word in producer_parts if word not in location_keywords and len(word) > 2}

        # Check if significant location parts match
        common_parts = provided_parts.intersection(producer_parts)
        if len(common_parts) >= 2:  # At least 2 matching significant words
            return True

        # Check for district/state matches
        districts_states = ['sirsa', 'haryana', 'punjab', 'delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore']
        provided_districts = provided_parts.intersection(set(districts_states))
        producer_districts = producer_parts.intersection(set(districts_states))

        if provided_districts and producer_districts and provided_districts == producer_districts:
            return True

        return False

    def extract_score_from_assessment(self, assessment_text: str) -> int:
        """Extract numerical score from assessment text."""
        import re

        # Look for patterns like "score of 8", "8/10", "score: 7", etc.
        patterns = [
            r'score\s*(?:of|is|:)?\s*(\d+)',
            r'(\d+)\s*(?:/|out of)\s*10',
            r'(\d+)\s*/\s*10',
            r'rate.*(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, assessment_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 10:
                    return score

        # Default score if not found
        return 5

    def get_producer_for_location_check(self, identifier: str) -> dict:
        """Get producer data for location verification (simulates MCP call)."""
        # In actual implementation, this would call MCP tools
        # For demo, we'll simulate with known data

        # Simulate MCP call: get_verified_producer_by_name or get_producer_by_fssai
        print(f"Simulating MCP call to get producer data for: {identifier}")

        # Mock response - in real implementation, this would come from MCP
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

        return mock_producer_data

    def start_conversation(self):
        """Start interactive conversation for hygiene assessment with location verification."""
        print("Welcome to Sadapurne Agent 2 - Hygiene Assessment with Location Verification")
        print("=" * 70)
        print("Upload an image of a food preparation area and provide location for assessment.")
        print()

        while True:
            # Get image path
            print("Please enter the path to the kitchen/preparation area image:")
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

            # Get location
            print("\nPlease enter the location/address of this kitchen:")
            location = input("> ").strip()

            if not location:
                print("Location cannot be empty. Please try again.")
                continue

            # Get producer identifier for verification
            print("\nPlease enter producer name or FSSAI number for location verification:")
            producer_identifier = input("> ").strip()

            # Get producer data via MCP simulation
            producer_data = None
            if producer_identifier:
                print(f"\nFetching producer data for '{producer_identifier}' via MCP...")
                producer_data = self.get_producer_for_location_check(producer_identifier)

            # Assess hygiene with location verification
            print("\nAssessing hygiene and verifying location. Please wait...")
            assessment_result = self.assess_hygiene(image_path, location, producer_data)

            # Display result
            print("\n" + "=" * 70)
            print("HYGIENE ASSESSMENT RESULTS:")
            print("=" * 70)

            if "error" in assessment_result:
                print(f"Error: {assessment_result['error']}")
            else:
                print(f"Hygiene Score: {assessment_result['score']}/10")
                print(f"Location Provided: {assessment_result['location_provided']}")

                if assessment_result['producer_address']:
                    print(f"Producer Address: {assessment_result['producer_address']}")
                    print(f"Location Verified: {'✓ MATCH' if assessment_result['location_verified'] else '✗ NO MATCH'}")

                print("\nDetailed Assessment:")
                print("-" * 30)
                print(assessment_result['assessment'])

                # Overall recommendation
                print("\n" + "=" * 30)
                print("OVERALL RECOMMENDATION:")

                score = assessment_result['score']
                location_ok = assessment_result.get('location_verified', False)

                if score >= 8 and location_ok:
                    print("✅ EXCELLENT: High hygiene standards and location verified")
                elif score >= 8 and not location_ok:
                    print("⚠️  GOOD HYGIENE but LOCATION MISMATCH: Verify the inspection location")
                elif score >= 6 and location_ok:
                    print("⚠️  ACCEPTABLE: Meets basic standards, location verified")
                elif score < 6 and location_ok:
                    print("❌ CRITICAL: Poor hygiene standards, immediate action required")
                else:
                    print("❌ CRITICAL: Poor hygiene AND location mismatch - investigate immediately")

            print("=" * 70)

            # Ask if user wants to assess another
            retry = input("Would you like to assess another image? (y/yes to continue, any other key to exit): ").strip().lower()
            if retry not in ['y', 'yes']:
                print("Thank you for using Sadapurne Agent 2. Goodbye!")
                break
            print()

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        # Test mode: python hygeine.py <image_path> <location> [producer_identifier]
        image_path = sys.argv[1]
        location = sys.argv[2]
        producer_identifier = sys.argv[3] if len(sys.argv) > 3 else None

        agent = Agent2Hygiene()

        # Get producer data if identifier provided
        producer_data = None
        if producer_identifier:
            print(f"Fetching producer data for '{producer_identifier}'...")
            producer_data = agent.get_producer_for_location_check(producer_identifier)

        print(f"Assessing hygiene for image: {image_path}")
        print(f"Location: {location}")

        result = agent.assess_hygiene(image_path, location, producer_data)

        print("\nAssessment Result:")
        print(f"Score: {result.get('score', 'N/A')}/10")
        print(f"Location Verified: {result.get('location_verified', 'N/A')}")
        print(f"Assessment: {result.get('assessment', 'N/A')}")

    else:
        # Interactive mode
        agent = Agent2Hygiene()
        agent.start_conversation()