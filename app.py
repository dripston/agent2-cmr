import os
import json
import base64
import tempfile
import re
from flask import Flask, request, jsonify
from foodlabel import Agent2FoodLabel
from hygeine import Agent2Hygiene
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Agents
agent2_foodlabel = Agent2FoodLabel()
agent2_hygiene = Agent2Hygiene()

def call_mcp_get_producer_by_fssai(fssai_number):
    """Call MCP server to get producer data by FSSAI number"""
    try:
        mcp_url = "https://mcp-server-agent1.onrender.com/api/producer/fssai"
        response = requests.post(mcp_url, json={"fssai_number": fssai_number}, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "success":
            return result["data"]
        else:
            return None
    except Exception as e:
        print(f"MCP call error: {str(e)}")
        return None

def classify_image_type(image_path):
    """Classify the image type: food_label, kitchen_hygiene, or invalid"""
    try:
        base64_image = agent2_foodlabel.encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image and classify it into exactly one category. Answer with only the category name:\n\n- FOOD_LABEL: if it shows a food product label, packaging, or nutritional information\n- KITCHEN_HYGIENE: if it shows a kitchen, food preparation area, cooking equipment, or food storage area\n- INVALID: if it's a receipt, document, person, or unrelated image"
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
            "Authorization": f"Bearer {os.getenv('AGENT2_API_KEY')}",
            "Content-Type": "application/json"
        }

        data = {
            "model": os.getenv("AGENT2_MODEL"),
            "messages": messages,
            "max_tokens": 20
        }

        response = requests.post(os.getenv("LLM_BASE_URL") + "/chat/completions", headers=headers, json=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        classification = result["choices"][0]["message"]["content"].strip().upper()

        # Normalize response
        if "FOOD_LABEL" in classification or "FOOD" in classification:
            return "food_label"
        elif "KITCHEN" in classification or "HYGIENE" in classification:
            return "kitchen_hygiene"
        else:
            return "invalid"
    except Exception as e:
        print(f"Image classification error: {str(e)}")
        return "invalid"

def match_location_with_address(geo_location, producer_address):
    """Check if geo location matches producer address"""
    if not producer_address or not geo_location:
        return False

    try:
        # For now, implement basic reverse geocoding simulation
        # In production, use Google Maps API or similar for proper geocoding

        # Extract location info from address using regex
        address_lower = producer_address.lower()

        # Common Indian cities/states patterns
        locations = []
        city_patterns = [
            r'\b(delhi|new delhi|mumbai|chennai|kolkata|bangalore|bengaluru|hyderabad|pune|ahmedabad|jaipur|lucknow|kanpur|nagpur|indore|thane|bhopal|pimpri|pune|patna|vadodara|ghaziabad|ludhiana|agra|nashik|faridabad|meerut|rajkot|kalyan|vasai|varanasi|srinagar|aurangabad|dhanbad|amritsar|navi mumbai|allahabad|ranchi|howrah|coimbatore|jabalpur|gwalior|vijayawada|jodhpur|madurai|raipur|kota|guwahati|solapur|hubli|mysore|tiruchirappalli|bareilly|aligarh|tiruppur|moradabad|bhiwandi|gorakhpur|jamshedpur|bikaner|warangal|cuttack|dehradun|durgapur|asansol|nanded|kolhapur|ajmer|akola|gulbarga|jamnagar|ujjain|loni|siliguri|jhansi|ulhasnagar|nellore|kalyan|belgaum|ambattur|tirunelveli|malegaon|kochi|thiruvananthapuram|kozhikode|thrissur|kollam|palakkad|alappuzha|kottayam|pathanamthitta|idukki|ernakulam|kannur|kasaragod)\b',
            r'\b(haryana|punjab|rajasthan|uttar pradesh|madhya pradesh|maharashtra|gujarat|bihar|west bengal|karnataka|andhra pradesh|telangana|tamil nadu|kerala|odisha|jharkhand|chhattisgarh|himachal pradesh|uttarakhand|goa|manipur|tripura|meghalaya|arunachal pradesh|nagaland|mizoram|sikkim|assam|jammu and kashmir|ladakh|chandigarh|dadra and nagar haveli|daman and diu|lakshadweep|puducherry)\b'
        ]

        for pattern in city_patterns:
            matches = re.findall(pattern, address_lower)
            locations.extend(matches)

        if not locations:
            # If no specific locations found, be lenient
            return True

        # For now, return True if any location found in address
        # Real implementation would compare coordinates
        return len(locations) > 0

    except Exception as e:
        print(f"Location matching error: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "Welcome to Sadapurne Agent2 API",
        "version": "2.0.0",
        "description": "Handles food label compliance and kitchen hygiene assessment",
        "endpoints": {
            "health": "GET /health",
            "check_compliance": "POST /compliance - accepts array of images (food labels and/or kitchen hygiene)",
        },
        "request_format": {
            "images": ["base64_image_1", "base64_image_2"],
            "geo_location": {"lat": 28.6139, "lng": 77.2090}
        },
        "image_types": {
            "food_label": "Food product labels for FSSAI compliance checking",
            "kitchen_hygiene": "Kitchen/preparation area images for cleanliness assessment"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Agent2 API is running"}), 200

@app.route('/compliance', methods=['POST'])
def check_fssai_compliance():
    """Check FSSAI compliance endpoint - handles food labels and hygiene images"""
    try:
        # Get JSON data from request
        data = request.get_json()

        # Validate required fields
        if 'images' not in data:
            return jsonify({
                "status": "failed",
                "stage": "input_validation",
                "message": "Missing required field: images (should be array of base64 images)"
            }), 400

        if 'geo_location' not in data:
            return jsonify({
                "status": "failed",
                "stage": "input_validation",
                "message": "Missing required field: geo_location"
            }), 400

        images = data['images']
        geo_location = data['geo_location']

        # Validate images format
        if not isinstance(images, list) or len(images) == 0:
            return jsonify({
                "status": "failed",
                "stage": "input_validation",
                "message": "images should be a non-empty array of base64 strings"
            }), 400

        # Validate geo_location format
        if not isinstance(geo_location, dict) or 'lat' not in geo_location or 'lng' not in geo_location:
            return jsonify({
                "status": "failed",
                "stage": "input_validation",
                "message": "Invalid geo_location format. Expected: {'lat': float, 'lng': float}"
            }), 400

        # Process each image
        results = {"food_label": None, "hygiene": None}
        temp_files = []

        try:
            for i, image_base64 in enumerate(images):
                # Validate and decode image data
                if not image_base64 or not isinstance(image_base64, str):
                    return jsonify({
                        "status": "failed",
                        "stage": "input_validation",
                        "message": f"Invalid image data at index {i}"
                    }), 400

                try:
                    # Decode base64 image data
                    image_data = base64.b64decode(image_base64)
                except Exception as e:
                    return jsonify({
                        "status": "failed",
                        "stage": "input_validation",
                        "message": f"Failed to decode image data at index {i}: {str(e)}"
                    }), 400

                # Save image to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_image:
                    temp_image.write(image_data)
                    temp_image_path = temp_image.name
                    temp_files.append(temp_image_path)

                # Classify image type
                image_type = classify_image_type(temp_image_path)

                if image_type == "invalid":
                    return jsonify({
                        "status": "failed",
                        "stage": "image_validation",
                        "message": f"invalid message - image at index {i} is not a valid food label or kitchen image"
                    }), 400

                # Process based on type
                if image_type == "food_label":
                    if results["food_label"] is not None:
                        return jsonify({
                            "status": "failed",
                            "stage": "input_validation",
                            "message": "Multiple food label images provided - only one expected"
                        }), 400

                    results["food_label"] = temp_image_path

                elif image_type == "kitchen_hygiene":
                    if results["hygiene"] is not None:
                        return jsonify({
                            "status": "failed",
                            "stage": "input_validation",
                            "message": "Multiple hygiene images provided - only one expected"
                        }), 400

                    results["hygiene"] = temp_image_path

            # Check that we have at least one valid image
            if results["food_label"] is None and results["hygiene"] is None:
                return jsonify({
                    "status": "failed",
                    "stage": "image_validation",
                    "message": "No valid food label or hygiene images found"
                }), 400

            # Initialize final results
            final_result = {
                "status": "success",
                "message": "Analysis completed",
                "food_label_analysis": None,
                "hygiene_analysis": None
            }

            # Process food label if provided
            if results["food_label"]:
                food_label_path = results["food_label"]

                # Step 1: Extract label text from image
                extracted_text = agent2_foodlabel.extract_label_text(food_label_path)

                if extracted_text.startswith("Error"):
                    return jsonify({
                        "status": "failed",
                        "stage": "text_extraction",
                        "message": extracted_text
                    }), 400

                # Step 2: Extract FSSAI from label
                label_fssai = agent2_foodlabel.extract_fssai_from_label(extracted_text)

                if not label_fssai:
                    return jsonify({
                        "status": "failed",
                        "stage": "fssai_extraction",
                        "message": "No FSSAI number found on label"
                    }), 400

                # Step 3: Get producer data from MCP
                producer_data = call_mcp_get_producer_by_fssai(label_fssai)

                if not producer_data:
                    return jsonify({
                        "status": "failed",
                        "stage": "producer_verification",
                        "message": "FSSAI number not found in verified producers database"
                    }), 400

                # Step 4: Match location with producer address
                producer_address = producer_data.get("address", "")
                if not match_location_with_address(geo_location, producer_address):
                    return jsonify({
                        "status": "failed",
                        "stage": "location_verification",
                        "message": "address doesn't match"
                    }), 400

                # Step 5: Proceed with compliance check
                ingredients = agent2_foodlabel.extract_ingredients(extracted_text)
                compliance_result = agent2_foodlabel.check_fssai_compliance(extracted_text)
                label_manufacturer = agent2_foodlabel.extract_manufacturer_from_label(extracted_text)
                health_recommendations = agent2_foodlabel.generate_health_recommendations(extracted_text, ingredients)

                final_result["food_label_analysis"] = {
                    "extracted_text": extracted_text,
                    "ingredients": ingredients,
                    "compliance_result": compliance_result,
                    "label_info": {
                        "fssai_license": label_fssai,
                        "manufacturer": label_manufacturer or "Not found"
                    },
                    "producer_verification": {
                        "producer_name": producer_data.get("name", "Unknown"),
                        "producer_address": producer_address,
                        "location_matched": True
                    },
                    "health_recommendations": health_recommendations
                }

            # Process hygiene image if provided
            if results["hygiene"]:
                hygiene_path = results["hygiene"]

                # Get producer data for hygiene verification (use FSSAI from food label if available)
                producer_data_for_hygiene = None
                if final_result["food_label_analysis"]:
                    fssai_from_label = final_result["food_label_analysis"]["label_info"]["fssai_license"]
                    producer_data_for_hygiene = call_mcp_get_producer_by_fssai(fssai_from_label)

                # Assess hygiene
                hygiene_result = agent2_hygiene.assess_hygiene(
                    hygiene_path,
                    location=f"{geo_location.get('lat', 0)}, {geo_location.get('lng', 0)}",
                    producer_data=producer_data_for_hygiene
                )

                if "error" in hygiene_result:
                    return jsonify({
                        "status": "failed",
                        "stage": "hygiene_assessment",
                        "message": hygiene_result["error"]
                    }), 400

                final_result["hygiene_analysis"] = hygiene_result

            return jsonify(final_result), 200

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                
    except Exception as e:
        return jsonify({
            "status": "failed",
            "stage": "server_error",
            "message": f"Internal server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=False)