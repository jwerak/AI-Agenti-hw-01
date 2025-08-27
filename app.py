import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from main import GeminiReactAgent, sum_two_numbers, multiply_two_numbers
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Gemini agent
agent = GeminiReactAgent()


@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "message": "Gemini ReAct Agent API is running",
            "version": "1.0.0",
        }
    )


@app.route("/api/query", methods=["POST"])
def query_agent():
    """
    Main endpoint to query the Gemini ReAct agent.
    Expects JSON payload with 'query' field.
    """
    try:
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify({"error": "Missing required field: query"}), 400

        query = data["query"]

        if not query.strip():
            return jsonify({"error": "Query cannot be empty"}), 400

        logger.info(f"Processing query: {query}")

        # Run the agent
        result = agent.run(query)

        return jsonify({"query": query, "result": result, "status": "success"})

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return (
            jsonify({"error": f"Internal server error: {str(e)}", "status": "error"}),
            500,
        )


@app.route("/api/sum", methods=["POST"])
def sum_numbers():
    """
    Direct endpoint for summing two numbers.
    Expects JSON payload with 'x' and 'y' fields.
    """
    try:
        data = request.get_json()

        if not data or "x" not in data or "y" not in data:
            return jsonify({"error": "Missing required fields: x and y"}), 400

        x = data["x"]
        y = data["y"]

        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return jsonify({"error": "x and y must be numbers"}), 400

        result = sum_two_numbers(int(x), int(y))

        return jsonify(
            {
                "operation": "sum",
                "x": x,
                "y": y,
                "result": result["result"],
                "status": "success",
            }
        )

    except Exception as e:
        logger.error(f"Error in sum operation: {str(e)}")
        return (
            jsonify({"error": f"Internal server error: {str(e)}", "status": "error"}),
            500,
        )


@app.route("/api/multiply", methods=["POST"])
def multiply_numbers():
    """
    Direct endpoint for multiplying two numbers.
    Expects JSON payload with 'x' and 'y' fields.
    """
    try:
        data = request.get_json()

        if not data or "x" not in data or "y" not in data:
            return jsonify({"error": "Missing required fields: x and y"}), 400

        x = data["x"]
        y = data["y"]

        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return jsonify({"error": "x and y must be numbers"}), 400

        result = multiply_two_numbers(int(x), int(y))

        return jsonify(
            {
                "operation": "multiply",
                "x": x,
                "y": y,
                "result": result["result"],
                "status": "success",
            }
        )

    except Exception as e:
        logger.error(f"Error in multiply operation: {str(e)}")
        return (
            jsonify({"error": f"Internal server error: {str(e)}", "status": "error"}),
            500,
        )


@app.route("/api/functions")
def list_functions():
    """List available functions."""
    return jsonify(
        {
            "functions": [
                {
                    "name": "sum_two_numbers",
                    "description": "Sum two numbers",
                    "parameters": ["x", "y"],
                },
                {
                    "name": "multiply_two_numbers",
                    "description": "Multiply two numbers",
                    "parameters": ["x", "y"],
                },
            ],
            "endpoints": [
                {
                    "path": "/api/query",
                    "method": "POST",
                    "description": "Query the Gemini ReAct agent with natural language",
                },
                {
                    "path": "/api/sum",
                    "method": "POST",
                    "description": "Direct sum operation",
                },
                {
                    "path": "/api/multiply",
                    "method": "POST",
                    "description": "Direct multiply operation",
                },
            ],
        }
    )


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found", "status": "error"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error", "status": "error"}), 500


if __name__ == "__main__":
    # Check if GEMINI_API_KEY is set
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY environment variable is not set!")

    # Run the Flask app
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    logger.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
