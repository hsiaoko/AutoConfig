#!/bin/bash
# Recommendation Service Script
# Starts a simple HTTP service for configuration recommendation

set -e

echo "=============================================="
echo "AutoConfig Recommendation Service"
echo "=============================================="

# Default parameters
MODEL_DIR=${1:-data/models}
HOST=${2:-0.0.0.0}
PORT=${3:-8080}

echo ""
echo "Parameters:"
echo "  Model Dir: $MODEL_DIR"
echo "  Host:      $HOST"
echo "  Port:      $PORT"
echo ""

# Check if models exist
if [ ! -f "$MODEL_DIR/time_model.pkl" ] || [ ! -f "$MODEL_DIR/cost_model.pkl" ]; then
    echo "Warning: Models not found in $MODEL_DIR"
    echo "Run training first: ./scripts/train_pipeline.sh"
    echo ""
    echo "Starting with heuristic mode..."
    echo ""
fi

# Start the service
echo "Starting recommendation service..."
echo ""

python -c "
from autoconfig.online.recommender import Recommender
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Initialize recommender
recommender = Recommender(model_dir='$MODEL_DIR', auto_train=False)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_loaded': recommender.models_loaded})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query_name = data.get('query')
    graph_features = data.get('graph_features')
    top_n = data.get('top_n', 3)
    
    if not query_name or not graph_features:
        return jsonify({'error': 'Missing query or graph_features'}), 400
    
    result = recommender.recommend(query_name, graph_features, top_n=top_n)
    return jsonify(result)

@app.route('/what-if', methods=['POST'])
def what_if():
    data = request.json
    query_name = data.get('query')
    graph_features = data.get('graph_features')
    config = data.get('config')
    
    if not all([query_name, graph_features, config]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    result = recommender.what_if(query_name, graph_features, config)
    return jsonify(result)

if __name__ == '__main__':
    print(f'Starting server on $HOST:$PORT')
    app.run(host='$HOST', port=$PORT, debug=False)
"

echo ""
echo "Service stopped."
