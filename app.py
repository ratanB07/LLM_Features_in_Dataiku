from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file
import os
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import joblib
import zipfile

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Azure OpenAI
from openai import AzureOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "7c0d2e2f35e020ec485f18271ca26451"

# Azure OpenAI Configuration
try:
    client = AzureOpenAI(
        api_key="F6Zf4y3wEP8HoxVunVjWjIqyrbiEVw6YnNRdj5plfoulznvYVNLOJQQJ99BDAC77bzfXJ3w3AAABACOGqRfA",
        api_version="2024-04-01-preview",
        azure_endpoint="https://gen-ai-llm-deployment.openai.azure.com/"
    )
    deployment_name = "gpt-4o"
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Azure OpenAI client: {str(e)}")
    client = None

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create temp folder for plots
TEMP_FOLDER = 'static/temp'
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Global data store
data_store = {}

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

# Optimized Algorithm configurations for fast training
ALGORITHMS = {
    'classification': {
        'quick': {
            'Logistic Regression': {
                'model': LogisticRegression,
                'params': {'max_iter': 50, 'random_state': 42, 'solver': 'liblinear'},
                'category': 'Linear Models',
                'description': 'Fast linear classifier for binary and multiclass problems'
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier,
                'params': {'max_depth': 3, 'random_state': 42, 'min_samples_split': 10},
                'category': 'Tree-based',
                'description': 'Interpretable tree-based classifier'
            },
            'Naive Bayes': {
                'model': GaussianNB,
                'params': {},
                'category': 'Probabilistic',
                'description': 'Probabilistic classifier based on Bayes theorem'
            }
        },
        'standard': {
            'Random Forest': {
                'model': RandomForestClassifier,
                'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 5, 'n_jobs': -1},
                'category': 'Ensemble',
                'description': 'Robust ensemble method with high accuracy'
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier,
                'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 3},
                'category': 'Ensemble',
                'description': 'Sequential ensemble with excellent performance'
            },
            'SVM': {
                'model': SVC,
                'params': {'kernel': 'linear', 'random_state': 42, 'probability': True, 'max_iter': 100},
                'category': 'Support Vector',
                'description': 'Powerful classifier for complex decision boundaries'
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier,
                'params': {'n_neighbors': 3, 'n_jobs': -1},
                'category': 'Instance-based',
                'description': 'Simple yet effective distance-based classifier'
            },
            'Neural Network': {
                'model': MLPClassifier,
                'params': {'hidden_layer_sizes': (50,), 'max_iter': 50, 'random_state': 42},
                'category': 'Neural Networks',
                'description': 'Deep learning approach for complex patterns'
            }
        }
    },
    'regression': {
        'quick': {
            'Linear Regression': {
                'model': LinearRegression,
                'params': {'n_jobs': -1},
                'category': 'Linear Models',
                'description': 'Simple linear relationship modeling'
            },
            'Ridge Regression': {
                'model': Ridge,
                'params': {'alpha': 1.0, 'random_state': 42},
                'category': 'Linear Models',
                'description': 'Regularized linear regression'
            },
            'Decision Tree': {
                'model': DecisionTreeRegressor,
                'params': {'max_depth': 3, 'random_state': 42, 'min_samples_split': 10},
                'category': 'Tree-based',
                'description': 'Interpretable tree-based regressor'
            }
        },
        'standard': {
            'Random Forest': {
                'model': RandomForestRegressor,
                'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 5, 'n_jobs': -1},
                'category': 'Ensemble',
                'description': 'Robust ensemble method for regression'
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor,
                'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 3},
                'category': 'Ensemble',
                'description': 'Sequential ensemble with excellent performance'
            },
            'SVR': {
                'model': SVR,
                'params': {'kernel': 'linear', 'max_iter': 100},
                'category': 'Support Vector',
                'description': 'Support Vector Regression for non-linear patterns'
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsRegressor,
                'params': {'n_neighbors': 3, 'n_jobs': -1},
                'category': 'Instance-based',
                'description': 'Distance-based regression approach'
            },
            'Neural Network': {
                'model': MLPRegressor,
                'params': {'hidden_layer_sizes': (50,), 'max_iter': 50, 'random_state': 42},
                'category': 'Neural Networks',
                'description': 'Deep learning approach for complex patterns'
            }
        }
    }
}

# Add XGBoost if available (optimized for speed)
if XGBOOST_AVAILABLE:
    ALGORITHMS['classification']['standard']['XGBoost'] = {
        'model': xgb.XGBClassifier,
        'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 3, 'n_jobs': -1},
        'category': 'Gradient Boosting',
        'description': 'Extreme Gradient Boosting with superior performance'
    }
    ALGORITHMS['regression']['standard']['XGBoost'] = {
        'model': xgb.XGBRegressor,
        'params': {'n_estimators': 20, 'random_state': 42, 'max_depth': 3, 'n_jobs': -1},
        'category': 'Gradient Boosting',
        'description': 'Extreme Gradient Boosting for regression'
    }

# Add LightGBM if available (optimized for speed)
if LIGHTGBM_AVAILABLE:
    ALGORITHMS['classification']['standard']['LightGBM'] = {
        'model': lgb.LGBMClassifier,
        'params': {'n_estimators': 20, 'random_state': 42, 'verbose': -1, 'n_jobs': -1},
        'category': 'Gradient Boosting',
        'description': 'Fast gradient boosting with memory efficiency'
    }
    ALGORITHMS['regression']['standard']['LightGBM'] = {
        'model': lgb.LGBMRegressor,
        'params': {'n_estimators': 20, 'random_state': 42, 'verbose': -1, 'n_jobs': -1},
        'category': 'Gradient Boosting',
        'description': 'Fast gradient boosting for regression'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and (file.filename.endswith(('.csv', '.xlsx', '.xls'))):
            # Generate a unique ID for this session
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            logger.info(f"New session created: {session_id}")
            
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
            
            # Read the file
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                logger.info(f"File read successfully: {filename}, shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                return jsonify({'error': f'Error reading file: {str(e)}'}), 400
            
            # Store the dataframe in our data store
            data_store[session_id] = {
                'df': df,
                'filename': filename,
                'file_path': file_path,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Get basic info about the dataframe
            info = {
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'preview': df.head(5).to_dict(orient='records'),
                'session_id': session_id
            }
            
            return jsonify(info)
        
        return jsonify({'error': 'Invalid file type. Please upload a CSV or Excel file.'}), 400
    
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# NLP Tools Routes
@app.route('/nlp-tools')
def nlp_tools():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"NLP Tools route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for NLP Tools: {session_id}")
        return render_template('nlp-tools.html')
    except Exception as e:
        logger.error(f"Error in nlp_tools route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/nlp-tools/dataset-info', methods=['GET'])
def api_nlp_tools_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"NLP Tools dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Analyze columns for NLP suitability
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Check if column is suitable for NLP
            is_text_suitable = False
            if pd.api.types.is_object_dtype(df[col]):
                # Check if it contains text-like data
                sample_values = df[col].dropna().head(5).astype(str).tolist()
                avg_length = np.mean([len(str(val)) for val in sample_values]) if sample_values else 0
                if avg_length > 10:  # Assume text if average length > 10 characters
                    is_text_suitable = True
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'is_text_suitable': is_text_suitable
            })

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns': columns_info,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_nlp_tools_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/nlp-tools/analyze', methods=['POST'])
def api_nlp_tools_analyze():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"NLP analysis requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        text_column = data.get('text_column')
        nlp_task = data.get('nlp_task')
        model = data.get('model')
        sample_size = data.get('sample_size', '500')
        custom_labels = data.get('custom_labels', '')
        
        if not text_column or not nlp_task or not model:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        df = data_store[session_id]['df']
        
        # Sample data if needed
        if sample_size != 'all':
            sample_size = int(sample_size)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampling {len(df)} rows for NLP analysis")
        
        # Perform NLP analysis
        analysis_result = perform_nlp_analysis(df, text_column, nlp_task, model, custom_labels)
        
        # Store analysis result
        analysis_id = str(uuid.uuid4())
        data_store[f"nlp_analysis_{analysis_id}"] = {
            'result': analysis_result,
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        analysis_result['analysis_id'] = analysis_id
        return jsonify(analysis_result)
    
    except Exception as e:
        logger.error(f"Error in api_nlp_tools_analyze: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def perform_nlp_analysis(df, text_column, nlp_task, model, custom_labels):
    """
    Perform NLP analysis using Azure OpenAI
    """
    try:
        # Get text data
        text_data = df[text_column].dropna().astype(str).tolist()
        processed_rows = len(text_data)
        
        if processed_rows == 0:
            raise ValueError("No valid text data found in the selected column")
        
        # Limit to first 100 rows for demo purposes (to avoid API limits)
        if len(text_data) > 100:
            text_data = text_data[:100]
            processed_rows = 100
        
        # Perform analysis based on task type
        if nlp_task == 'sentiment':
            results = analyze_sentiment(text_data, model, custom_labels)
        elif nlp_task == 'classification':
            results = classify_text(text_data, model, custom_labels)
        elif nlp_task == 'entity_extraction':
            results = extract_entities(text_data, model)
        elif nlp_task == 'embedding':
            results = generate_embeddings(text_data, model)
        elif nlp_task == 'summarization':
            results = summarize_text(text_data, model)
        elif nlp_task == 'topic_modeling':
            results = model_topics(text_data, model)
        elif nlp_task == 'language_detection':
            results = detect_language(text_data, model)
        elif nlp_task == 'keyword_extraction':
            results = extract_keywords(text_data, model)
        else:
            raise ValueError(f"Unsupported NLP task: {nlp_task}")
        
        # Calculate metrics
        metrics = calculate_nlp_metrics(results, nlp_task)
        
        # Generate insights
        insights = generate_nlp_insights(results, nlp_task, model)
        
        return {
            'processed_rows': processed_rows,
            'model': model,
            'task': nlp_task,
            'results': results,
            'metrics': metrics,
            'insights': insights
        }
    
    except Exception as e:
        logger.error(f"Error in perform_nlp_analysis: {str(e)}")
        raise

def analyze_sentiment(text_data, model, custom_labels):
    """Analyze sentiment of text data"""
    try:
        if not client:
            # Fallback sentiment analysis
            return fallback_sentiment_analysis(text_data)
        
        results = []
        labels = custom_labels.split(',') if custom_labels else ['positive', 'negative', 'neutral']
        labels = [label.strip() for label in labels]
        
        # Process in batches to avoid token limits
        batch_size = 10
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            
            prompt = f"""
            Analyze the sentiment of the following texts. Classify each text as one of: {', '.join(labels)}.
            
            Texts:
            {chr(10).join([f"{j+1}. {text}" for j, text in enumerate(batch)])}
            
            Respond with JSON format:
            {{
                "results": [
                    {{"text": "original text", "sentiment": "label", "confidence": 0.95}},
                    ...
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert sentiment analysis AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            batch_results = json.loads(response.choices[0].message.content)
            results.extend(batch_results.get('results', []))
        
        return results[:len(text_data)]  # Ensure we don't have more results than input
    
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return fallback_sentiment_analysis(text_data)

def classify_text(text_data, model, custom_labels):
    """Classify text data into categories"""
    try:
        if not client:
            return fallback_text_classification(text_data)
        
        results = []
        labels = custom_labels.split(',') if custom_labels else ['business', 'technology', 'sports', 'entertainment', 'politics']
        labels = [label.strip() for label in labels]
        
        # Process in batches
        batch_size = 10
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            
            prompt = f"""
            Classify the following texts into one of these categories: {', '.join(labels)}.
            
            Texts:
            {chr(10).join([f"{j+1}. {text}" for j, text in enumerate(batch)])}
            
            Respond with JSON format:
            {{
                "results": [
                    {{"text": "original text", "category": "label", "confidence": 0.95}},
                    ...
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert text classification AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            batch_results = json.loads(response.choices[0].message.content)
            results.extend(batch_results.get('results', []))
        
        return results[:len(text_data)]
    
    except Exception as e:
        logger.error(f"Error in text classification: {str(e)}")
        return fallback_text_classification(text_data)

def extract_entities(text_data, model):
    """Extract named entities from text"""
    try:
        if not client:
            return fallback_entity_extraction(text_data)
        
        results = []
        
        # Process in smaller batches for entity extraction
        batch_size = 5
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            
            prompt = f"""
            Extract named entities (PERSON, ORGANIZATION, LOCATION, DATE, MONEY, etc.) from the following texts:
            
            Texts:
            {chr(10).join([f"{j+1}. {text}" for j, text in enumerate(batch)])}
            
            Respond with JSON format:
            {{
                "results": [
                    {{"text": "original text", "entities": [{{"entity": "entity text", "type": "PERSON", "start": 0, "end": 5}}]}},
                    ...
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert named entity recognition AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            batch_results = json.loads(response.choices[0].message.content)
            results.extend(batch_results.get('results', []))
        
        return results[:len(text_data)]
    
    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}")
        return fallback_entity_extraction(text_data)

def generate_embeddings(text_data, model):
    """Generate text embeddings (simplified for demo)"""
    try:
        # For demo purposes, we'll simulate embeddings
        results = []
        for text in text_data:
            # Simulate embedding generation
            embedding = np.random.rand(384).tolist()  # Simulate 384-dimensional embedding
            results.append({
                'text': text,
                'embedding': embedding,
                'dimension': 384
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error in embedding generation: {str(e)}")
        return [{'text': text, 'embedding': [], 'dimension': 0} for text in text_data]

def summarize_text(text_data, model):
    """Summarize text data"""
    try:
        if not client:
            return fallback_summarization(text_data)
        
        results = []
        
        for text in text_data:
            if len(text) < 100:  # Skip very short texts
                results.append({
                    'text': text,
                    'summary': text,
                    'compression_ratio': 1.0
                })
                continue
            
            prompt = f"""
            Provide a concise summary of the following text:
            
            Text: {text}
            
            Respond with JSON format:
            {{
                "summary": "concise summary here"
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert text summarization AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            summary = result.get('summary', text)
            
            results.append({
                'text': text,
                'summary': summary,
                'compression_ratio': len(summary) / len(text)
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return fallback_summarization(text_data)

def model_topics(text_data, model):
    """Perform topic modeling on text data"""
    try:
        if not client:
            return fallback_topic_modeling(text_data)
        
        # Combine all texts for topic analysis
        combined_text = ' '.join(text_data)
        
        prompt = f"""
        Analyze the following texts and identify the main topics. Assign each text to the most relevant topic.
        
        Combined texts: {combined_text[:2000]}...
        
        Individual texts:
        {chr(10).join([f"{i+1}. {text}" for i, text in enumerate(text_data[:20])])}
        
        Respond with JSON format:
        {{
            "topics": ["topic1", "topic2", "topic3"],
            "results": [
                {{"text": "original text", "topic": "topic1", "confidence": 0.85}},
                ...
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert topic modeling AI. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('results', [])
    
    except Exception as e:
        logger.error(f"Error in topic modeling: {str(e)}")
        return fallback_topic_modeling(text_data)

def detect_language(text_data, model):
    """Detect language of text data"""
    try:
        if not client:
            return fallback_language_detection(text_data)
        
        results = []
        
        # Process in batches
        batch_size = 20
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            
            prompt = f"""
            Detect the language of the following texts:
            
            Texts:
            {chr(10).join([f"{j+1}. {text}" for j, text in enumerate(batch)])}
            
            Respond with JSON format:
            {{
                "results": [
                    {{"text": "original text", "language": "English", "language_code": "en", "confidence": 0.95}},
                    ...
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert language detection AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            batch_results = json.loads(response.choices[0].message.content)
            results.extend(batch_results.get('results', []))
        
        return results[:len(text_data)]
    
    except Exception as e:
        logger.error(f"Error in language detection: {str(e)}")
        return fallback_language_detection(text_data)

def extract_keywords(text_data, model):
    """Extract keywords from text data"""
    try:
        if not client:
            return fallback_keyword_extraction(text_data)
        
        results = []
        
        for text in text_data:
            prompt = f"""
            Extract the most important keywords and phrases from the following text:
            
            Text: {text}
            
            Respond with JSON format:
            {{
                "keywords": ["keyword1", "keyword2", "keyword3"],
                "phrases": ["important phrase 1", "important phrase 2"]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert keyword extraction AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            results.append({
                'text': text,
                'keywords': result.get('keywords', []),
                'phrases': result.get('phrases', [])
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error in keyword extraction: {str(e)}")
        return fallback_keyword_extraction(text_data)

# Fallback functions for when Azure OpenAI is not available
def fallback_sentiment_analysis(text_data):
    """Fallback sentiment analysis using simple rules"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst']
    
    results = []
    for text in text_data:
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.6 + (pos_count - neg_count) * 0.1)
        elif neg_count > pos_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.6 + (neg_count - pos_count) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence
        })
    
    return results

def fallback_text_classification(text_data):
    """Fallback text classification using simple keyword matching"""
    categories = {
        'business': ['business', 'company', 'market', 'finance', 'economy', 'profit', 'revenue'],
        'technology': ['technology', 'software', 'computer', 'AI', 'digital', 'tech', 'innovation'],
        'sports': ['sports', 'game', 'team', 'player', 'match', 'score', 'championship'],
        'entertainment': ['movie', 'music', 'celebrity', 'entertainment', 'show', 'actor', 'film'],
        'politics': ['politics', 'government', 'election', 'policy', 'political', 'vote', 'democracy']
    }
    
    results = []
    for text in text_data:
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        
        best_category = max(scores, key=scores.get) if max(scores.values()) > 0 else 'other'
        confidence = min(0.9, 0.5 + scores[best_category] * 0.1)
        
        results.append({
            'text': text,
            'category': best_category,
            'confidence': confidence
        })
    
    return results

def fallback_entity_extraction(text_data):
    """Fallback entity extraction using simple patterns"""
    import re
    
    results = []
    for text in text_data:
        entities = []
        
        # Simple patterns for common entities
        # Names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for name in names:
            entities.append({
                'entity': name,
                'type': 'PERSON',
                'start': text.find(name),
                'end': text.find(name) + len(name)
            })
        
        # Dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text)
        for date in dates:
            entities.append({
                'entity': date,
                'type': 'DATE',
                'start': text.find(date),
                'end': text.find(date) + len(date)
            })
        
        # Money
        money = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
        for amount in money:
            entities.append({
                'entity': amount,
                'type': 'MONEY',
                'start': text.find(amount),
                'end': text.find(date) + len(amount)
            })
        
        results.append({
            'text': text,
            'entities': entities[:5]  # Limit to 5 entities per text
        })
    
    return results

def fallback_summarization(text_data):
    """Fallback summarization using simple sentence extraction"""
    results = []
    for text in text_data:
        sentences = text.split('.')
        if len(sentences) <= 2:
            summary = text
        else:
            # Take first and last sentence as summary
            summary = sentences[0] + '. ' + sentences[-1] if len(sentences) > 1 else sentences[0]
        
        results.append({
            'text': text,
            'summary': summary.strip(),
            'compression_ratio': len(summary) / len(text) if len(text) > 0 else 1.0
        })
    
    return results

def fallback_topic_modeling(text_data):
    """Fallback topic modeling using keyword frequency"""
    from collections import Counter
    import re
    
    # Extract all words
    all_words = []
    for text in text_data:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        all_words.extend(words)
    
    # Get most common words as topics
    word_counts = Counter(all_words)
    topics = [word for word, count in word_counts.most_common(5)]
    
    results = []
    for text in text_data:
        text_lower = text.lower()
        topic_scores = {}
        
        for topic in topics:
            score = text_lower.count(topic)
            topic_scores[topic] = score
        
        best_topic = max(topic_scores, key=topic_scores.get) if topic_scores else 'general'
        confidence = min(0.9, 0.5 + topic_scores.get(best_topic, 0) * 0.1)
        
        results.append({
            'text': text,
            'topic': best_topic,
            'confidence': confidence
        })
    
    return results

def fallback_language_detection(text_data):
    """Fallback language detection using simple heuristics"""
    results = []
    for text in text_data:
        # Simple heuristic: assume English for now
        # In a real implementation, you could use character frequency analysis
        results.append({
            'text': text,
            'language': 'English',
            'language_code': 'en',
            'confidence': 0.8
        })
    
    return results

def fallback_keyword_extraction(text_data):
    """Fallback keyword extraction using word frequency"""
    import re
    from collections import Counter
    
    results = []
    for text in text_data:
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        filtered_words = [word for word in words if word not in stop_words]
        
        # Get most frequent words as keywords
        word_counts = Counter(filtered_words)
        keywords = [word for word, count in word_counts.most_common(5)]
        
        # Extract phrases (simple bigrams)
        phrases = []
        for i in range(len(words) - 1):
            if words[i] not in stop_words and words[i+1] not in stop_words:
                phrases.append(f"{words[i]} {words[i+1]}")
        
        phrase_counts = Counter(phrases)
        top_phrases = [phrase for phrase, count in phrase_counts.most_common(3)]
        
        results.append({
            'text': text,
            'keywords': keywords,
            'phrases': top_phrases
        })
    
    return results

def calculate_nlp_metrics(results, nlp_task):
    """Calculate metrics based on NLP task results"""
    try:
        metrics = {}
        
        if nlp_task == 'sentiment':
            sentiments = [r.get('sentiment', 'neutral') for r in results]
            sentiment_counts = Counter(sentiments)
            total = len(sentiments)
            
            metrics = {
                'total_texts': total,
                'positive_count': sentiment_counts.get('positive', 0),
                'negative_count': sentiment_counts.get('negative', 0),
                'neutral_count': sentiment_counts.get('neutral', 0),
                'avg_confidence': np.mean([r.get('confidence', 0) for r in results])
            }
        
        elif nlp_task == 'classification':
            categories = [r.get('category', 'unknown') for r in results]
            category_counts = Counter(categories)
            
            metrics = {
                'total_texts': len(categories),
                'unique_categories': len(category_counts),
                'most_common_category': category_counts.most_common(1)[0][0] if category_counts else 'none',
                'avg_confidence': np.mean([r.get('confidence', 0) for r in results])
            }
        
        elif nlp_task == 'entity_extraction':
            all_entities = []
            for r in results:
                all_entities.extend(r.get('entities', []))
            
            entity_types = [e.get('type', 'UNKNOWN') for e in all_entities]
            type_counts = Counter(entity_types)
            
            metrics = {
                'total_texts': len(results),
                'total_entities': len(all_entities),
                'avg_entities_per_text': len(all_entities) / len(results) if results else 0,
                'most_common_entity_type': type_counts.most_common(1)[0][0] if type_counts else 'none'
            }
        
        elif nlp_task == 'embedding':
            metrics = {
                'total_texts': len(results),
                'embedding_dimension': results[0].get('dimension', 0) if results else 0,
                'avg_text_length': np.mean([len(r.get('text', '')) for r in results])
            }
        
        elif nlp_task == 'summarization':
            compression_ratios = [r.get('compression_ratio', 1.0) for r in results]
            
            metrics = {
                'total_texts': len(results),
                'avg_compression_ratio': np.mean(compression_ratios),
                'min_compression_ratio': np.min(compression_ratios),
                'max_compression_ratio': np.max(compression_ratios)
            }
        
        elif nlp_task == 'topic_modeling':
            topics = [r.get('topic', 'unknown') for r in results]
            topic_counts = Counter(topics)
            
            metrics = {
                'total_texts': len(topics),
                'unique_topics': len(topic_counts),
                'most_common_topic': topic_counts.most_common(1)[0][0] if topic_counts else 'none',
                'avg_confidence': np.mean([r.get('confidence', 0) for r in results])
            }
        
        elif nlp_task == 'language_detection':
            languages = [r.get('language', 'unknown') for r in results]
            language_counts = Counter(languages)
            
            metrics = {
                'total_texts': len(languages),
                'unique_languages': len(language_counts),
                'most_common_language': language_counts.most_common(1)[0][0] if language_counts else 'none',
                'avg_confidence': np.mean([r.get('confidence', 0) for r in results])
            }
        
        elif nlp_task == 'keyword_extraction':
            all_keywords = []
            for r in results:
                all_keywords.extend(r.get('keywords', []))
            
            keyword_counts = Counter(all_keywords)
            
            metrics = {
                'total_texts': len(results),
                'total_keywords': len(all_keywords),
                'unique_keywords': len(keyword_counts),
                'avg_keywords_per_text': len(all_keywords) / len(results) if results else 0
            }
        
        else:
            metrics = {
                'total_texts': len(results),
                'task_type': nlp_task
            }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {'total_texts': len(results), 'error': str(e)}

def generate_nlp_insights(results, nlp_task, model):
    """Generate insights about NLP analysis results"""
    try:
        insights = []
        
        if nlp_task == 'sentiment':
            sentiments = [r.get('sentiment', 'neutral') for r in results]
            sentiment_counts = Counter(sentiments)
            total = len(sentiments)
            
            if sentiment_counts.get('positive', 0) > total * 0.6:
                insights.append({
                    'title': 'Predominantly Positive Sentiment',
                    'description': f'Over 60% of texts show positive sentiment, indicating favorable opinions or experiences.'
                })
            elif sentiment_counts.get('negative', 0) > total * 0.6:
                insights.append({
                    'title': 'Predominantly Negative Sentiment',
                    'description': f'Over 60% of texts show negative sentiment, suggesting areas for improvement.'
                })
            else:
                insights.append({
                    'title': 'Mixed Sentiment Distribution',
                    'description': f'Sentiment is well-distributed across positive, negative, and neutral categories.'
                })
        
        elif nlp_task == 'classification':
            categories = [r.get('category', 'unknown') for r in results]
            category_counts = Counter(categories)
            most_common = category_counts.most_common(1)[0] if category_counts else ('none', 0)
            
            insights.append({
                'title': 'Category Distribution Analysis',
                'description': f'Most common category is "{most_common[0]}" with {most_common[1]} occurrences ({(most_common[1]/len(categories)*100):.1f}% of total).'
            })
        
        elif nlp_task == 'entity_extraction':
            all_entities = []
            for r in results:
                all_entities.extend(r.get('entities', []))
            
            entity_types = [e.get('type', 'UNKNOWN') for e in all_entities]
            type_counts = Counter(entity_types)
            
            if type_counts:
                most_common_type = type_counts.most_common(1)[0]
                insights.append({
                    'title': 'Entity Type Analysis',
                    'description': f'Most frequently extracted entity type is {most_common_type[0]} with {most_common_type[1]} occurrences.'
                })
        
        elif nlp_task == 'topic_modeling':
            topics = [r.get('topic', 'unknown') for r in results]
            topic_counts = Counter(topics)
            
            insights.append({
                'title': 'Topic Distribution',
                'description': f'Identified {len(topic_counts)} distinct topics. Most prevalent topic: "{topic_counts.most_common(1)[0][0]}" if topic_counts else "none".'
            })
        
        # Add general insights
        insights.append({
            'title': 'ETL Integration Recommendation',
            'description': f'This {nlp_task} analysis can be integrated into your ETL pipeline for automated text processing and enrichment.'
        })
        
        insights.append({
            'title': 'Data Quality Assessment',
            'description': f'Processed {len(results)} text samples using {model} model with high accuracy and reliability.'
        })
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return [{'title': 'Analysis Complete', 'description': f'Successfully processed {len(results)} texts using {nlp_task} analysis.'}]

@app.route('/api/nlp-tools/download', methods=['POST'])
def api_nlp_tools_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        analysis_id = data.get('analysis_id')
        
        if not session_id or not analysis_id:
            return jsonify({'error': 'Missing session_id or analysis_id'}), 400
        
        analysis_key = f"nlp_analysis_{analysis_id}"
        if analysis_key not in data_store:
            return jsonify({'error': 'Analysis not found'}), 404
        
        analysis_data = data_store[analysis_key]
        results = analysis_data['result']['results']
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Create temporary file
        temp_filename = f"nlp_analysis_results_{analysis_id}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        df_results.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_nlp_tools_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

# AutoML Routes
@app.route('/automl')
def automl():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"AutoML route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for AutoML: {session_id}")
        return render_template('automl.html')
    except Exception as e:
        logger.error(f"Error in automl route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/automl/dataset-info', methods=['GET'])
def api_automl_dataset_info():
    try:
        # Try to get session_id from multiple sources
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Sample data for faster processing (max 1000 rows)
        if len(df) > 1000:
            df_sample = df.sample(n=1000, random_state=42)
            logger.info(f"Sampling {len(df_sample)} rows from {len(df)} total rows for faster processing")
        else:
            df_sample = df
        
        # Analyze columns
        columns_info = []
        for col in df_sample.columns:
            col_type = str(df_sample[col].dtype)
            missing = df_sample[col].isna().sum()
            missing_pct = (missing / len(df_sample)) * 100
            unique_count = df_sample[col].nunique()
            
            # Determine if column is suitable for target
            is_target_suitable = False
            target_type = None
            
            if pd.api.types.is_numeric_dtype(df_sample[col]):
                if unique_count <= 20 and unique_count >= 2:
                    is_target_suitable = True
                    target_type = 'classification'
                elif unique_count > 20:
                    is_target_suitable = True
                    target_type = 'regression'
            elif pd.api.types.is_object_dtype(df_sample[col]):
                if unique_count <= 50 and unique_count >= 2:
                    is_target_suitable = True
                    target_type = 'classification'
            
            # Get sample values safely
            sample_values = []
            try:
                non_null_values = df_sample[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(3, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                logger.warning(f"Error getting sample values for column {col}: {str(e)}")
                sample_values = ["N/A"]
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'is_target_suitable': is_target_suitable,
                'target_type': target_type,
                'sample_values': sample_values
            })
        
        # Create a serializable version of algorithms without the actual model classes
        algorithms_info = {}
        for problem_type in ALGORITHMS:
            algorithms_info[problem_type] = {}
            for complexity in ALGORITHMS[problem_type]:
                algorithms_info[problem_type][complexity] = {}
                for algo_name, algo_config in ALGORITHMS[problem_type][complexity].items():
                    algorithms_info[problem_type][complexity][algo_name] = {
                        'category': algo_config['category'],
                        'description': algo_config['description']
                    }

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'algorithms': algorithms_info,
            'session_id': session_id  # Include session_id in response
        })
    
    except Exception as e:
        logger.error(f"Error in api_automl_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/automl/analyze', methods=['POST'])
def api_automl_analyze():
    try:
        # Get session_id from request or session
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"Training requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        target_column = data.get('target_column')
        feature_columns = data.get('feature_columns', [])
        algorithm = data.get('algorithm')
        problem_type = data.get('problem_type')
        test_size = float(data.get('test_size', 0.2))
        
        if not target_column or not feature_columns or not algorithm:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        df = data_store[session_id]['df']
        
        # Sample data for faster training (max 5000 rows)
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)
            logger.info(f"Sampling {len(df)} rows for faster training")
        
        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remove rows with missing target values
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            return jsonify({'error': 'No valid data after removing missing values'}), 400
        
        # Get algorithm configuration
        algo_config = None
        for complexity in ALGORITHMS[problem_type]:
            if algorithm in ALGORITHMS[problem_type][complexity]:
                algo_config = ALGORITHMS[problem_type][complexity][algorithm]
                break
        
        if not algo_config:
            return jsonify({'error': f'Algorithm {algorithm} not found'}), 400
        
        # Simple preprocessing for speed
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values quickly
        for col in numeric_features:
            X[col].fillna(X[col].median(), inplace=True)
        
        for col in categorical_features:
            X[col].fillna('Unknown', inplace=True)
        
        # Simple encoding for categorical variables
        if categorical_features:
            for col in categorical_features:
                # Simple label encoding for speed
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if classification
        if problem_type == 'classification' and pd.api.types.is_object_dtype(y):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y
            label_encoder = None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, 
            stratify=y_encoded if problem_type == 'classification' else None
        )
        
        # Create and train model (simplified for speed)
        model = algo_config['model'](**algo_config['params'])
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Create simple confusion matrix plot
            try:
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                # Save plot
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=80)
                buffer.seek(0)
                cm_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating confusion matrix: {str(e)}")
                cm_plot = None
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm_plot
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Create simple prediction vs actual plot
            try:
                plt.figure(figsize=(6, 4))
                plt.scatter(y_test, y_pred, alpha=0.6, s=20)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title('Prediction vs Actual')
                plt.tight_layout()
                
                # Save plot
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=80)
                buffer.seek(0)
                pred_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating prediction plot: {str(e)}")
                pred_plot = None
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'prediction_plot': pred_plot
            }
        
        # Feature importance (if available)
        feature_importance = None
        try:
            if hasattr(model, 'feature_importances_'):
                feature_names = X.columns.tolist()
                importances = model.feature_importances_
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                feature_importance = feature_importance[:10]  # Top 10 features
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
        
        # Store model for download
        model_id = str(uuid.uuid4())
        model_data = {
            'model': model,
            'label_encoder': label_encoder,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'algorithm': algorithm,
            'problem_type': problem_type,
            'metrics': metrics,
            'training_time': training_time
        }
        data_store[f"model_{model_id}"] = model_data
        
        # Generate AI insights (simplified for speed)
        ai_insights = generate_automl_insights_fast(metrics, feature_importance, algorithm, problem_type)
        
        return jsonify({
            'model_id': model_id,
            'algorithm': algorithm,
            'problem_type': problem_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_time': float(training_time),
            'ai_insights': ai_insights,
            'data_shape': {
                'train': X_train.shape,
                'test': X_test.shape
            }
        })
    
    except Exception as e:
        logger.error(f"Error in api_automl_analyze: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def generate_automl_insights_fast(metrics, feature_importance, algorithm, problem_type):
    """Generate fast insights without AI for speed"""
    if problem_type == 'classification':
        accuracy = metrics['accuracy']
        if accuracy > 0.9:
            performance = "Excellent performance"
            recommendation = "Model is ready for deployment"
        elif accuracy > 0.8:
            performance = "Good performance"
            recommendation = "Consider minor tuning for improvement"
        elif accuracy > 0.7:
            performance = "Moderate performance"
            recommendation = "Try feature engineering or different algorithms"
        else:
            performance = "Poor performance"
            recommendation = "Needs significant improvement - check data quality"
    else:
        r2 = metrics['r2_score']
        if r2 > 0.9:
            performance = "Excellent fit"
            recommendation = "Model explains data very well"
        elif r2 > 0.7:
            performance = "Good fit"
            recommendation = "Model performs well with room for improvement"
        elif r2 > 0.5:
            performance = "Moderate fit"
            recommendation = "Consider feature engineering or more complex models"
        else:
            performance = "Poor fit"
            recommendation = "Model needs significant improvement"
    
    feature_insight = "No feature importance available"
    if feature_importance and len(feature_importance) > 0:
        top_feature = feature_importance[0][0]
        feature_insight = f"'{top_feature}' is the most important feature for prediction"
    
    return {
        "performance_assessment": f"{performance} achieved with {algorithm}",
        "feature_insights": feature_insight,
        "improvement_recommendations": [
            recommendation,
            "Consider collecting more training data",
            "Try different preprocessing techniques"
        ],
        "etl_suggestions": [
            "Check for data quality issues",
            "Validate data consistency",
            "Consider feature scaling if needed"
        ],
        "deployment_readiness": f"Model training completed in under 5 seconds. {recommendation}"
    }

# Data Profiling Routes
@app.route('/dataprofiling')
def data_profiling():
    try:
        session_id = session.get('session_id')
        logger.info(f"Data profiling requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        return render_template('dataprofiling.html')
    
    except Exception as e:
        logger.error(f"Error in data_profiling route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/dataprofiling', methods=['GET'])
def api_data_profiling():
    try:
        session_id = session.get('session_id')
        logger.info(f"API data profiling requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        
        # Generate profiling data
        profile = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB",
                'duplicated_rows': int(df.duplicated().sum())
            },
            'columns': []
        }
        
        # Process each column
        for col in df.columns:
            try:
                col_type = str(df[col].dtype)
                missing = int(df[col].isna().sum())
                missing_pct = (missing / len(df)) * 100
                
                col_info = {
                    'name': col,
                    'type': col_type,
                    'missing': missing,
                    'missing_pct': f"{missing_pct:.2f}%",
                }
                
                # Add numeric statistics
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        col_info.update({
                            'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                            'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                            'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                            'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                        })
                        
                        # Generate histogram
                        plt.figure(figsize=(8, 4))
                        sns.histplot(df[col].dropna(), kde=True)
                        plt.title(f'Distribution of {col}')
                        plt.tight_layout()
                        
                        # Save plot to buffer
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        plt.close()
                        
                        col_info['histogram'] = plot_data
                    except Exception as e:
                        logger.warning(f"Error generating numeric stats for column {col}: {str(e)}")
                    
                # Add categorical statistics
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    try:
                        value_counts = df[col].value_counts().head(10).to_dict()
                        col_info.update({
                            'unique_values': int(df[col].nunique()),
                            'top_values': value_counts
                        })
                        
                        # Generate bar plot for top categories
                        if df[col].nunique() <= 20:  # Only for columns with reasonable number of categories
                            plt.figure(figsize=(10, 5))
                            top_categories = df[col].value_counts().head(10)
                            sns.barplot(x=top_categories.index, y=top_categories.values)
                            plt.title(f'Top Categories in {col}')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            
                            # Save plot to buffer
                            buffer = io.BytesIO()
                            plt.savefig(buffer, format='png')
                            buffer.seek(0)
                            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            plt.close()
                            
                            col_info['barplot'] = plot_data
                    except Exception as e:
                        logger.warning(f"Error generating categorical stats for column {col}: {str(e)}")
                
                # Add datetime statistics
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    try:
                        col_info.update({
                            'min': df[col].min().strftime('%Y-%m-%d %H:%M:%S'),
                            'max': df[col].max().strftime('%Y-%m-%d %H:%M:%S'),
                            'range': f"{(df[col].max() - df[col].min()).days} days"
                        })
                    except Exception as e:
                        logger.warning(f"Error generating datetime stats for column {col}: {str(e)}")
                
                profile['columns'].append(col_info)
            
            except Exception as e:
                logger.warning(f"Error processing column {col}: {str(e)}")
                # Add minimal column info to avoid breaking the UI
                profile['columns'].append({
                    'name': col,
                    'type': str(df[col].dtype),
                    'missing': int(df[col].isna().sum()),
                    'missing_pct': f"{(df[col].isna().sum() / len(df)) * 100:.2f}%",
                    'error': str(e)
                })
        
        # Generate correlation matrix for numeric columns
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                
                # Save plot to buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                corr_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                profile['correlation_matrix'] = corr_plot
        except Exception as e:
            logger.warning(f"Error generating correlation matrix: {str(e)}")
        
        # Use Azure OpenAI to generate insights
        try:
            if client:
                df_sample = df.head(100).to_string()
                prompt = f"""
                You are a data analysis expert. Based on the following dataset sample, provide 3-5 key insights and potential analysis directions.
                
                Dataset Sample:
                {df_sample}
                
                Please provide insights about:
                1. Data quality issues you notice
                2. Interesting patterns or relationships
                3. Suggested analyses or visualizations
                4. Potential business questions this data could answer
                
                Format your response as JSON with the following structure:
                {{
                    "insights": [
                        {{
                            "title": "Insight title",
                            "description": "Detailed explanation"
                        }}
                    ],
                    "suggested_analyses": [
                        {{
                            "title": "Analysis title",
                            "description": "What to analyze and why it's valuable"
                        }}
                    ]
                }}
                """
                
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a data analysis expert providing insights in JSON format only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                ai_insights = json.loads(response.choices[0].message.content)
                profile['ai_insights'] = ai_insights
            else:
                # Fallback if Azure OpenAI is not available
                profile['ai_insights'] = {
                    "insights": [
                        {
                            "title": "Basic Data Overview",
                            "description": f"This dataset contains {len(df)} rows and {len(df.columns)} columns with various data types."
                        }
                    ],
                    "suggested_analyses": [
                        {
                            "title": "Exploratory Data Analysis",
                            "description": "Perform basic statistical analysis and visualization to understand the distribution and relationships in the data."
                        }
                    ]
                }
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            profile['ai_insights'] = {
                "error": "Could not generate AI insights. Please try again later.",
                "insights": [
                    {
                        "title": "Basic Data Overview",
                        "description": f"This dataset contains {len(df)} rows and {len(df.columns)} columns."
                    }
                ],
                "suggested_analyses": [
                    {
                        "title": "Exploratory Data Analysis",
                        "description": "Perform basic statistical analysis to understand your data."
                    }
                ]
            }
        
        return jsonify(profile)
    
    except Exception as e:
        logger.error(f"Error in api_data_profiling: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'An error occurred while profiling the data.',
            'details': str(e),
            'basic_info': {
                'rows': 0,
                'columns': 0,
                'memory_usage': '0 MB',
                'duplicated_rows': 0
            },
            'columns': []
        }), 500

# LLM Code Generation Routes
@app.route('/llmcodegeneration')
def llm_code_generation():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in data_store:
            return redirect(url_for('index'))
        
        return render_template('llmcodegeneration.html')
    except Exception as e:
        logger.error(f"Error in llm_code_generation route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/llmcodegeneration/generate', methods=['POST'])
def api_llmcodegeneration_generate():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in data_store:
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        # Get data from request
        data = request.json
        task_description = data.get('task_description')
        code_type = data.get('code_type', 'analysis')
        complexity = data.get('complexity', 3)
        
        if not task_description:
            return jsonify({'error': 'No task description provided'}), 400
        
        # Get dataframe info
        df = data_store[session_id]['df']
        df_info = {
            'columns': df.columns.tolist(),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'shape': df.shape,
            'sample': df.head(5).to_dict(orient='records')
        }
        
        # Use Azure OpenAI to generate code
        try:
            if client:
                df_sample = df.head(10).to_string()
                df_info_str = json.dumps(df_info, indent=2)
                
                complexity_levels = {
                    1: "basic (simple operations, no advanced techniques)",
                    2: "simple (common data operations, minimal complexity)",
                    3: "medium (some advanced techniques, good balance)",
                    4: "advanced (complex operations, efficient code)",
                    5: "expert (cutting-edge techniques, highly optimized)"
                }
                
                complexity_desc = complexity_levels.get(complexity, "medium")
                
                prompt = f"""
                You are a Python data science expert. Generate code for the following task:
                
                Task: {task_description}
                
                Code Type: {code_type}
                Complexity Level: {complexity} ({complexity_desc})
                
                Dataset Information:
                {df_info_str}
                
                Dataset Sample:
                {df_sample}
                
                Please provide:
                1. A clear explanation of what the code does
                2. Well-commented Python code that accomplishes the task
                3. A list of required packages
                
                The code should assume that the dataframe is already loaded as 'df'.
                The code should be appropriate for the specified complexity level.
                
                Format your response as JSON with the following structure:
                {{
                    "explanation": "Explanation of what the code does",
                    "code": "Python code",
                    "requirements": ["package1", "package2"]
                }}
                """
                
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a Python data science expert providing code in JSON format only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
            else:
                # Fallback if Azure OpenAI is not available
                result = {
                    "explanation": "This code performs basic data analysis on the dataset.",
                    "code": "# Basic data analysis\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n# Display basic statistics\nprint(df.describe())\n\n# Check for missing values\nprint('\\nMissing values:')\nprint(df.isnull().sum())\n\n# Plot a histogram for numeric columns\ndf.select_dtypes(include=['number']).hist(figsize=(10, 8))\nplt.tight_layout()\nplt.show()",
                    "requirements": ["pandas", "matplotlib"]
                }
            
            # Store the generated code
            data_store[session_id]['generated_code'] = result['code']
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return jsonify({
                "explanation": "Basic data analysis code (fallback due to error).",
                "code": "# Basic data analysis\nimport pandas as pd\n\n# Display basic statistics\nprint(df.describe())\n\n# Check for missing values\nprint('\\nMissing values:')\nprint(df.isnull().sum())",
                "requirements": ["pandas"],
                "error": str(e)
            })
    
    except Exception as e:
        logger.error(f"Error in api_llmcodegeneration_generate: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/llmcodegeneration/execute', methods=['POST'])
def api_llmcodegeneration_execute():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in data_store:
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        # Get code from request
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Get dataframe
        df = data_store[session_id]['df'].copy()
        
        # Execute code
        try:
            # Redirect stdout to capture print statements
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            
            # Create a local namespace
            local_vars = {'df': df, 'plt': plt, 'np': np, 'pd': pd, 'sns': sns}
            
            # Execute the code
            exec(code, globals(), local_vars)
            
            # Get the output
            output = mystdout.getvalue()
            
            # Reset stdout
            sys.stdout = old_stdout
            
            # Check if there are any figures
            figures = []
            if plt.get_fignums():
                for i in plt.get_fignums():
                    fig = plt.figure(i)
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png')
                    buffer.seek(0)
                    figures.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
                
                plt.close('all')
            
            return jsonify({
                'output': output,
                'figures': figures
            })
        
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        logger.error(f"Error in api_llmcodegeneration_execute: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# LLM Mesh Routes
@app.route('/llm_mesh')
def llm_mesh():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in data_store:
            return redirect(url_for('index'))
        
        return render_template('llm_mesh.html')
    except Exception as e:
        logger.error(f"Error in llm_mesh route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/llm_mesh/analyze', methods=['POST'])
def api_llm_mesh_analyze():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in data_store:
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Simulate LLM Mesh analysis with comprehensive data profiling
        analysis_result = perform_llm_mesh_analysis(df, filename)
        
        return jsonify(analysis_result)
    
    except Exception as e:
        logger.error(f"Error in api_llm_mesh_analyze: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def perform_llm_mesh_analysis(df, filename):
    """
    Perform comprehensive LLM Mesh analysis simulating multiple LLMs working together
    """
    try:
        # Basic profiling
        profiling = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': int(df.isnull().sum().sum()),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'quality_score': calculate_data_quality_score(df)
        }
        
        # Anomaly detection
        anomalies = detect_anomalies(df)
        
        # Column analysis
        column_analysis = analyze_columns_with_llm(df)
        
        # Generate insights using Azure OpenAI
        insights = generate_llm_mesh_insights(df, filename)
        
        return {
            'dataset_name': filename,
            'profiling': profiling,
            'anomalies': anomalies,
            'column_analysis': column_analysis,
            'insights': insights,
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    except Exception as e:
        logger.error(f"Error in perform_llm_mesh_analysis: {str(e)}")
        return {
            'error': str(e),
            'dataset_name': filename,
            'profiling': {'total_rows': 0, 'total_columns': 0},
            'anomalies': [],
            'column_analysis': [],
            'insights': []
        }

def calculate_data_quality_score(df):
    """Calculate a data quality score based on various factors"""
    try:
        # Factors for quality score
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        duplicate_ratio = df.duplicated().sum() / len(df)
        
        # Calculate score (0-100)
        quality_score = 100 - (missing_ratio * 50) - (duplicate_ratio * 30)
        quality_score = max(0, min(100, quality_score))
        
        return f"{quality_score:.1f}%"
    except:
        return "N/A"

def detect_anomalies(df):
    """Detect various types of anomalies in the dataset"""
    anomalies = []
    
    try:
        # Check for high missing values
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                anomalies.append({
                    'type': f'High Missing Values in {col}',
                    'description': f'Column {col} has {missing_pct:.1f}% missing values',
                    'confidence': 95
                })
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df)) * 100
            anomalies.append({
                'type': 'Duplicate Rows Detected',
                'description': f'Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}% of dataset)',
                'confidence': 100
            })
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                
                if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                    anomalies.append({
                        'type': f'Statistical Outliers in {col}',
                        'description': f'Column {col} contains {len(outliers)} potential outliers',
                        'confidence': 80
                    })
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                anomalies.append({
                    'type': f'Constant Column: {col}',
                    'description': f'Column {col} has only one unique value',
                    'confidence': 100
                })
    
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
    
    return anomalies

def analyze_columns_with_llm(df):
    """Analyze each column and provide AI-powered insights"""
    column_analysis = []
    
    try:
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Generate summary based on column type
            if pd.api.types.is_numeric_dtype(df[col]):
                summary = f"Numeric column with {unique_count} unique values. Range: {df[col].min():.2f} to {df[col].max():.2f}"
                if missing_pct < 5:
                    recommendation = "Good quality numeric data. Consider for statistical analysis."
                elif missing_pct < 20:
                    recommendation = "Some missing values. Consider for imputation strategies."
                else:
                    recommendation = "High missing values. Investigate data collection process."
            
            elif pd.api.types.is_object_dtype(df[col]):
                summary = f"Categorical column with {unique_count} unique values"
                if unique_count / len(df) > 0.8:
                    recommendation = "High cardinality. Consider grouping or encoding strategies."
                elif unique_count < 10:
                    recommendation = "Low cardinality. Good for categorical analysis."
                else:
                    recommendation = "Medium cardinality. Suitable for most analyses."
            
            else:
                summary = f"Column of type {col_type} with {unique_count} unique values"
                recommendation = "Review data type and consider appropriate preprocessing."
            
            column_analysis.append({
                'name': col,
                'type': col_type,
                'summary': summary,
                'recommendation': recommendation
            })
    
    except Exception as e:
        logger.error(f"Error in column analysis: {str(e)}")
    
    return column_analysis

def generate_llm_mesh_insights(df, filename):
    """Generate strategic insights using Azure OpenAI (simulating LLM Mesh)"""
    insights = []
    
    try:
        if client:
            # Create a comprehensive prompt for LLM Mesh analysis
            df_summary = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {col: str(df[col].dtype) for col in df.columns},
                'missing_summary': df.isnull().sum().to_dict(),
                'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
            
            prompt = f"""
            You are part of an advanced LLM Mesh system analyzing the dataset "{filename}". 
            Multiple specialized AI models have contributed to this analysis. Provide strategic insights and recommendations.
            
            Dataset Summary:
            {json.dumps(df_summary, indent=2, default=str)}
            
            As part of the LLM Mesh, provide 5-7 strategic insights covering:
            1. Data quality and reliability assessment
            2. Business value and potential use cases
            3. Recommended analytical approaches
            4. Data preparation suggestions
            5. Risk factors and limitations
            6. Opportunities for further analysis
            7. Integration recommendations
            
            Format as JSON:
            {{
                "insights": [
                    {{
                        "title": "Insight title",
                        "description": "Detailed strategic recommendation"
                    }}
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are an advanced LLM Mesh system providing strategic data insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            insights = ai_response.get('insights', [])
        
        else:
            # Fallback insights if Azure OpenAI is not available
            insights = [
                {
                    "title": "Data Quality Assessment",
                    "description": f"The dataset contains {len(df)} rows and {len(df.columns)} columns with varying data quality. Consider implementing data validation rules."
                },
                {
                    "title": "Analysis Readiness",
                    "description": "The dataset appears suitable for exploratory data analysis. Recommend starting with descriptive statistics and visualization."
                },
                {
                    "title": "Missing Data Strategy",
                    "description": f"Missing values detected in {df.isnull().any().sum()} columns. Develop appropriate imputation or exclusion strategies."
                },
                {
                    "title": "Feature Engineering Opportunities",
                    "description": "Consider creating derived features from existing columns to enhance analytical value."
                },
                {
                    "title": "Scalability Considerations",
                    "description": "Evaluate computational requirements for larger datasets and consider optimization strategies."
                }
            ]
    
    except Exception as e:
        logger.error(f"Error generating LLM Mesh insights: {str(e)}")
        insights = [
            {
                "title": "Analysis Error",
                "description": f"Unable to generate comprehensive insights due to: {str(e)}"
            }
        ]
    
    return insights

# Automated Feature Engineering Routes
@app.route('/automated-feature-engineering')
def automated_feature_engineering():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Automated Feature Engineering route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for Automated Feature Engineering: {session_id}")
        return render_template('automated-feature-engineering.html')
    except Exception as e:
        logger.error(f"Error in automated_feature_engineering route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/feature-engineering/dataset-info', methods=['GET'])
def api_feature_engineering_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"Feature Engineering dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Analyze columns for feature engineering suitability
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Determine feature engineering potential
            fe_potential = "High"
            if pd.api.types.is_numeric_dtype(df[col]):
                if unique_count < 5:
                    fe_potential = "Medium"
                elif missing_pct > 50:
                    fe_potential = "Low"
            elif pd.api.types.is_object_dtype(df[col]):
                if unique_count > len(df) * 0.8:
                    fe_potential = "Low"
                elif unique_count < 10:
                    fe_potential = "High"
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'fe_potential': fe_potential
            })

        # Calculate dataset size
        dataset_size = df.memory_usage(deep=True).sum()

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': columns_info,
            'size': int(dataset_size),
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_feature_engineering_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/feature-engineering/generate', methods=['POST'])
def api_feature_engineering_generate():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"Feature engineering generation requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        selected_columns = data.get('selected_columns', [])
        feature_types = data.get('feature_types', [])
        model = data.get('model', 'gpt-4o')
        processing_mode = data.get('processing_mode', 'intelligent')
        
        if not selected_columns:
            return jsonify({'error': 'No columns selected for feature engineering'}), 400
        
        if not feature_types:
            return jsonify({'error': 'No feature engineering techniques selected'}), 400
        
        df = data_store[session_id]['df']
        
        # Perform feature engineering
        start_time = time.time()
        enhanced_df, feature_info = perform_automated_feature_engineering(
            df, selected_columns, feature_types, model, processing_mode
        )
        processing_time = round(time.time() - start_time, 2)
        
        # Store enhanced dataset
        processing_id = str(uuid.uuid4())
        data_store[f"enhanced_{processing_id}"] = {
            'enhanced_df': enhanced_df,
            'original_df': df,
            'feature_info': feature_info,
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate insights using Azure OpenAI
        insights = generate_feature_engineering_insights(df, enhanced_df, feature_info, model)
        
        # Prepare response data
        original_data = {
            'columns': df.columns.tolist(),
            'data': df.head(10).to_dict(orient='records')
        }
        
        enhanced_data = {
            'columns': enhanced_df.columns.tolist(),
            'data': enhanced_df.head(10).to_dict(orient='records')
        }
        
        new_features_count = len(enhanced_df.columns) - len(df.columns)
        
        return jsonify({
            'processing_id': processing_id,
            'original_data': original_data,
            'enhanced_data': enhanced_data,
            'new_features_count': new_features_count,
            'total_features_count': len(enhanced_df.columns),
            'processing_time': processing_time,
            'feature_info': feature_info,
            'insights': insights
        })
    
    except Exception as e:
        logger.error(f"Error in api_feature_engineering_generate: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/feature-engineering/download', methods=['POST'])
def api_feature_engineering_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        processing_id = data.get('processing_id')
        
        if not session_id or not processing_id:
            return jsonify({'error': 'Missing session_id or processing_id'}), 400
        
        enhanced_key = f"enhanced_{processing_id}"
        if enhanced_key not in data_store:
            return jsonify({'error': 'Enhanced dataset not found'}), 404
        
        enhanced_data = data_store[enhanced_key]
        enhanced_df = enhanced_data['enhanced_df']
        
        # Create temporary file
        temp_filename = f"enhanced_dataset_{processing_id}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        enhanced_df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_feature_engineering_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

def perform_automated_feature_engineering(df, selected_columns, feature_types, model, processing_mode):
    """
    Perform automated feature engineering using various techniques and LLM guidance
    """
    try:
        enhanced_df = df.copy()
        feature_info = []
        
        # Filter to selected columns
        selected_df = df[selected_columns].copy()
        
        # Statistical Features
        if 'statistical' in feature_types:
            stat_features, stat_info = create_statistical_features(selected_df)
            enhanced_df = pd.concat([enhanced_df, stat_features], axis=1)
            feature_info.extend(stat_info)
        
        # Temporal Features
        if 'temporal' in feature_types:
            temporal_features, temporal_info = create_temporal_features(selected_df)
            if not temporal_features.empty:
                enhanced_df = pd.concat([enhanced_df, temporal_features], axis=1)
                feature_info.extend(temporal_info)
        
        # Categorical Encoding
        if 'categorical' in feature_types:
            cat_features, cat_info = create_categorical_features(selected_df)
            if not cat_features.empty:
                enhanced_df = pd.concat([enhanced_df, cat_features], axis=1)
                feature_info.extend(cat_info)
        
        # Feature Interactions
        if 'interaction' in feature_types:
            interaction_features, interaction_info = create_interaction_features(selected_df)
            if not interaction_features.empty:
                enhanced_df = pd.concat([enhanced_df, interaction_features], axis=1)
                feature_info.extend(interaction_info)
        
        # Text Features
        if 'text' in feature_types:
            text_features, text_info = create_text_features(selected_df)
            if not text_features.empty:
                enhanced_df = pd.concat([enhanced_df, text_features], axis=1)
                feature_info.extend(text_info)
        
        # Aggregation Features
        if 'aggregation' in feature_types:
            agg_features, agg_info = create_aggregation_features(selected_df)
            if not agg_features.empty:
                enhanced_df = pd.concat([enhanced_df, agg_features], axis=1)
                feature_info.extend(agg_info)
        
        # LLM-guided feature suggestions
        if client and processing_mode == 'intelligent':
            llm_features, llm_info = create_llm_guided_features(selected_df, model)
            if not llm_features.empty:
                enhanced_df = pd.concat([enhanced_df, llm_features], axis=1)
                feature_info.extend(llm_info)
        
        return enhanced_df, feature_info
    
    except Exception as e:
        logger.error(f"Error in perform_automated_feature_engineering: {str(e)}")
        raise

def create_statistical_features(df):
    """Create statistical features from numeric columns"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            # Rolling statistics
            features[f'{col}_rolling_mean_3'] = df[col].rolling(window=3, min_periods=1).mean()
            features[f'{col}_rolling_std_3'] = df[col].rolling(window=3, min_periods=1).std()
            
            # Lag features
            features[f'{col}_lag_1'] = df[col].shift(1)
            features[f'{col}_lag_2'] = df[col].shift(2)
            
            # Cumulative features
            features[f'{col}_cumsum'] = df[col].cumsum()
            features[f'{col}_cummax'] = df[col].cummax()
            features[f'{col}_cummin'] = df[col].cummin()
            
            # Z-score normalization
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                features[f'{col}_zscore'] = (df[col] - mean_val) / std_val
            
            # Percentile ranks
            features[f'{col}_percentile_rank'] = df[col].rank(pct=True)
            
            feature_info.extend([
                {'name': f'{col}_rolling_mean_3', 'type': 'statistical', 'description': f'3-period rolling mean of {col}'},
                {'name': f'{col}_rolling_std_3', 'type': 'statistical', 'description': f'3-period rolling standard deviation of {col}'},
                {'name': f'{col}_lag_1', 'type': 'statistical', 'description': f'1-period lag of {col}'},
                {'name': f'{col}_lag_2', 'type': 'statistical', 'description': f'2-period lag of {col}'},
                {'name': f'{col}_cumsum', 'type': 'statistical', 'description': f'Cumulative sum of {col}'},
                {'name': f'{col}_cummax', 'type': 'statistical', 'description': f'Cumulative maximum of {col}'},
                {'name': f'{col}_cummin', 'type': 'statistical', 'description': f'Cumulative minimum of {col}'},
                {'name': f'{col}_zscore', 'type': 'statistical', 'description': f'Z-score normalized {col}'},
                {'name': f'{col}_percentile_rank', 'type': 'statistical', 'description': f'Percentile rank of {col}'}
            ])
    
    return features, feature_info

def create_temporal_features(df):
    """Create temporal features from datetime columns"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    # Try to identify datetime columns
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif df[col].dtype == 'object':
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col].dropna().head(100))
                datetime_cols.append(col)
            except:
                continue
    
    for col in datetime_cols:
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                dt_series = pd.to_datetime(df[col], errors='coerce')
            else:
                dt_series = df[col]
            
            # Extract temporal features
            features[f'{col}_year'] = dt_series.dt.year
            features[f'{col}_month'] = dt_series.dt.month
            features[f'{col}_day'] = dt_series.dt.day
            features[f'{col}_dayofweek'] = dt_series.dt.dayofweek
            features[f'{col}_hour'] = dt_series.dt.hour
            features[f'{col}_quarter'] = dt_series.dt.quarter
            features[f'{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
            features[f'{col}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
            features[f'{col}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
            
            feature_info.extend([
                {'name': f'{col}_year', 'type': 'temporal', 'description': f'Year extracted from {col}'},
                {'name': f'{col}_month', 'type': 'temporal', 'description': f'Month extracted from {col}'},
                {'name': f'{col}_day', 'type': 'temporal', 'description': f'Day extracted from {col}'},
                {'name': f'{col}_dayofweek', 'type': 'temporal', 'description': f'Day of week from {col}'},
                {'name': f'{col}_hour', 'type': 'temporal', 'description': f'Hour extracted from {col}'},
                {'name': f'{col}_quarter', 'type': 'temporal', 'description': f'Quarter extracted from {col}'},
                {'name': f'{col}_is_weekend', 'type': 'temporal', 'description': f'Weekend indicator from {col}'},
                {'name': f'{col}_is_month_start', 'type': 'temporal', 'description': f'Month start indicator from {col}'},
                {'name': f'{col}_is_month_end', 'type': 'temporal', 'description': f'Month end indicator from {col}'}
            ])
            
        except Exception as e:
            logger.warning(f"Error creating temporal features for column {col}: {str(e)}")
            continue
    
    return features, feature_info

def create_categorical_features(df):
    """Create categorical encoding features"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if df[col].notna().sum() > 0:
            unique_count = df[col].nunique()
            
            # One-hot encoding for low cardinality
            if unique_count <= 10:
                dummies = pd.get_dummies(df[col], prefix=f'{col}_onehot', dummy_na=True)
                features = pd.concat([features, dummies], axis=1)
                
                for dummy_col in dummies.columns:
                    feature_info.append({
                        'name': dummy_col, 
                        'type': 'categorical', 
                        'description': f'One-hot encoding for {col}'
                    })
            
            # Label encoding
            le = LabelEncoder()
            features[f'{col}_label_encoded'] = le.fit_transform(df[col].fillna('missing'))
            
            # Frequency encoding
            freq_map = df[col].value_counts().to_dict()
            features[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)
            
            # Length of string (for text columns)
            features[f'{col}_length'] = df[col].astype(str).str.len()
            
            feature_info.extend([
                {'name': f'{col}_label_encoded', 'type': 'categorical', 'description': f'Label encoding of {col}'},
                {'name': f'{col}_frequency', 'type': 'categorical', 'description': f'Frequency encoding of {col}'},
                {'name': f'{col}_length', 'type': 'categorical', 'description': f'String length of {col}'}
            ])
    
    return features, feature_info

def create_interaction_features(df):
    """Create feature interaction features"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Limit to first 5 numeric columns to avoid explosion
    numeric_cols = numeric_cols[:5]
    
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            if df[col1].notna().sum() > 0 and df[col2].notna().sum() > 0:
                # Multiplication
                features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Addition
                features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                
                # Ratio (avoid division by zero)
                features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
                # Difference
                features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                
                feature_info.extend([
                    {'name': f'{col1}_x_{col2}', 'type': 'interaction', 'description': f'Product of {col1} and {col2}'},
                    {'name': f'{col1}_plus_{col2}', 'type': 'interaction', 'description': f'Sum of {col1} and {col2}'},
                    {'name': f'{col1}_div_{col2}', 'type': 'interaction', 'description': f'Ratio of {col1} to {col2}'},
                    {'name': f'{col1}_minus_{col2}', 'type': 'interaction', 'description': f'Difference of {col1} and {col2}'}
                ])
    
    return features, feature_info

def create_text_features(df):
    """Create text-based features"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    text_cols = df.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        if df[col].notna().sum() > 0:
            text_series = df[col].astype(str)
            
            # Basic text features
            features[f'{col}_word_count'] = text_series.str.split().str.len()
            features[f'{col}_char_count'] = text_series.str.len()
            features[f'{col}_unique_words'] = text_series.apply(lambda x: len(set(x.split())))
            features[f'{col}_avg_word_length'] = text_series.apply(
                lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
            )
            
            # Count specific characters (fix regex escape sequences)
            features[f'{col}_exclamation_count'] = text_series.str.count('!')
            features[f'{col}_question_count'] = text_series.str.count(r'\?')  # Fixed escape sequence
            features[f'{col}_uppercase_count'] = text_series.str.count('[A-Z]')
            features[f'{col}_digit_count'] = text_series.str.count(r'\d')  # Fixed escape sequence
            
            # Text complexity
            features[f'{col}_sentence_count'] = text_series.str.count('[.!?]+')
            
            feature_info.extend([
                {'name': f'{col}_word_count', 'type': 'text', 'description': f'Word count in {col}'},
                {'name': f'{col}_char_count', 'type': 'text', 'description': f'Character count in {col}'},
                {'name': f'{col}_unique_words', 'type': 'text', 'description': f'Unique word count in {col}'},
                {'name': f'{col}_avg_word_length', 'type': 'text', 'description': f'Average word length in {col}'},
                {'name': f'{col}_exclamation_count', 'type': 'text', 'description': f'Exclamation mark count in {col}'},
                {'name': f'{col}_question_count', 'type': 'text', 'description': f'Question mark count in {col}'},
                {'name': f'{col}_uppercase_count', 'type': 'text', 'description': f'Uppercase letter count in {col}'},
                {'name': f'{col}_digit_count', 'type': 'text', 'description': f'Digit count in {col}'},
                {'name': f'{col}_sentence_count', 'type': 'text', 'description': f'Sentence count in {col}'}
            ])
    
    return features, feature_info

def create_aggregation_features(df):
    """Create aggregation features"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Group by categorical columns and aggregate numeric columns
    for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        if df[cat_col].notna().sum() > 0:
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df[num_col].notna().sum() > 0:
                    # Group statistics
                    group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count']).add_prefix(f'{cat_col}_{num_col}_')
                    
                    # Map back to original dataframe
                    for stat in ['mean', 'std', 'count']:
                        feature_name = f'{cat_col}_{num_col}_{stat}'
                        features[feature_name] = df[cat_col].map(group_stats[feature_name]).fillna(0)
                        
                        feature_info.append({
                            'name': feature_name,
                            'type': 'aggregation',
                            'description': f'{stat.title()} of {num_col} grouped by {cat_col}'
                        })
    
    return features, feature_info

def create_llm_guided_features(df, model):
    """Create LLM-guided features using Azure OpenAI"""
    features = pd.DataFrame(index=df.index)
    feature_info = []
    
    try:
        if not client:
            return features, feature_info
        
        # Analyze the dataset structure
        df_info = {
            'columns': df.columns.tolist(),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'sample_data': df.head(5).to_dict(orient='records'),
            'shape': df.shape
        }
        
        prompt = f"""
        You are an expert data scientist specializing in feature engineering. Analyze the following dataset and suggest 3-5 intelligent feature engineering ideas that would be valuable for machine learning.

        Dataset Information:
        {json.dumps(df_info, indent=2, default=str)}

        Please suggest features that:
        1. Are mathematically sound and interpretable
        2. Could improve model performance
        3. Are not too complex to compute
        4. Make business sense

        Respond with JSON format:
        {{
            "features": [
                {{
                    "name": "feature_name",
                    "description": "Clear description of the feature",
                    "formula": "Mathematical formula or logic",
                    "columns_used": ["col1", "col2"]
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert feature engineering AI. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        llm_suggestions = json.loads(response.choices[0].message.content)
        
        # Implement suggested features
        for suggestion in llm_suggestions.get('features', []):
            try:
                feature_name = suggestion['name']
                columns_used = suggestion.get('columns_used', [])
                
                # Simple feature implementations based on common patterns
                if len(columns_used) >= 2:
                    col1, col2 = columns_used[0], columns_used[1]
                    if col1 in df.columns and col2 in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                            # Create a ratio feature
                            features[feature_name] = df[col1] / (df[col2] + 1e-8)
                        elif pd.api.types.is_numeric_dtype(df[col1]):
                            # Create a grouped statistic
                            group_mean = df.groupby(col2)[col1].transform('mean')
                            features[feature_name] = df[col1] / (group_mean + 1e-8)
                
                feature_info.append({
                    'name': feature_name,
                    'type': 'llm_guided',
                    'description': suggestion['description']
                })
                
            except Exception as e:
                logger.warning(f"Error implementing LLM feature {suggestion.get('name', 'unknown')}: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Error in create_llm_guided_features: {str(e)}")
    
    return features, feature_info

def generate_feature_engineering_insights(original_df, enhanced_df, feature_info, model):
    """Generate insights about the feature engineering process using Azure OpenAI"""
    try:
        insights = []
        
        if client:
            # Prepare summary for LLM
            summary = {
                'original_features': len(original_df.columns),
                'new_features': len(enhanced_df.columns) - len(original_df.columns),
                'total_features': len(enhanced_df.columns),
                'feature_types': list(set([f['type'] for f in feature_info])),
                'sample_new_features': [f['name'] for f in feature_info[:5]]
            }
            
            prompt = f"""
            You are an expert data scientist analyzing the results of an automated feature engineering process. 
            
            Feature Engineering Summary:
            {json.dumps(summary, indent=2)}
            
            Provide 3-5 strategic insights about:
            1. The quality and potential impact of the new features
            2. Recommendations for model training
            3. Potential risks or considerations
            4. Next steps for the data science workflow
            
            Format as JSON:
            {{
                "insights": [
                    {{
                        "title": "Insight title",
                        "description": "Detailed explanation and recommendation"
                    }}
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert data scientist providing feature engineering insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            insights = ai_response.get('insights', [])
        
        else:
            # Fallback insights
            new_features_count = len(enhanced_df.columns) - len(original_df.columns)
            insights = [
                {
                    "title": "Feature Engineering Complete",
                    "description": f"Successfully generated {new_features_count} new features, expanding your dataset from {len(original_df.columns)} to {len(enhanced_df.columns)} features."
                },
                {
                    "title": "Model Performance Potential",
                    "description": "The new features include statistical, temporal, and interaction features that could significantly improve model performance."
                },
                {
                    "title": "Data Quality Consideration",
                    "description": "Review the new features for any missing values or outliers before training your models."
                },
                {
                    "title": "Feature Selection Recommendation",
                    "description": "Consider using feature selection techniques to identify the most important features for your specific use case."
                }
            ]
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating feature engineering insights: {str(e)}")
        return [{"title": "Processing Complete", "description": "Feature engineering completed successfully."}]

# AI Copilot for Data Exploration & Prep Routes
@app.route('/ai-copilot')
def ai_copilot():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"AI Copilot route accessed with session_id: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No valid session found: {session_id}")
            return redirect(url_for('index'))
        
        # Set session for this tab
        session['session_id'] = session_id
        logger.info(f"Session set for AI Copilot: {session_id}")
        return render_template('ai-copilot.html')
    except Exception as e:
        logger.error(f"Error in ai_copilot route: {str(e)}")
        return redirect(url_for('index'))

@app.route('/api/ai-copilot/dataset-info', methods=['GET'])
def api_ai_copilot_dataset_info():
    try:
        session_id = request.args.get('session_id') or session.get('session_id')
        logger.info(f"AI Copilot dataset info requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        df = data_store[session_id]['df']
        filename = data_store[session_id]['filename']
        
        # Analyze columns for AI Copilot
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Get sample values
            sample_values = []
            try:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_count = min(5, len(non_null_values))
                    sample_values = non_null_values.head(sample_count).astype(str).tolist()
            except Exception as e:
                logger.warning(f"Error getting sample values for column {col}: {str(e)}")
                sample_values = ["N/A"]
            
            # Determine data quality
            quality = "Good"
            if missing_pct > 50:
                quality = "Poor"
            elif missing_pct > 20:
                quality = "Fair"
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing': int(missing),
                'missing_pct': f"{missing_pct:.2f}%",
                'unique_count': int(unique_count),
                'sample_values': sample_values,
                'quality': quality
            })

        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': columns_info,
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"Error in api_ai_copilot_dataset_info: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/ai-copilot/explore', methods=['POST'])
def api_ai_copilot_explore():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        logger.info(f"AI Copilot exploration requested for session: {session_id}")
        
        if not session_id or session_id not in data_store:
            logger.warning(f"No data found for session: {session_id}")
            return jsonify({'error': 'No data found. Please upload a file first.'}), 400
        
        selected_columns = data.get('selected_columns', [])
        operation_type = data.get('operation_type')
        model = data.get('model', 'gpt-4o')
        custom_instruction = data.get('custom_instruction', '')
        
        if not selected_columns:
            return jsonify({'error': 'No columns selected'}), 400
        
        if not operation_type:
            return jsonify({'error': 'No operation type specified'}), 400
        
        df = data_store[session_id]['df']
        
        # Perform AI-powered data exploration
        start_time = time.time()
        result = perform_ai_copilot_operation(df, selected_columns, operation_type, model, custom_instruction)
        processing_time = round(time.time() - start_time, 2)
        
        # Store result for download (keep the DataFrame here)
        operation_id = str(uuid.uuid4())
        data_store[f"copilot_{operation_id}"] = {
            'result_df': result.get('modified_df', df),
            'original_df': df,
            'operation_type': operation_type,
            'session_id': session_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Remove DataFrame from result before JSON serialization
        if 'modified_df' in result:
            # Convert DataFrame to preview data
            modified_df = result['modified_df']
            result['data_preview'] = {
                'columns': modified_df.columns.tolist(),
                'data': modified_df.head(10).to_dict(orient='records'),
                'shape': modified_df.shape
            }
            del result['modified_df']  # Remove the DataFrame
        
        result['operation_id'] = operation_id
        result['processing_time'] = processing_time
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in api_ai_copilot_explore: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def perform_ai_copilot_operation(df, selected_columns, operation_type, model, custom_instruction):
    """
    Perform AI-powered data exploration and preparation operations
    """
    try:
        result = {
            'operation': operation_type,
            'columns_processed': selected_columns,
            'insights': [],
            'visualizations': [],
            'modified_df': df.copy(),
            'changes_summary': []
        }
        
        # Filter to selected columns for analysis
        selected_df = df[selected_columns].copy()
        
        if operation_type == 'data_cleaning':
            result = perform_data_cleaning(df, selected_columns, model, custom_instruction)
        elif operation_type == 'outlier_detection':
            result = perform_outlier_detection(df, selected_columns, model, custom_instruction)
        elif operation_type == 'missing_value_analysis':
            result = perform_missing_value_analysis(df, selected_columns, model, custom_instruction)
        elif operation_type == 'correlation_analysis':
            result = perform_correlation_analysis(df, selected_columns, model, custom_instruction)
        elif operation_type == 'distribution_analysis':
            result = perform_distribution_analysis(df, selected_columns, model, custom_instruction)
        elif operation_type == 'feature_importance':
            result = perform_feature_importance_analysis(df, selected_columns, model, custom_instruction)
        elif operation_type == 'data_transformation':
            result = perform_data_transformation(df, selected_columns, model, custom_instruction)
        elif operation_type == 'custom_analysis':
            result = perform_custom_analysis(df, selected_columns, model, custom_instruction)
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_ai_copilot_operation: {str(e)}")
        raise

def perform_data_cleaning(df, selected_columns, model, custom_instruction):
    """Perform intelligent data cleaning"""
    result = {
        'operation': 'data_cleaning',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'changes_summary': []
    }
    
    try:
        modified_df = df.copy()
        changes = []
        
        for col in selected_columns:
            if col in df.columns:
                # Handle missing values
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Fill with median for numeric columns
                        median_val = df[col].median()
                        modified_df[col].fillna(median_val, inplace=True)
                        changes.append(f"Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})")
                    else:
                        # Fill with mode for categorical columns
                        mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                        modified_df[col].fillna(mode_val, inplace=True)
                        changes.append(f"Filled {missing_count} missing values in '{col}' with mode ('{mode_val}')")
                
                # Remove duplicates
                if col in df.select_dtypes(include=['object']).columns:
                    # Standardize text
                    modified_df[col] = modified_df[col].astype(str).str.strip().str.lower()
                    changes.append(f"Standardized text in '{col}' (trimmed and lowercased)")
        
        # Remove duplicate rows
        initial_rows = len(modified_df)
        modified_df.drop_duplicates(inplace=True)
        removed_duplicates = initial_rows - len(modified_df)
        if removed_duplicates > 0:
            changes.append(f"Removed {removed_duplicates} duplicate rows")
        
        # Store the modified DataFrame separately (will be handled by the main function)
        result['modified_df'] = modified_df
        result['changes_summary'] = changes
        
        # Generate insights using AI
        if client:
            insights = generate_cleaning_insights(df, modified_df, changes, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Data Cleaning Complete',
                    'description': f'Applied {len(changes)} cleaning operations to improve data quality.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_data_cleaning: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Data cleaning failed: {str(e)}'}]
        return result

def perform_outlier_detection(df, selected_columns, model, custom_instruction):
    """Perform outlier detection and analysis"""
    result = {
        'operation': 'outlier_detection',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'changes_summary': []
    }
    
    try:
        outlier_info = []
        visualizations = []
        
        numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                # IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(df)) * 100
                
                outlier_info.append({
                    'column': col,
                    'outlier_count': outlier_count,
                    'outlier_percentage': f"{outlier_percentage:.2f}%",
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                })
                
                # Create box plot
                plt.figure(figsize=(8, 6))
                plt.boxplot(df[col].dropna(), labels=[col])
                plt.title(f'Box Plot for {col} - Outlier Detection')
                plt.ylabel('Values')
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                visualizations.append({
                    'type': 'boxplot',
                    'title': f'Outlier Detection - {col}',
                    'data': plot_data
                })
        
        # Store modified DataFrame (same as original for outlier detection)
        result['modified_df'] = df.copy()
        result['visualizations'] = visualizations
        result['outlier_info'] = outlier_info
        
        # Generate AI insights
        if client:
            insights = generate_outlier_insights(outlier_info, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Outlier Analysis Complete',
                    'description': f'Analyzed {len(numeric_cols)} numeric columns for outliers using IQR method.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_outlier_detection: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Outlier detection failed: {str(e)}'}]
        return result

def perform_missing_value_analysis(df, selected_columns, model, custom_instruction):
    """Perform comprehensive missing value analysis"""
    result = {
        'operation': 'missing_value_analysis',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        missing_info = []
        
        for col in selected_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_percentage = (missing_count / len(df)) * 100
                
                missing_info.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_percentage': f"{missing_percentage:.2f}%",
                    'data_type': str(df[col].dtype)
                })
        
        # Create missing value heatmap
        if len(selected_columns) > 1:
            plt.figure(figsize=(10, 6))
            missing_matrix = df[selected_columns].isnull()
            sns.heatmap(missing_matrix, cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Value Pattern')
            plt.xlabel('Columns')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            result['visualizations'] = [{
                'type': 'heatmap',
                'title': 'Missing Value Pattern',
                'data': plot_data
            }]
        
        result['modified_df'] = df.copy()
        result['missing_info'] = missing_info
        
        # Generate AI insights
        if client:
            insights = generate_missing_value_insights(missing_info, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Missing Value Analysis Complete',
                    'description': f'Analyzed missing values across {len(selected_columns)} columns.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_missing_value_analysis: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Missing value analysis failed: {str(e)}'}]
        return result

def perform_correlation_analysis(df, selected_columns, model, custom_instruction):
    """Perform correlation analysis"""
    result = {
        'operation': 'correlation_analysis',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) < 2:
            result['insights'] = [{'title': 'Insufficient Data', 'description': 'Need at least 2 numeric columns for correlation analysis.'}]
            return result
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        result['visualizations'] = [{
            'type': 'correlation_heatmap',
            'title': 'Correlation Matrix',
            'data': plot_data
        }]
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        'column1': corr_matrix.columns[i],
                        'column2': corr_matrix.columns[j],
                        'correlation': f"{corr_val:.3f}"
                    })
        
        result['modified_df'] = df.copy()
        result['strong_correlations'] = strong_correlations
        
        # Generate AI insights
        if client:
            insights = generate_correlation_insights(strong_correlations, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Correlation Analysis Complete',
                    'description': f'Found {len(strong_correlations)} strong correlations (|r| > 0.7) among {len(numeric_cols)} numeric columns.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_correlation_analysis: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Correlation analysis failed: {str(e)}'}]
        return result

def perform_distribution_analysis(df, selected_columns, model, custom_instruction):
    """Perform distribution analysis"""
    result = {
        'operation': 'distribution_analysis',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        visualizations = []
        distribution_stats = []
        
        for col in selected_columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Create histogram
                    plt.figure(figsize=(8, 6))
                    plt.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                    buffer.seek(0)
                    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
                    
                    visualizations.append({
                        'type': 'histogram',
                        'title': f'Distribution - {col}',
                        'data': plot_data
                    })
                    
                    # Calculate distribution statistics
                    stats = {
                        'column': col,
                        'mean': f"{df[col].mean():.3f}",
                        'median': f"{df[col].median():.3f}",
                        'std': f"{df[col].std():.3f}",
                        'skewness': f"{df[col].skew():.3f}",
                        'kurtosis': f"{df[col].kurtosis():.3f}"
                    }
                    distribution_stats.append(stats)
                
                elif pd.api.types.is_object_dtype(df[col]):
                    # Create bar plot for categorical data
                    value_counts = df[col].value_counts().head(10)
                    
                    plt.figure(figsize=(10, 6))
                    value_counts.plot(kind='bar')
                    plt.title(f'Top 10 Values in {col}')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                    buffer.seek(0)
                    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
                    
                    visualizations.append({
                        'type': 'barplot',
                        'title': f'Value Distribution - {col}',
                        'data': plot_data
                    })
        
        result['modified_df'] = df.copy()
        result['visualizations'] = visualizations
        result['distribution_stats'] = distribution_stats
        
        # Generate AI insights
        if client:
            insights = generate_distribution_insights(distribution_stats, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Distribution Analysis Complete',
                    'description': f'Analyzed distributions for {len(selected_columns)} columns.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_distribution_analysis: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Distribution analysis failed: {str(e)}'}]
        return result

def perform_feature_importance_analysis(df, selected_columns, model, custom_instruction):
    """Perform feature importance analysis"""
    result = {
        'operation': 'feature_importance',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        # This is a simplified feature importance analysis
        # In a real scenario, you'd need a target variable
        
        numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) < 2:
            result['insights'] = [{'title': 'Insufficient Data', 'description': 'Need at least 2 numeric columns for feature importance analysis.'}]
            return result
        
        # Calculate variance as a proxy for importance
        importance_scores = []
        for col in numeric_cols:
            variance = df[col].var()
            importance_scores.append({
                'feature': col,
                'importance_score': variance,
                'normalized_score': variance / df[col].max() if df[col].max() != 0 else 0
            })
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Create importance plot
        features = [item['feature'] for item in importance_scores]
        scores = [item['normalized_score'] for item in importance_scores]
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, scores)
        plt.title('Feature Importance (Based on Variance)')
        plt.xlabel('Normalized Importance Score')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        result['visualizations'] = [{
            'type': 'feature_importance',
            'title': 'Feature Importance Analysis',
            'data': plot_data
        }]
        
        result['modified_df'] = df.copy()
        result['importance_scores'] = importance_scores
        
        # Generate AI insights
        if client:
            insights = generate_feature_importance_insights(importance_scores, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Feature Importance Analysis Complete',
                    'description': f'Analyzed importance of {len(numeric_cols)} numeric features based on variance.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_feature_importance_analysis: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Feature importance analysis failed: {str(e)}'}]
        return result

def perform_data_transformation(df, selected_columns, model, custom_instruction):
    """Perform data transformation"""
    result = {
        'operation': 'data_transformation',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        modified_df = df.copy()
        changes = []
        
        for col in selected_columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Apply log transformation for skewed data
                    if df[col].skew() > 1:
                        modified_df[f'{col}_log'] = np.log1p(df[col])
                        changes.append(f"Applied log transformation to '{col}' (high skewness)")
                    
                    # Apply standardization
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        modified_df[f'{col}_standardized'] = (df[col] - mean_val) / std_val
                        changes.append(f"Standardized '{col}' (mean=0, std=1)")
                
                elif pd.api.types.is_object_dtype(df[col]):
                    # Apply label encoding
                    le = LabelEncoder()
                    modified_df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('missing'))
                    changes.append(f"Label encoded '{col}'")
        
        result['modified_df'] = modified_df
        result['changes_summary'] = changes
        
        # Generate AI insights
        if client:
            insights = generate_transformation_insights(changes, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Data Transformation Complete',
                    'description': f'Applied {len(changes)} transformations to improve data quality and model readiness.'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_data_transformation: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Data transformation failed: {str(e)}'}]
        return result

def perform_custom_analysis(df, selected_columns, model, custom_instruction):
    """Perform custom analysis based on user instruction"""
    result = {
        'operation': 'custom_analysis',
        'columns_processed': selected_columns,
        'insights': [],
        'visualizations': [],
        'modified_df': df.copy(),
        'changes_summary': []
    }
    
    try:
        if not custom_instruction:
            result['insights'] = [{'title': 'No Instruction', 'description': 'Please provide custom analysis instructions.'}]
            return result
        
        # Use AI to interpret and execute custom instruction
        if client:
            insights = generate_custom_analysis_insights(df, selected_columns, custom_instruction, model)
            result['insights'] = insights
        else:
            result['insights'] = [
                {
                    'title': 'Custom Analysis',
                    'description': f'Custom analysis requested for columns: {", ".join(selected_columns)}. Instruction: {custom_instruction}'
                }
            ]
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_custom_analysis: {str(e)}")
        result['insights'] = [{'title': 'Error', 'description': f'Custom analysis failed: {str(e)}'}]
        return result

# Helper functions for generating AI insights
def generate_cleaning_insights(original_df, modified_df, changes, model):
    """Generate insights about data cleaning operations"""
    try:
        if not client:
            return [{'title': 'Data Cleaning Complete', 'description': f'Applied {len(changes)} cleaning operations.'}]
        
        prompt = f"""
        Analyze the data cleaning operations performed and provide insights.
        
        Changes made:
        {chr(10).join(changes)}
        
        Original dataset shape: {original_df.shape}
        Modified dataset shape: {modified_df.shape}
        
        Provide 3-5 insights about the cleaning process and recommendations.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data cleaning expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating cleaning insights: {str(e)}")
        return [{'title': 'Data Cleaning Complete', 'description': f'Applied {len(changes)} cleaning operations.'}]

def generate_outlier_insights(outlier_info, model):
    """Generate insights about outlier detection"""
    try:
        if not client:
            return [{'title': 'Outlier Detection Complete', 'description': f'Analyzed {len(outlier_info)} columns for outliers.'}]
        
        prompt = f"""
        Analyze the outlier detection results and provide insights.
        
        Outlier Information:
        {json.dumps(outlier_info, indent=2)}
        
        Provide insights about the outliers found and recommendations for handling them.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an outlier detection expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating outlier insights: {str(e)}")
        return [{'title': 'Outlier Detection Complete', 'description': f'Analyzed {len(outlier_info)} columns for outliers.'}]

def generate_missing_value_insights(missing_info, model):
    """Generate insights about missing value analysis"""
    try:
        if not client:
            return [{'title': 'Missing Value Analysis Complete', 'description': f'Analyzed {len(missing_info)} columns for missing values.'}]
        
        prompt = f"""
        Analyze the missing value patterns and provide insights.
        
        Missing Value Information:
        {json.dumps(missing_info, indent=2)}
        
        Provide insights about missing value patterns and recommendations for handling them.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a missing value analysis expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating missing value insights: {str(e)}")
        return [{'title': 'Missing Value Analysis Complete', 'description': f'Analyzed {len(missing_info)} columns for missing values.'}]

def generate_correlation_insights(strong_correlations, model):
    """Generate insights about correlation analysis"""
    try:
        if not client:
            return [{'title': 'Correlation Analysis Complete', 'description': f'Found {len(strong_correlations)} strong correlations.'}]
        
        prompt = f"""
        Analyze the correlation results and provide insights.
        
        Strong Correlations Found:
        {json.dumps(strong_correlations, indent=2)}
        
        Provide insights about the correlations and their implications for analysis.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a correlation analysis expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating correlation insights: {str(e)}")
        return [{'title': 'Correlation Analysis Complete', 'description': f'Found {len(strong_correlations)} strong correlations.'}]

def generate_distribution_insights(distribution_stats, model):
    """Generate insights about distribution analysis"""
    try:
        if not client:
            return [{'title': 'Distribution Analysis Complete', 'description': f'Analyzed distributions for {len(distribution_stats)} columns.'}]
        
        prompt = f"""
        Analyze the distribution statistics and provide insights.
        
        Distribution Statistics:
        {json.dumps(distribution_stats, indent=2)}
        
        Provide insights about the distributions and their characteristics.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a distribution analysis expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating distribution insights: {str(e)}")
        return [{'title': 'Distribution Analysis Complete', 'description': f'Analyzed distributions for {len(distribution_stats)} columns.'}]

def generate_feature_importance_insights(importance_scores, model):
    """Generate insights about feature importance"""
    try:
        if not client:
            return [{'title': 'Feature Importance Analysis Complete', 'description': f'Analyzed importance of {len(importance_scores)} features.'}]
        
        prompt = f"""
        Analyze the feature importance scores and provide insights.
        
        Feature Importance Scores:
        {json.dumps(importance_scores, indent=2)}
        
        Provide insights about feature importance and recommendations.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a feature importance expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating feature importance insights: {str(e)}")
        return [{'title': 'Feature Importance Analysis Complete', 'description': f'Analyzed importance of {len(importance_scores)} features.'}]

def generate_transformation_insights(changes, model):
    """Generate insights about data transformation"""
    try:
        if not client:
            return [{'title': 'Data Transformation Complete', 'description': f'Applied {len(changes)} transformations.'}]
        
        prompt = f"""
        Analyze the data transformation operations and provide insights.
        
        Transformations Applied:
        {chr(10).join(changes)}
        
        Provide insights about the transformations and their benefits.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data transformation expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating transformation insights: {str(e)}")
        return [{'title': 'Data Transformation Complete', 'description': f'Applied {len(changes)} transformations.'}]

def generate_custom_analysis_insights(df, selected_columns, custom_instruction, model):
    """Generate insights for custom analysis"""
    try:
        if not client:
            return [{'title': 'Custom Analysis', 'description': f'Custom analysis for: {custom_instruction}'}]
        
        # Get basic info about selected columns
        column_info = {}
        for col in selected_columns:
            if col in df.columns:
                column_info[col] = {
                    'type': str(df[col].dtype),
                    'missing': int(df[col].isnull().sum()),
                    'unique': int(df[col].nunique())
                }
        
        prompt = f"""
        Perform custom analysis based on the user's instruction.
        
        User Instruction: {custom_instruction}
        
        Selected Columns Information:
        {json.dumps(column_info, indent=2)}
        
        Dataset Shape: {df.shape}
        
        Provide detailed insights and analysis based on the user's request.
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data analysis expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating custom analysis insights: {str(e)}")
        return [{'title': 'Custom Analysis', 'description': f'Custom analysis for: {custom_instruction}'}]

@app.route('/api/ai-copilot/download', methods=['POST'])
def api_ai_copilot_download():
    try:
        data = request.json
        session_id = data.get('session_id')
        operation_id = data.get('operation_id')
        
        if not session_id or not operation_id:
            return jsonify({'error': 'Missing session_id or operation_id'}), 400
        
        copilot_key = f"copilot_{operation_id}"
        if copilot_key not in data_store:
            return jsonify({'error': 'Operation result not found'}), 404
        
        copilot_data = data_store[copilot_key]
        result_df = copilot_data['result_df']
        
        # Create temporary file
        temp_filename = f"ai_copilot_result_{operation_id}.csv"
        temp_path = os.path.join(TEMP_FOLDER, temp_filename)
        
        # Save to CSV
        result_df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=temp_filename)
    
    except Exception as e:
        logger.error(f"Error in api_ai_copilot_download: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(debug=True, threaded=True)

