from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, NumberRange, Optional
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize model lazily
semantic_model = None

def get_semantic_model():
    global semantic_model
    if semantic_model is None:
        # Use a smaller model to reduce size
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return semantic_model

class DebateCardForm(FlaskForm):
    source_type = StringField('Source Type', default='text')
    url = StringField('Article URL', validators=[Optional()])
    custom_text = TextAreaField('Custom Text', validators=[Optional()])
    tagline = StringField('Tagline', validators=[DataRequired()])
    threshold = DecimalField('Relevance Threshold (0.0-1.0)',
                           validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    submit = SubmitField('Process')

def extract_content(url: str) -> dict:
    headers = {'User-Agent': 'LDCardCutter (+https://example.com)'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve content: {response.status_code}")
    soup = BeautifulSoup(response.content, 'html.parser')
    for element in soup(['script', 'style', 'nav', 'footer']):
        element.decompose()
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    return {
        'paragraphs': paragraphs,
        'source_url': url,
        'accessed_date': datetime.now().isoformat()
    }

def process_text(paragraphs: List[str], tagline: str, threshold: float) -> List[Dict]:
    semantic_model = get_semantic_model()
    
    # Process in batches to reduce memory usage
    batch_size = 32
    processed_chunks = []
    
    for i in range(0, len(paragraphs), batch_size):
        batch_paragraphs = paragraphs[i:i + batch_size]
        sentences = []
        for paragraph in batch_paragraphs:
            sentences.extend(sent_tokenize(paragraph))
        
        # Process sentences in batches
        for j in range(0, len(sentences), batch_size):
            sentence_batch = sentences[j:j + batch_size]
            
            # Encode tagline once per batch
            tagline_embedding = semantic_model.encode([tagline])
            
            # Process sentences
            sentence_embeddings = semantic_model.encode(sentence_batch)
            scores = util.cos_sim(
                torch.tensor(tagline_embedding),
                torch.tensor(sentence_embeddings)
            ).numpy()[0]
            
            processed_chunks.extend([
                {'text': sentence, 'score': float(score)}
                for sentence, score in zip(sentence_batch, scores)
            ])
    
    return processed_chunks

@app.route('/', methods=['GET', 'POST'])
def index():
    form = DebateCardForm()
    results = None
    if form.validate_on_submit():
        try:
            if form.source_type.data == 'url':
                content_data = extract_content(form.url.data)
            else:
                content_data = {
                    'paragraphs': [form.custom_text.data],
                    'source_url': 'Custom Text',
                    'accessed_date': datetime.now().isoformat()
                }
            
            processed_paragraphs = []
            for paragraph in content_data['paragraphs']:
                sentences = sent_tokenize(paragraph)
                processed_sentences = []
                semantic_model = get_semantic_model()
                tagline_embedding = semantic_model.encode([form.tagline.data])
                
                for sentence in sentences:
                    sentence_embedding = semantic_model.encode([sentence])
                    semantic_score = np.dot(tagline_embedding[0], sentence_embedding[0]) / (
                        np.linalg.norm(tagline_embedding[0]) * 
                        np.linalg.norm(sentence_embedding[0])
                    )
                    processed_sentences.append({
                        'text': sentence,
                        'score': float(semantic_score)
                    })
                
                processed_paragraphs.append(processed_sentences)
            
            html_output = []
            for paragraph in processed_paragraphs:
                paragraph_html = []
                for sentence in paragraph:
                    base_style = 'font-family: Calibri, sans-serif !important; font-size: 11pt !important;'
                    if sentence['score'] >= form.threshold.data:
                        paragraph_html.append(
                            f'<span style="{base_style}text-decoration: underline !important;">'
                            f'{sentence["text"]}</span>'
                        )
                    else:
                        paragraph_html.append(
                            f'<span style="{base_style}">{sentence["text"]}</span>'
                        )
                html_output.append(' '.join(paragraph_html))
            
            results = {
                'text': '<br>'.join(html_output),
                'source_url': content_data['source_url'],
                'accessed_date': content_data['accessed_date']
            }
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return render_template('index.html', form=form, results=results)

if __name__ == '__main__':
    app.run()
