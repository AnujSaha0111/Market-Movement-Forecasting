import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

def _get_pipeline():
    from transformers import pipeline
    return pipeline

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class SentimentAnalyzer:
    
    def __init__(self, use_finbert: bool = True, device: str = None):
        self.use_finbert = use_fassification.from_pretrained(model_name)
                
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if device == 'cuda' else -1,
                    max_length=512,
                    truncation=True
                )
                
                logger.info("FinBERT model loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading FinBERT: {str(e)}")
                logger.info("Falling back to VADER and TextBlob only")
                self.use_finbert = False
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        if pd.isna(text) or not text:
            return {'vader_compound': 0.0, 'vader_pos': 0.0, 'vader_neu': 0.0, 'vader_neg': 0.0}
        
        scores = self.vader.polarity_scores(str(text))
        return {
            'vader_compound': scores['compound'],
            'vader_pos': scores['pos'],
            'vader_neu': scores['neu'],
            'vader_neg': scores['neg']
        }
    
at]:
ipeline is None:
            return {'finbert_score': 0.0, 'finbert_label': 'neutral'}
        
        if pd.isna(text) or not text:
            return {'finbert_score': 0.0, 'finbert_label': 'neutral'}
        
            label = result['label'].lower()
            score = result['score']
            
            if label == 'positive':
              ore,
                'finbert_label': label
            }
            
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment: {str(e)}")
            return {'finbert_score': 0.0, 'finbert_label': 'neutral'}
    

        # VADER
        results.update(self.vader_sentiment(text))
        
        # TextBlob
        results.update(self.textblob_sentiment(text))
        
        # FinBERT
        if self.use_finbert:
            results.update(self.finbert_sentiment(text))
        
        return results
  
        sentiments = []
        for idx, text in enumerate(df[text_column]):
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} texts")
            

        sentiment_df = pd.DataFrame(sentiments)
        df = pd.concat([df, sentiment_df], axis=1)
        
        logger.info("Sentiment analysis complete")
        
        return df
    
    def aggregate_sentiment(self, df: pd.DataFrame, group_by: str = 'date',
mns if any(
            x in col for x in ['vader', 'textblob', 'finbert']
        )]
        
        if method == 'weighted_average':
            # Weight by news count if available
            if 'news_count' in df.columns:
                def weighted_avg(group):
                    weights = group['news_count']
                    return (group[sentiment_cols].multiply(weights, axis=0).sum() / weights.sum())
                
                agg_df = df.groupby(group_by).apply(weighted_avg).reset_index()
            else:
                agg_df = df.groupby(group_by)[sentiment_cols].mean().reset_index()
        
        elif method == 'mean':
            agg_df = df.groupby(group_by)[sentiment_cols].mean().reset_index()
        
        elif method == 'median':
            agg_df = df.groupby(group_by)[sentiment_cols].median().reset_index()
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        logger.info(f"Aggregated to {len(agg_df)} records")
        
        return agg_df
    
    
            df['sentiment_momentum'] = df['vader_compound'].diff()
            df['sentiment_momentum_3d'] = df['vader_compound'].diff(3)
            df['sentiment_momentum_5d'] = df['vader_compound'].diff(5)
        nd(df['vader_compound'])
        if 'textblob_polarity' in df.columns:
            sentiment_scores.append(df['textblob_polarity'])
        if 'finbert_score' in df.columns:
            sentiment_scores.append(df['finbert_score'])
        
        if sentiment_scores:
            df['sentiment_combined'] = pd.concat(sentiment_scores, axis=1).mean(axis=1)
        
        logger.info(f"Created {len([c for c in df.columns if 'sentiment' in c])} sentiment features")
        
        return df


if __name__ == "__main__":
    # Test the analyzer
    text(text)
        print(f"Sentiment: {sentiment}\n")
    
    # Test DataFrame analysis
    test_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'full_text': texts
    })
    
    result_df = analyzer.analyze_dataframe(test_df)
    print("\nDataFrame Analysis:")
    print(result_df)
