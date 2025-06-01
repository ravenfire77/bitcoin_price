import mwclient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class WikipediaSentimentAnalyzer:
    def __init__(self):
        # Initialize sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Initialize text classification for financial sentiment
        self.finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        
        self.edits_data = None
        
    def fetch_wikipedia_edits(self, page_title='Bitcoin', days_back=365):
        """Fetch Wikipedia edits for a specific page"""
        print(f"Fetching Wikipedia edits for '{page_title}' page...")
        
        site = mwclient.Site('en.wikipedia.org')
        page = site.pages[page_title]
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        edits = []
        
        # Fetch revisions
        for revision in page.revisions(limit=5000, 
                                      start=start_date.isoformat(), 
                                      end=end_date.isoformat()):
            edit = {
                'timestamp': revision['timestamp'],
                'user': revision.get('user', 'Anonymous'),
                'userid': revision.get('userid', 0),
                'comment': revision.get('comment', ''),
                'size': revision.get('size', 0),
                'revid': revision['revid'],
                'parentid': revision.get('parentid', 0),
                'minor': revision.get('minor', False)
            }
            
            # Calculate size change
            if edit['parentid'] != 0:
                edit['size_change'] = edit['size'] - revision.get('parent_size', edit['size'])
            else:
                edit['size_change'] = edit['size']
            
            edits.append(edit)
        
        # Convert to DataFrame
        df = pd.DataFrame(edits)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Fetched {len(df)} edits from the last {days_back} days")
        
        self.edits_data = df
        return df
    
    def analyze_edit_patterns(self, df):
        """Analyze editing patterns and user behavior"""
        analysis = {}
        
        # Daily edit patterns
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Edit frequency by day
        daily_edits = df.groupby('date').size()
        analysis['daily_edits'] = daily_edits
        
        # Edit frequency by hour
        hourly_edits = df.groupby('hour').size()
        analysis['hourly_pattern'] = hourly_edits
        
        # Top editors
        top_editors = df['user'].value_counts().head(20)
        analysis['top_editors'] = top_editors
        
        # Edit size statistics
        analysis['avg_edit_size'] = df['size_change'].abs().mean()
        analysis['total_content_change'] = df['size_change'].sum()
        
        # Minor vs major edits
        analysis['minor_edit_ratio'] = df['minor'].sum() / len(df)
        
        # User diversity (unique users per day)
        user_diversity = df.groupby('date')['user'].nunique()
        analysis['user_diversity'] = user_diversity
        
        return analysis
    
    def extract_edit_features(self, comment):
        """Extract features from edit comments"""
        features = {
            'length': len(comment),
            'has_url': 1 if re.search(r'https?://', comment) else 0,
            'has_revert': 1 if 'revert' in comment.lower() else 0,
            'has_vandalism': 1 if 'vandal' in comment.lower() else 0,
            'has_update': 1 if any(word in comment.lower() for word in ['update', 'add', 'fix']) else 0,
            'has_reference': 1 if 'ref' in comment.lower() or 'citation' in comment.lower() else 0,
            'is_bot': 1 if 'bot' in comment.lower() else 0
        }
        return features
    
    def analyze_sentiment(self, df):
        """Analyze sentiment of edit comments"""
        print("Analyzing sentiment of edit comments...")
        
        sentiments = []
        financial_sentiments = []
        comment_features = []
        
        for idx, comment in enumerate(df['comment'].fillna('')):
            if idx % 100 == 0:
                print(f"Processing edit {idx}/{len(df)}")
            
            # Extract features
            features = self.extract_edit_features(comment)
            comment_features.append(features)
            
            if comment.strip():
                try:
                    # General sentiment
                    truncated = comment[:512]  # Truncate for model limits
                    general_sentiment = self.sentiment_pipeline(truncated)[0]
                    sentiments.append({
                        'label': general_sentiment['label'],
                        'score': general_sentiment['score'],
                        'positive_score': general_sentiment['score'] if general_sentiment['label'] == 'POSITIVE' else 1 - general_sentiment['score']
                    })
                    
                    # Financial sentiment (if applicable)
                    if any(word in comment.lower() for word in ['price', 'market', 'value', 'invest', 'trade']):
                        fin_sentiment = self.finbert(truncated)[0]
                        financial_sentiments.append({
                            'label': fin_sentiment['label'],
                            'score': fin_sentiment['score']
                        })
                    else:
                        financial_sentiments.append({'label': 'neutral', 'score': 0.5})
                        
                except Exception as e:
                    print(f"Error processing comment: {e}")
                    sentiments.append({'label': 'NEUTRAL', 'score': 0.5, 'positive_score': 0.5})
                    financial_sentiments.append({'label': 'neutral', 'score': 0.5})
            else:
                sentiments.append({'label': 'NEUTRAL', 'score': 0.5, 'positive_score': 0.5})
                financial_sentiments.append({'label': 'neutral', 'score': 0.5})
        
        # Add sentiment data to DataFrame
        df['sentiment_label'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['score'] for s in sentiments]
        df['positive_sentiment'] = [s['positive_score'] for s in sentiments]
        
        df['fin_sentiment_label'] = [s['label'] for s in financial_sentiments]
        df['fin_sentiment_score'] = [s['score'] for s in financial_sentiments]
        
        # Add comment features
        features_df = pd.DataFrame(comment_features)
        df = pd.concat([df, features_df], axis=1)
        
        return df
    
    def aggregate_daily_sentiment(self, df):
        """Aggregate sentiment data by day"""
        print("Aggregating daily sentiment metrics...")
        
        daily_agg = df.groupby('date').agg({
            # Edit counts
            'revid': 'count',
            'minor': 'sum',
            'size_change': ['sum', 'mean', 'std'],
            
            # User metrics
            'user': lambda x: x.nunique(),
            'userid': lambda x: (x != 0).sum(),  # Registered users
            
            # Sentiment metrics
            'positive_sentiment': ['mean', 'std'],
            'sentiment_score': 'mean',
            
            # Financial sentiment
            'fin_sentiment_score': 'mean',
            
            # Comment features
            'has_url': 'sum',
            'has_revert': 'sum',
            'has_vandalism': 'sum',
            'has_update': 'sum',
            'has_reference': 'sum',
            'is_bot': 'sum',
            'length': 'mean'
        })
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() if col[1] else col[0] 
                            for col in daily_agg.columns.values]
        
        # Rename columns for clarity
        daily_agg.rename(columns={
            'revid_count': 'edit_count',
            'minor_sum': 'minor_edits',
            'user_<lambda>': 'unique_users',
            'userid_<lambda>': 'registered_users',
            'length_mean': 'avg_comment_length'
        }, inplace=True)
        
        # Calculate additional metrics
        daily_agg['major_edits'] = daily_agg['edit_count'] - daily_agg['minor_edits']
        daily_agg['minor_edit_ratio'] = daily_agg['minor_edits'] / daily_agg['edit_count']
        daily_agg['registered_user_ratio'] = daily_agg['registered_users'] / daily_agg['edit_count']
        daily_agg['controversy_score'] = (daily_agg['has_revert'] + daily_agg['has_vandalism']) / daily_agg['edit_count']
        
        # Calculate engagement score
        daily_agg['engagement_score'] = (
            daily_agg['edit_count'] * 0.3 +
            daily_agg['unique_users'] * 0.3 +
            daily_agg['has_update'] * 0.2 +
            daily_agg['has_reference'] * 0.2
        )
        
        # Rolling averages for smoothing
        for col in ['edit_count', 'positive_sentiment_mean', 'engagement_score', 'controversy_score']:
            daily_agg[f'{col}_ma7'] = daily_agg[col].rolling(window=7, min_periods=1).mean()
            daily_agg[f'{col}_ma30'] = daily_agg[col].rolling(window=30, min_periods=1).mean()
        
        return daily_agg
    
    def save_sentiment_data(self, daily_data, filename='wikipedia_edits.csv'):
        """Save the processed sentiment data"""
        daily_data.to_csv(filename)
        print(f"Saved sentiment data to {filename}")
        return filename
    
    def visualize_sentiment_trends(self, daily_data):
        """Create visualizations of sentiment trends"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Wikipedia Bitcoin Page: Edit Activity and Sentiment Analysis', fontsize=16)
        
        # 1. Edit count over time
        ax = axes[0, 0]
        ax.plot(daily_data.index, daily_data['edit_count'], alpha=0.5, label='Daily')
        ax.plot(daily_data.index, daily_data['edit_count_ma7'], label='7-day MA')
        ax.set_title('Daily Edit Count')
        ax.set_ylabel('Number of Edits')
        ax.legend()
        
        # 2. Sentiment over time
        ax = axes[0, 1]
        ax.plot(daily_data.index, daily_data['positive_sentiment_mean'], alpha=0.5, label='Daily')
        ax.plot(daily_data.index, daily_data['positive_sentiment_mean_ma7'], label='7-day MA')
        ax.set_title('Average Positive Sentiment')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        
        # 3. User engagement
        ax = axes[1, 0]
        ax.plot(daily_data.index, daily_data['unique_users'], alpha=0.5, label='Unique Users')
        ax.plot(daily_data.index, daily_data['registered_users'], alpha=0.5, label='Registered Users')
        ax.set_title('User Activity')
        ax.set_ylabel('Number of Users')
        ax.legend()
        
        # 4. Engagement score
        ax = axes[1, 1]
        ax.plot(daily_data.index, daily_data['engagement_score'], alpha=0.5, label='Daily')
        ax.plot(daily_data.index, daily_data['engagement_score_ma7'], label='7-day MA')
        ax.set_title('Engagement Score')
        ax.set_ylabel('Score')
        ax.legend()
        
        # 5. Edit types
        ax = axes[2, 0]
        ax.bar(daily_data.index, daily_data['major_edits'], label='Major', alpha=0.7)
        ax.bar(daily_data.index, daily_data['minor_edits'], bottom=daily_data['major_edits'], 
               label='Minor', alpha=0.7)
        ax.set_title('Edit Types')
        ax.set_ylabel('Number of Edits')
        ax.legend()
        
        # 6. Controversy score
        ax = axes[2, 1]
        ax.plot(daily_data.index, daily_data['controversy_score'], alpha=0.5, label='Daily')
        ax.plot(daily_data.index, daily_data['controversy_score_ma7'], label='7-day MA')
        ax.set_title('Controversy Score (Reverts & Vandalism)')
        ax.set_ylabel('Score')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('wikipedia_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_full_analysis(self, page_title='Bitcoin', days_back=365, output_file='wikipedia_edits.csv'):
        """Run the complete sentiment analysis pipeline"""
        print("Starting Wikipedia Sentiment Analysis Pipeline")
        print("=" * 50)
        
        # 1. Fetch Wikipedia edits
        edits_df = self.fetch_wikipedia_edits(page_title, days_back)
        
        # 2. Analyze edit patterns
        patterns = self.analyze_edit_patterns(edits_df)
        print(f"\nEdit Patterns Summary:")
        print(f"- Average edits per day: {patterns['daily_edits'].mean():.2f}")
        print(f"- Top editor: {patterns['top_editors'].index[0]} ({patterns['top_editors'].iloc[0]} edits)")
        print(f"- Minor edit ratio: {patterns['minor_edit_ratio']:.2%}")
        
        # 3. Analyze sentiment
        edits_with_sentiment = self.analyze_sentiment(edits_df)
        
        # 4. Aggregate daily
        daily_sentiment = self.aggregate_daily_sentiment(edits_with_sentiment)
        
        # 5. Save results
        self.save_sentiment_data(daily_sentiment, output_file)
        
        # 6. Visualize trends
        self.visualize_sentiment_trends(daily_sentiment)
        
        print(f"\nAnalysis complete! Data saved to {output_file}")
        
        return daily_sentiment

# Example usage
if __name__ == "__main__":
    analyzer = WikipediaSentimentAnalyzer()
    
    # Run full analysis
    sentiment_data = analyzer.run_full_analysis(
        page_title='Bitcoin',
        days_back=365,
        output_file='wikipedia_edits.csv'
    )
    
    # Display recent sentiment summary
    print("\nRecent Sentiment Summary (last 7 days):")
    print(sentiment_data.tail(7)[['edit_count', 'positive_sentiment_mean', 
                                  'engagement_score', 'controversy_score']])
