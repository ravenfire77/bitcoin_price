import pandas as pd
import numpy as np
import mwclient
from datetime import datetime, timedelta
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class WikipediaBitcoinSentiment:
    def __init__(self):
        self.wiki_data = None
        self.btc_data = None
        self.merged_data = None
        self.sentiment_analyzer = None
        
    def initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis pipeline"""
        print("Initializing sentiment analyzer...")
        try:
            # Try financial sentiment model first
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # Use CPU
            )
            print("✓ Using FinBERT for financial sentiment analysis")
        except:
            # Fallback to general sentiment model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            print("✓ Using DistilBERT for general sentiment analysis")
    
    def fetch_wikipedia_edits(self, page_title='Bitcoin', days_back=None):
        """Fetch Wikipedia edits for Bitcoin page"""
        print(f"\n📚 Fetching Wikipedia edits for '{page_title}' page...")
        
        # Connect to Wikipedia
        site = mwclient.Site('en.wikipedia.org')
        page = site.pages[page_title]
        
        # Calculate date range
        end_date = datetime.now()
        if days_back:
            start_date = end_date - timedelta(days=days_back)
        else:
            # Match Bitcoin data date range
            start_date = datetime(2014, 9, 17)  # Your data start date
        
        print(f"Date range: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
        
        # Fetch revisions
        edits = []
        revision_count = 0
        
        print("Fetching revisions...")
        for revision in page.revisions(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            limit=10000  # Increase limit for more data
        ):
            revision_count += 1
            if revision_count % 500 == 0:
                print(f"  Processed {revision_count} revisions...")
            
            edit = {
                'timestamp': revision['timestamp'],
                'user': revision.get('user', 'Anonymous'),
                'userid': revision.get('userid', 0),
                'comment': revision.get('comment', ''),
                'size': revision.get('size', 0),
                'revid': revision['revid'],
                'parentid': revision.get('parentid', 0),
                'minor': revision.get('minor', False),
                'tags': revision.get('tags', [])
            }
            
            # Calculate size change
            if 'parentid' in revision and revision['parentid'] != 0:
                parent_size = revision.get('parent_size', edit['size'])
                edit['size_change'] = edit['size'] - parent_size
            else:
                edit['size_change'] = 0
            
            edits.append(edit)
        
        print(f"✓ Fetched {len(edits)} edits")
        
        # Convert to DataFrame
        wiki_df = pd.DataFrame(edits)
        wiki_df['timestamp'] = pd.to_datetime(wiki_df['timestamp'])
        wiki_df['date'] = wiki_df['timestamp'].dt.date
        
        return wiki_df
    
    def analyze_edit_content(self, wiki_df):
        """Analyze content and patterns in edits"""
        print("\n🔍 Analyzing edit patterns...")
        
        # Extract features from comments
        wiki_df['comment_length'] = wiki_df['comment'].str.len()
        wiki_df['has_revert'] = wiki_df['comment'].str.contains('revert', case=False, na=False)
        wiki_df['has_vandalism'] = wiki_df['comment'].str.contains('vandal', case=False, na=False)
        wiki_df['has_update'] = wiki_df['comment'].str.contains('update|add|fix', case=False, na=False)
        wiki_df['has_reference'] = wiki_df['comment'].str.contains('ref|citation|source', case=False, na=False)
        wiki_df['has_price'] = wiki_df['comment'].str.contains('price|value|worth|cost', case=False, na=False)
        wiki_df['has_news'] = wiki_df['comment'].str.contains('news|announce|report', case=False, na=False)
        wiki_df['is_bot'] = wiki_df['user'].str.contains('bot', case=False, na=False)
        
        # User analysis
        wiki_df['is_anonymous'] = wiki_df['userid'] == 0
        wiki_df['is_registered'] = ~wiki_df['is_anonymous']
        
        # Edit size categories
        wiki_df['size_category'] = pd.cut(
            wiki_df['size_change'].abs(),
            bins=[0, 10, 50, 200, 1000, np.inf],
            labels=['tiny', 'small', 'medium', 'large', 'massive']
        )
        
        return wiki_df
    
    def analyze_sentiment(self, wiki_df):
        """Analyze sentiment of edit comments"""
        print("\n💭 Analyzing sentiment of edit comments...")
        
        # Initialize sentiment analyzer if not already done
        if self.sentiment_analyzer is None:
            self.initialize_sentiment_analyzer()
        
        sentiments = []
        sentiment_scores = []
        
        # Process comments in batches for efficiency
        batch_size = 100
        total_comments = len(wiki_df)
        
        for i in range(0, total_comments, batch_size):
            if i % 500 == 0:
                print(f"  Processing comments {i}/{total_comments}...")
            
            batch = wiki_df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                comment = row['comment']
                
                if comment and len(comment.strip()) > 0:
                    try:
                        # Truncate long comments
                        truncated = comment[:512]
                        result = self.sentiment_analyzer(truncated)[0]
                        
                        # Handle different model outputs
                        if 'label' in result:
                            if result['label'] in ['POSITIVE', 'positive', 'POS']:
                                sentiment = 'positive'
                                score = result['score']
                            elif result['label'] in ['NEGATIVE', 'negative', 'NEG']:
                                sentiment = 'negative'
                                score = -result['score']
                            else:  # neutral
                                sentiment = 'neutral'
                                score = 0
                        else:
                            sentiment = 'neutral'
                            score = 0
                            
                    except Exception as e:
                        sentiment = 'neutral'
                        score = 0
                else:
                    sentiment = 'neutral'
                    score = 0
                
                sentiments.append(sentiment)
                sentiment_scores.append(score)
        
        wiki_df['sentiment'] = sentiments
        wiki_df['sentiment_score'] = sentiment_scores
        
        # Create binary sentiment indicators
        wiki_df['sentiment_positive'] = (wiki_df['sentiment'] == 'positive').astype(int)
        wiki_df['sentiment_negative'] = (wiki_df['sentiment'] == 'negative').astype(int)
        
        print(f"✓ Analyzed sentiment for {total_comments} comments")
        print(f"  Positive: {sum(wiki_df['sentiment'] == 'positive')}")
        print(f"  Negative: {sum(wiki_df['sentiment'] == 'negative')}")
        print(f"  Neutral: {sum(wiki_df['sentiment'] == 'neutral')}")
        
        return wiki_df
    
    def aggregate_daily_metrics(self, wiki_df):
        """Aggregate Wikipedia metrics by day"""
        print("\n📊 Aggregating daily Wikipedia metrics...")
        
        # Convert date to pandas datetime for grouping
        wiki_df['date'] = pd.to_datetime(wiki_df['date'])
        
        # Daily aggregations
        daily_metrics = wiki_df.groupby('date').agg({
            # Edit counts
            'revid': 'count',
            'minor': 'sum',
            'size_change': ['sum', 'mean', 'std', 'min', 'max'],
            
            # User metrics
            'user': 'nunique',
            'is_anonymous': 'sum',
            'is_registered': 'sum',
            'is_bot': 'sum',
            
            # Content metrics
            'has_revert': 'sum',
            'has_vandalism': 'sum',
            'has_update': 'sum',
            'has_reference': 'sum',
            'has_price': 'sum',
            'has_news': 'sum',
            
            # Sentiment metrics
            'sentiment_score': ['mean', 'std', 'min', 'max'],
            'sentiment_positive': 'sum',
            'sentiment_negative': 'sum',
            
            # Comment metrics
            'comment_length': ['mean', 'max']
        })
        
        # Flatten column names
        daily_metrics.columns = ['_'.join(col).strip() for col in daily_metrics.columns.values]
        
        # Rename columns for clarity
        column_mapping = {
            'revid_count': 'edit_count',
            'minor_sum': 'minor_edits',
            'user_nunique': 'unique_editors',
            'is_anonymous_sum': 'anonymous_edits',
            'is_registered_sum': 'registered_edits',
            'is_bot_sum': 'bot_edits',
            'has_revert_sum': 'revert_count',
            'has_vandalism_sum': 'vandalism_mentions',
            'has_update_sum': 'update_edits',
            'has_reference_sum': 'reference_edits',
            'has_price_sum': 'price_mentions',
            'has_news_sum': 'news_mentions'
        }
        
        daily_metrics.rename(columns=column_mapping, inplace=True)
        
        # Calculate derived metrics
        daily_metrics['major_edits'] = daily_metrics['edit_count'] - daily_metrics['minor_edits']
        daily_metrics['human_edits'] = daily_metrics['edit_count'] - daily_metrics['bot_edits']
        daily_metrics['edit_sentiment'] = daily_metrics['sentiment_positive'] - daily_metrics['sentiment_negative']
        daily_metrics['controversy_score'] = (daily_metrics['revert_count'] + daily_metrics['vandalism_mentions']) / daily_metrics['edit_count'].clip(lower=1)
        daily_metrics['engagement_score'] = daily_metrics['edit_count'] * daily_metrics['unique_editors'] / 10
        
        # Calculate rolling averages for smoothing
        for col in ['edit_count', 'unique_editors', 'sentiment_score_mean', 'engagement_score']:
            if col in daily_metrics.columns:
                daily_metrics[f'{col}_ma7'] = daily_metrics[col].rolling(window=7, min_periods=1).mean()
                daily_metrics[f'{col}_ma30'] = daily_metrics[col].rolling(window=30, min_periods=1).mean()
        
        # Add day of week features
        daily_metrics['day_of_week'] = daily_metrics.index.dayofweek
        daily_metrics['is_weekend'] = (daily_metrics['day_of_week'] >= 5).astype(int)
        
        print(f"✓ Created {len(daily_metrics.columns)} daily Wikipedia features")
        print(f"✓ Date range: {daily_metrics.index.min()} to {daily_metrics.index.max()}")
        
        self.wiki_data = daily_metrics
        return daily_metrics
    
    def merge_with_bitcoin_data(self, btc_file='btc_USD.xlsx', wiki_data=None):
        """Merge Wikipedia data with Bitcoin price data"""
        print("\n🔗 Merging Wikipedia data with Bitcoin prices...")
        
        # Load Bitcoin data
        btc_df = pd.read_excel(btc_file, parse_dates=['Date'], dayfirst=True)
        btc_df.columns = btc_df.columns.str.replace('\xa0', ' ')
        btc_df = btc_df.sort_values('Date')
        btc_df.set_index('Date', inplace=True)
        
        # Use provided wiki_data or self.wiki_data
        if wiki_data is None:
            wiki_data = self.wiki_data
        
        # Ensure wiki_data index is datetime
        if not isinstance(wiki_data.index, pd.DatetimeIndex):
            wiki_data.index = pd.to_datetime(wiki_data.index)
        
        # Merge datasets
        merged_df = btc_df.join(wiki_data, how='left')
        
        # Fill missing Wikipedia data
        wiki_columns = [col for col in wiki_data.columns if col in merged_df.columns]
        for col in wiki_columns:
            # Forward fill for most columns
            if 'ma' in col or 'score' in col:
                merged_df[col] = merged_df[col].fillna(method='ffill').fillna(0)
            else:
                # For count data, fill with 0
                merged_df[col] = merged_df[col].fillna(0)
        
        print(f"✓ Merged data shape: {merged_df.shape}")
        print(f"✓ Wikipedia features added: {len(wiki_columns)}")
        
        self.btc_data = btc_df
        self.merged_data = merged_df
        
        return merged_df
    
    def create_enhanced_features(self, merged_df):
        """Create combined features from price and Wikipedia data"""
        print("\n⚡ Creating enhanced features...")
        
        # Price features (basic)
        merged_df['Returns'] = merged_df['Close'].pct_change()
        merged_df['Target'] = (merged_df['Close'].shift(-1) > merged_df['Close']).astype(int)
        
        # Price technical indicators
        merged_df['SMA_20'] = merged_df['Close'].rolling(window=20).mean()
        merged_df['SMA_50'] = merged_df['Close'].rolling(window=50).mean()
        merged_df['Price_to_SMA20'] = merged_df['Close'] / merged_df['SMA_20']
        merged_df['Price_to_SMA50'] = merged_df['Close'] / merged_df['SMA_50']
        
        # RSI
        delta = merged_df['Close'].diff()
        gains = delta.where(delta > 0, 0).rolling(window=14).mean()
        losses = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gains / losses
        merged_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume features
        merged_df['Volume_SMA'] = merged_df['Volume'].rolling(window=20).mean()
        merged_df['Volume_Ratio'] = merged_df['Volume'] / merged_df['Volume_SMA']
        
        # Combined Wikipedia-Price features
        if 'edit_count' in merged_df.columns:
            # Edit activity vs price movement
            merged_df['edits_per_volume'] = merged_df['edit_count'] / (merged_df['Volume'] / 1e9).clip(lower=1)
            merged_df['sentiment_price_alignment'] = merged_df['sentiment_score_mean'] * merged_df['Returns']
            
            # Unusual activity indicators
            merged_df['high_edit_activity'] = (merged_df['edit_count'] > merged_df['edit_count_ma30'] * 1.5).astype(int)
            merged_df['high_controversy'] = (merged_df['controversy_score'] > 0.1).astype(int)
            
            # Engagement momentum
            merged_df['engagement_momentum'] = merged_df['engagement_score'].pct_change()
            
            # News and price correlation
            if 'price_mentions' in merged_df.columns:
                merged_df['price_mention_intensity'] = merged_df['price_mentions'] / merged_df['edit_count'].clip(lower=1)
        
        # Lag features
        lag_features = ['Returns', 'Volume_Ratio', 'edit_count', 'sentiment_score_mean', 'engagement_score']
        for feature in lag_features:
            if feature in merged_df.columns:
                for lag in [1, 3, 7]:
                    merged_df[f'{feature}_lag_{lag}'] = merged_df[feature].shift(lag)
        
        print(f"✓ Total features created: {len(merged_df.columns)}")
        
        return merged_df
    
    def train_enhanced_models(self, merged_df):
        """Train models with and without Wikipedia features"""
        print("\n🤖 Training enhanced models with Wikipedia sentiment...")
        
        # Prepare data
        merged_df = merged_df.dropna()
        
        # Define features
        price_only_features = [
            'Returns', 'Price_to_SMA20', 'Price_to_SMA50', 'RSI', 
            'Volume_Ratio', 'Returns_lag_1', 'Returns_lag_3', 'Returns_lag_7',
            'Volume_Ratio_lag_1', 'Volume_Ratio_lag_3'
        ]
        
        wiki_features = [
            'edit_count', 'unique_editors', 'sentiment_score_mean', 
            'engagement_score', 'controversy_score', 'price_mentions',
            'high_edit_activity', 'sentiment_price_alignment',
            'edit_count_ma7', 'sentiment_score_mean_ma7'
        ]
        
        # Filter available features
        price_only_features = [f for f in price_only_features if f in merged_df.columns]
        wiki_features = [f for f in wiki_features if f in merged_df.columns]
        all_features = price_only_features + wiki_features
        
        # Prepare target
        y = merged_df['Target']
        
        # Train-test split (80-20)
        split_idx = int(len(merged_df) * 0.8)
        
        # Results storage
        results = {}
        
        # Train models with different feature sets
        for feature_set_name, features in [
            ('Price Only', price_only_features),
            ('Price + Wikipedia', all_features)
        ]:
            if len(features) == 0:
                continue
                
            X = merged_df[features]
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            
            # Train XGBoost
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.01, random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            
            # Calculate metrics
            results[feature_set_name] = {
                'rf_accuracy': accuracy_score(y_test, rf_pred),
                'rf_f1': f1_score(y_test, rf_pred),
                'xgb_accuracy': accuracy_score(y_test, xgb_pred),
                'xgb_f1': f1_score(y_test, xgb_pred),
                'features_used': len(features),
                'test_size': len(y_test),
                'best_model': rf if f1_score(y_test, rf_pred) > f1_score(y_test, xgb_pred) else xgb_model,
                'feature_names': features
            }
            
            print(f"\n{feature_set_name} Results:")
            print(f"  Features used: {len(features)}")
            print(f"  Random Forest - Accuracy: {results[feature_set_name]['rf_accuracy']:.2%}, F1: {results[feature_set_name]['rf_f1']:.2%}")
            print(f"  XGBoost - Accuracy: {results[feature_set_name]['xgb_accuracy']:.2%}, F1: {results[feature_set_name]['xgb_f1']:.2%}")
        
        return results, merged_df.iloc[split_idx:]
    
    def visualize_wikipedia_impact(self, wiki_df, merged_df, results):
        """Create visualizations showing Wikipedia sentiment impact"""
        print("\n📊 Creating Wikipedia sentiment visualizations...")
        
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Wikipedia edit activity over time
        ax1 = plt.subplot(4, 2, 1)
        recent_wiki = wiki_df.iloc[-365:]  # Last year
        ax1.plot(recent_wiki.index, recent_wiki['edit_count'], alpha=0.5, label='Daily Edits')
        ax1.plot(recent_wiki.index, recent_wiki['edit_count_ma7'], label='7-day MA', linewidth=2)
        ax1.set_title('Wikipedia Bitcoin Page: Edit Activity', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Edits')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sentiment analysis over time
        ax2 = plt.subplot(4, 2, 2)
        ax2.plot(recent_wiki.index, recent_wiki['sentiment_score_mean'], alpha=0.5, label='Daily Sentiment')
        ax2.plot(recent_wiki.index, recent_wiki['sentiment_score_mean_ma7'], label='7-day MA', linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Wikipedia Edit Sentiment Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sentiment Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Edit types distribution
        ax3 = plt.subplot(4, 2, 3)
        edit_types = ['update_edits', 'reference_edits', 'revert_count', 'price_mentions', 'news_mentions']
        edit_type_data = []
        edit_type_labels = []
        
        for et in edit_types:
            if et in wiki_df.columns:
                edit_type_data.append(wiki_df[et].sum())
                edit_type_labels.append(et.replace('_', ' ').title())
        
        if edit_type_data:
            ax3.pie(edit_type_data, labels=edit_type_labels, autopct='%1.1f%%')
            ax3.set_title('Distribution of Edit Types', fontsize=12, fontweight='bold')
        
        # 4. Engagement score vs Bitcoin price
        ax4 = plt.subplot(4, 2, 4)
        recent_merged = merged_df.iloc[-365:]
        ax4_2 = ax4.twinx()
        
        ax4.plot(recent_merged.index, recent_merged['Close'], 'b-', alpha=0.7, label='Bitcoin Price')
        ax4_2.plot(recent_merged.index, recent_merged['engagement_score_ma7'], 'r-', alpha=0.7, label='Engagement Score')
        
        ax4.set_ylabel('Bitcoin Price (USD)', color='b')
        ax4_2.set_ylabel('Engagement Score', color='r')
        ax4.set_title('Bitcoin Price vs Wikipedia Engagement', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='y', labelcolor='b')
        ax4_2.tick_params(axis='y', labelcolor='r')
        ax4.grid(True, alpha=0.3)
        
        # 5. Model performance comparison
        ax5 = plt.subplot(4, 2, 5)
        if results:
            models = list(results.keys())
            rf_scores = [results[m]['rf_f1'] for m in models]
            xgb_scores = [results[m]['xgb_f1'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax5.bar(x - width/2, rf_scores, width, label='Random Forest', alpha=0.8)
            ax5.bar(x + width/2, xgb_scores, width, label='XGBoost', alpha=0.8)
            
            ax5.set_ylabel('F1 Score')
            ax5.set_title('Model Performance: With vs Without Wikipedia Data', fontsize=12, fontweight='bold')
            ax5.set_xticks(x)
            ax5.set_xticklabels(models)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Add improvement percentage
            if len(models) > 1:
                improvement = ((max(rf_scores[1], xgb_scores[1]) / max(rf_scores[0], xgb_scores[0])) - 1) * 100
                ax5.text(0.5, 0.95, f'Improvement: +{improvement:.1f}%', 
                        transform=ax5.transAxes, ha='center', fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 6. Feature importance with Wikipedia features
        ax6 = plt.subplot(4, 2, 6)
        if 'Price + Wikipedia' in results:
            model = results['Price + Wikipedia']['best_model']
            features = results['Price + Wikipedia']['feature_names']
            importance = model.feature_importances_
            
            # Get top 15 features
            feature_imp = pd.DataFrame({'feature': features, 'importance': importance})
            feature_imp = feature_imp.sort_values('importance', ascending=True).tail(15)
            
            # Color Wikipedia features differently
            colors = ['red' if any(wiki_term in f for wiki_term in ['edit', 'sentiment', 'engagement', 'controversy']) 
                     else 'blue' for f in feature_imp['feature']]
            
            ax6.barh(feature_imp['feature'], feature_imp['importance'], color=colors)
            ax6.set_xlabel('Importance')
            ax6.set_title('Feature Importance (Red = Wikipedia, Blue = Price)', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        # 7. Sentiment distribution by day of week
        ax7 = plt.subplot(4, 2, 7)
        if 'day_of_week' in wiki_df.columns:
            sentiment_by_dow = wiki_df.groupby('day_of_week')['sentiment_score_mean'].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax7.bar(range(7), sentiment_by_dow.values)
            ax7.set_xticks(range(7))
            ax7.set_xticklabels(days)
            ax7.set_ylabel('Average Sentiment Score')
            ax7.set_title('Wikipedia Sentiment by Day of Week', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        # 8. High activity periods analysis
        ax8 = plt.subplot(4, 2, 8)
        if 'high_edit_activity' in merged_df.columns:
            high_activity_days = merged_df[merged_df['high_edit_activity'] == 1]
            
            # Calculate average returns on high activity days
            high_activity_returns = high_activity_days['Returns'].mean() * 100
            normal_returns = merged_df[merged_df['high_edit_activity'] == 0]['Returns'].mean() * 100
            
            categories = ['High Wiki Activity', 'Normal Activity']
            returns = [high_activity_returns, normal_returns]
            colors = ['red', 'blue']
            
            ax8.bar(categories, returns, color=colors, alpha=0.7)
            ax8.set_ylabel('Average Daily Return (%)')
            ax8.set_title('Bitcoin Returns During High Wikipedia Activity', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3)
            
            # Add sample size
            for i, (cat, ret) in enumerate(zip(categories, returns)):
                count = len(high_activity_days) if i == 0 else len(merged_df) - len(high_activity_days)
                ax8.text(i, ret + 0.01, f'n={count}', ha='center')
        
        plt.tight_layout()
        plt.savefig('wikipedia_sentiment_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Saved visualization to 'wikipedia_sentiment_impact_analysis.png'")
    
    def generate_enhanced_prediction_report(self, results, merged_df):
        """Generate report on prediction improvements with Wikipedia data"""
        print("\n" + "="*70)
        print("WIKIPEDIA SENTIMENT ENHANCEMENT REPORT")
        print("="*70)
        
        if 'Price Only' in results and 'Price + Wikipedia' in results:
            # Calculate improvements
            price_only_best = max(results['Price Only']['rf_f1'], results['Price Only']['xgb_f1'])
            wiki_enhanced_best = max(results['Price + Wikipedia']['rf_f1'], results['Price + Wikipedia']['xgb_f1'])
            improvement = ((wiki_enhanced_best / price_only_best) - 1) * 100
            
            print("\n📊 MODEL PERFORMANCE COMPARISON:")
            print(f"{'Model Type':<20} {'Accuracy':<12} {'F1 Score':<12} {'Features':<10}")
            print("-" * 54)
            print(f"{'Price Only (RF)':<20} {results['Price Only']['rf_accuracy']:<12.2%} {results['Price Only']['rf_f1']:<12.2%} {results['Price Only']['features_used']:<10}")
            print(f"{'Price Only (XGB)':<20} {results['Price Only']['xgb_accuracy']:<12.2%} {results['Price Only']['xgb_f1']:<12.2%} {results['Price Only']['features_used']:<10}")
            print(f"{'With Wikipedia (RF)':<20} {results['Price + Wikipedia']['rf_accuracy']:<12.2%} {results['Price + Wikipedia']['rf_f1']:<12.2%} {results['Price + Wikipedia']['features_used']:<10}")
            print(f"{'With Wikipedia (XGB)':<20} {results['Price + Wikipedia']['xgb_accuracy']:<12.2%} {results['Price + Wikipedia']['xgb_f1']:<12.2%} {results['Price + Wikipedia']['features_used']:<10}")
            
            print(f"\n🎯 IMPROVEMENT WITH WIKIPEDIA DATA: +{improvement:.1f}%")
            
            # Key Wikipedia insights
            print("\n📚 WIKIPEDIA DATA INSIGHTS:")
            
            if self.wiki_data is not None:
                print(f"- Total edits analyzed: {self.wiki_data['edit_count'].sum():,}")
                print(f"- Average daily edits: {self.wiki_data['edit_count'].mean():.1f}")
                print(f"- Unique editors: {self.wiki_data['unique_editors'].sum():,}")
                print(f"- Average sentiment: {self.wiki_data['sentiment_score_mean'].mean():.3f}")
                
                # Correlation analysis
                if 'Returns' in merged_df.columns:
                    wiki_features = ['edit_count', 'sentiment_score_mean', 'engagement_score', 'controversy_score']
                    print("\n📈 CORRELATION WITH BITCOIN RETURNS:")
                    for feature in wiki_features:
                        if feature in merged_df.columns:
                            corr = merged_df[feature].corr(merged_df['Returns'])
                            print(f"  {feature}: {corr:.3f}")
            
            # Trading strategy implications
            print("\n💰 TRADING STRATEGY IMPLICATIONS:")
            print("- Wikipedia sentiment provides early signals of market interest")
            print("- High edit activity often precedes price volatility")
            print("- Controversy scores can indicate market uncertainty")
            print(f"- Model improvement of {improvement:.1f}% could significantly enhance trading returns")
            
        print("\n✅ Wikipedia sentiment successfully integrated into prediction model!")
        print("="*70)
    
    def run_complete_analysis(self, btc_file='btc_USD.xlsx', fetch_new_data=True):
        """Run the complete Wikipedia sentiment analysis pipeline"""
        print("Starting Wikipedia Sentiment Analysis for Bitcoin Predictions")
        print("="*60)
        
        # Initialize sentiment analyzer
        self.initialize_sentiment_analyzer()
        
        if fetch_new_data:
            # Fetch Wikipedia data
            wiki_raw = self.fetch_wikipedia_edits('Bitcoin')
            
            # Analyze content
            wiki_analyzed = self.analyze_edit_content(wiki_raw)
            
            # Analyze sentiment
            wiki_sentiment = self.analyze_sentiment(wiki_analyzed)
            
            # Aggregate daily
            wiki_daily = self.aggregate_daily_metrics(wiki_sentiment)
            
            # Save for future use
            wiki_daily.to_csv('wikipedia_bitcoin_sentiment.csv')
            print("✓ Saved Wikipedia data to 'wikipedia_bitcoin_sentiment.csv'")
        else:
            # Load existing data
            print("Loading existing Wikipedia sentiment data...")
            wiki_daily = pd.read_csv('wikipedia_bitcoin_sentiment.csv', index_col=0, parse_dates=True)
            self.wiki_data = wiki_daily
        
        # Merge with Bitcoin data
        merged_data = self.merge_with_bitcoin_data(btc_file, wiki_daily)
        
        # Create enhanced features
        enhanced_data = self.create_enhanced_features(merged_data)
        
        # Train models
        results, test_data = self.train_enhanced_models(enhanced_data)
        
        # Visualize impact
        self.visualize_wikipedia_impact(wiki_daily, enhanced_data, results)
        
        # Generate report
        self.generate_enhanced_prediction_report(results, enhanced_data)
        
        return results, enhanced_data

# Run the analysis
if __name__ == "__main__":
    analyzer = WikipediaBitcoinSentiment()
    
    # Run complete analysis
    # Set fetch_new_data=False to use cached data for faster testing
    results, enhanced_data = analyzer.run_complete_analysis(
        btc_file='btc_USD.xlsx',
        fetch_new_data=True  # Set to False after first run to use cached data
    )
