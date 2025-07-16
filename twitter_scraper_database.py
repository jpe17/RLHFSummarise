import time
import random
import json
import re
import sqlite3
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

class TwitterDatabaseScraper:
    def __init__(self, headless=True, db_path="tweets_database.db"):
        """
        Initialize the Twitter scraper with database storage.
        
        Args:
            headless (bool): Run browser in headless mode
            db_path (str): Path to SQLite database file
        """
        self.driver = None
        self.headless = headless
        self.db_path = db_path
        self.setup_driver()
        self.setup_database()
    
    def setup_driver(self):
        """Set up the Chrome WebDriver with appropriate options."""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Add options to avoid detection
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Set user agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            print(f"❌ Error setting up Chrome driver: {e}")
            print("💡 Make sure you have Chrome installed and chromedriver in your PATH")
            self.driver = None
    
    def setup_database(self):
        """Set up the SQLite database with tweets table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tweets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tweets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT,
                    likes INTEGER DEFAULT 0,
                    retweets INTEGER DEFAULT 0,
                    replies INTEGER DEFAULT 0,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(username, content)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON tweets(username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_likes ON tweets(likes)')
            
            conn.commit()
            conn.close()
            print(f"✅ Database setup complete: {self.db_path}")
            
        except Exception as e:
            print(f"❌ Error setting up database: {e}")
    
    def store_tweets_in_database(self, username, tweets):
        """Store tweets in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stored_count = 0
            for tweet in tweets:
                try:
                    # Convert metrics to integers
                    likes = self._parse_metric_to_int(tweet.get('likes', '0'))
                    retweets = self._parse_metric_to_int(tweet.get('retweets', '0'))
                    replies = self._parse_metric_to_int(tweet.get('replies', '0'))
                    
                    cursor.execute('''
                        INSERT OR IGNORE INTO tweets 
                        (username, content, timestamp, likes, retweets, replies)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        username,
                        tweet['content'],
                        tweet.get('timestamp', 'Recent'),
                        likes,
                        retweets,
                        replies
                    ))
                    
                    if cursor.rowcount > 0:
                        stored_count += 1
                        
                except Exception as e:
                    print(f"   ⚠️ Error storing tweet: {e}")
                    continue
            
            conn.commit()
            conn.close()
            print(f"✅ Stored {stored_count} new tweets for @{username}")
            return stored_count
            
        except Exception as e:
            print(f"❌ Error storing tweets in database: {e}")
            return 0
    
    def get_stored_tweets_count(self, username):
        """Get the number of tweets stored for a username."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM tweets WHERE username = ?', (username,))
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception as e:
            print(f"❌ Error getting tweet count: {e}")
            return 0
    
    def get_top_liked_tweets_from_db(self, username, count=10):
        """Get top liked tweets from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT content, timestamp, likes, retweets, replies
                FROM tweets 
                WHERE username = ?
                ORDER BY likes DESC
                LIMIT ?
            ''', (username, count))
            
            results = cursor.fetchall()
            conn.close()
            
            tweets = []
            for row in results:
                tweets.append({
                    'content': row[0],
                    'timestamp': row[1],
                    'likes': str(row[2]),
                    'retweets': str(row[3]),
                    'replies': str(row[4])
                })
            
            return tweets
            
        except Exception as e:
            print(f"❌ Error getting tweets from database: {e}")
            return []
    
    def scrape_and_store_tweets(self, username, target_count=100):
        """
        Scrape tweets from a user and store them in the database.
        
        Args:
            username (str): Twitter username (without @)
            target_count (int): Number of tweets to scrape (default: 100)
            
        Returns:
            int: Number of new tweets stored
        """
        if not self.driver:
            print("❌ WebDriver not available")
            return 0
        
        # Check how many tweets we already have
        existing_count = self.get_stored_tweets_count(username)
        print(f"📊 @{username}: {existing_count} tweets already stored")
        
        if existing_count >= target_count:
            print(f"✅ Already have {existing_count} tweets for @{username}, skipping...")
            return 0
        
        needed_count = target_count - existing_count
        print(f"🎯 Need to scrape {needed_count} more tweets for @{username}")
        
        try:
            # Navigate to user's timeline
            url = f"https://twitter.com/{username}"
            print(f"🔍 Navigating to @{username}'s timeline...")
            
            self.driver.get(url)
            time.sleep(3)
            
            # Wait for tweets to load
            print("⏳ Waiting for tweets to load...")
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda driver: len(driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')) > 0
                )
                print("✅ Tweets found on page")
            except TimeoutException:
                print("❌ No tweets found")
                return 0
            
            # Extract tweets with more aggressive scrolling
            tweets = self._extract_tweets_for_storage(needed_count, max_scrolls=15)
            
            if not tweets:
                print("⚠️ No tweets extracted")
                return 0
            
            # Store in database
            stored_count = self.store_tweets_in_database(username, tweets)
            
            print(f"✅ Successfully stored {stored_count} new tweets for @{username}")
            return stored_count
            
        except Exception as e:
            print(f"❌ Error scraping tweets for @{username}: {e}")
            return 0
    
    def _extract_tweets_for_storage(self, target_count, max_scrolls=15):
        """Extract tweets with aggressive scrolling for storage."""
        tweets = []
        processed_elements = set()
        
        try:
            scroll_attempts = 0
            
            while len(tweets) < target_count and scroll_attempts < max_scrolls:
                # Get all tweet elements on the current page
                tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                
                if not tweet_elements:
                    tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, 'article')
                
                print(f"📝 Found {len(tweet_elements)} tweet elements on page (scroll {scroll_attempts + 1})")
                
                # Process each tweet element
                for element in tweet_elements:
                    if len(tweets) >= target_count:
                        break
                    
                    # Skip if we've already processed this element
                    element_id = element.id
                    if element_id in processed_elements:
                        continue
                    
                    processed_elements.add(element_id)
                    
                    try:
                        tweet_data = self._extract_single_tweet_with_metrics(element)
                        if tweet_data and len(tweet_data['content'].strip()) > 5:
                            if not self._is_obvious_retweet(tweet_data['content']):
                                tweets.append(tweet_data)
                                print(f"   ✅ Extracted tweet {len(tweets)}: {tweet_data['content'][:60]}... (❤️ {tweet_data['likes']})")
                    except Exception as e:
                        continue
                
                # Scroll for more tweets if needed
                if len(tweets) < target_count:
                    print(f"📜 Scrolling for more tweets ({len(tweets)}/{target_count})...")
                    self.driver.execute_script("window.scrollBy(0, 1200);")
                    time.sleep(2.5)
                    scroll_attempts += 1
                else:
                    break
            
        except Exception as e:
            print(f"⚠️ Error extracting tweets: {e}")
        
        return tweets
    
    def _extract_single_tweet_with_metrics(self, element):
        """Extract tweet data with improved metric extraction for sorting."""
        try:
            # Extract tweet text (same as before)
            content = ""
            text_selectors = [
                '[data-testid="tweetText"]',
                'div[data-testid="tweetText"]',
                'span[data-testid="tweetText"]',
                'div[lang]',
                'div[dir="auto"]'
            ]
            
            for selector in text_selectors:
                try:
                    text_elements = element.find_elements(By.CSS_SELECTOR, selector)
                    for text_element in text_elements:
                        text = text_element.text.strip()
                        if text and len(text) > 5:
                            content = text
                            break
                    if content:
                        break
                except:
                    continue
            
            # Fallback text extraction
            if not content:
                full_text = element.text.strip()
                lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                meaningful_lines = []
                
                for line in lines:
                    if (len(line) > 5 and 
                        not line.startswith('@') and
                        not line.startswith('http') and
                        not line.isdigit() and
                        not any(word in line.lower() for word in ['follow', 'following', 'followers', 'retweet', 'like', 'reply', 'show this thread', 'view translation'])):
                        meaningful_lines.append(line)
                
                if meaningful_lines:
                    content = max(meaningful_lines, key=len)
            
            if not content or len(content.strip()) < 5:
                return None
            
            # Extract timestamp
            timestamp = "Recent"
            try:
                time_elements = element.find_elements(By.CSS_SELECTOR, 'time')
                if time_elements:
                    dt = time_elements[0].get_attribute('datetime')
                    if dt:
                        timestamp = dt
                    else:
                        timestamp = time_elements[0].text.strip()
            except:
                pass
            
            # Extract engagement metrics with improved accuracy
            likes = self._extract_metric_improved(element, 'like')
            retweets = self._extract_metric_improved(element, 'retweet')
            replies = self._extract_metric_improved(element, 'reply')
            
            return {
                'content': content,
                'timestamp': timestamp,
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'likes_count': self._parse_metric_to_int(likes)  # For sorting
            }
            
        except Exception as e:
            print(f"   ⚠️ Error in _extract_single_tweet_with_metrics: {e}")
            return None
    
    def _extract_metric_improved(self, element, metric_type):
        """Extract engagement metric with improved accuracy."""
        try:
            # Try specific selectors for metrics
            metric_selectors = {
                'like': [
                    '[data-testid="like"]',
                    '[data-testid="unlike"]',
                    'div[aria-label*="like"]',
                    'div[aria-label*="Like"]'
                ],
                'retweet': [
                    '[data-testid="retweet"]',
                    '[data-testid="unretweet"]',
                    'div[aria-label*="retweet"]',
                    'div[aria-label*="Retweet"]'
                ],
                'reply': [
                    '[data-testid="reply"]',
                    'div[aria-label*="reply"]',
                    'div[aria-label*="Reply"]'
                ]
            }
            
            selectors = metric_selectors.get(metric_type, [])
            
            for selector in selectors:
                try:
                    metric_elements = element.find_elements(By.CSS_SELECTOR, selector)
                    for metric_element in metric_elements:
                        # Look for the count in the parent or sibling elements
                        parent = metric_element.find_element(By.XPATH, "./..")
                        text = parent.text.strip()
                        
                        # Extract numbers from the text
                        numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                        if numbers:
                            return numbers[0]
                except:
                    continue
            
            # Fallback: look in the entire element text
            text = element.text.lower()
            if metric_type == 'like':
                if 'like' in text:
                    numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                    return numbers[0] if numbers else "0"
            elif metric_type == 'retweet':
                if 'retweet' in text or 'repost' in text:
                    numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                    return numbers[0] if numbers else "0"
            elif metric_type == 'reply':
                if 'reply' in text:
                    numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                    return numbers[0] if numbers else "0"
        except:
            pass
        return "0"
    
    def _parse_metric_to_int(self, metric_str):
        """Convert metric string to integer for sorting."""
        try:
            # Remove commas and convert to int
            return int(metric_str.replace(',', ''))
        except:
            return 0
    
    def _is_obvious_retweet(self, content):
        """Check if content is obviously a retweet (more lenient)."""
        content_lower = content.lower()
        
        # Only flag obvious retweets
        obvious_indicators = [
            'rt @',  # Old-style retweets
            'retweeted by',  # New-style retweets
            'reposted by',  # Reposts
        ]
        
        for indicator in obvious_indicators:
            if indicator in content_lower:
                return True
        
        return False
    
    def get_database_stats(self):
        """Get statistics about the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total tweets
            cursor.execute('SELECT COUNT(*) FROM tweets')
            total_tweets = cursor.fetchone()[0]
            
            # Get unique usernames
            cursor.execute('SELECT COUNT(DISTINCT username) FROM tweets')
            unique_users = cursor.fetchone()[0]
            
            # Get tweets per user
            cursor.execute('''
                SELECT username, COUNT(*) as tweet_count, 
                       AVG(likes) as avg_likes, MAX(likes) as max_likes
                FROM tweets 
                GROUP BY username 
                ORDER BY tweet_count DESC
            ''')
            user_stats = cursor.fetchall()
            
            conn.close()
            
            print(f"\n📊 DATABASE STATISTICS:")
            print(f"   Total tweets: {total_tweets}")
            print(f"   Unique users: {unique_users}")
            print(f"\n📈 TWEETS PER USER:")
            for username, count, avg_likes, max_likes in user_stats:
                print(f"   @{username}: {count} tweets (avg: {avg_likes:.0f} likes, max: {max_likes})")
            
            return {
                'total_tweets': total_tweets,
                'unique_users': unique_users,
                'user_stats': user_stats
            }
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return None
    
    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()

def main():
    """Demo the Twitter Database scraper with famous people."""
    scraper = TwitterDatabaseScraper(headless=True)
    
    # List of famous people to scrape (with their Twitter usernames)
    famous_people = [
        "AOC",              # Alexandria Ocasio-Cortez
        "sama",             # Sam Altman
        "elonmusk",         # Elon Musk
        "realDonaldTrump",  # Donald Trump
        "dril",             # Dril (famous Twitter personality)
        "horse_ebooks",     # Horse ebooks (famous Twitter bot)
        "elonmusk",         # Elon Musk (already listed, but keeping for completeness)
        "BarackObama",      # Barack Obama
        "NASA",             # NASA
        "elonmusk",         # Elon Musk (duplicate, but keeping for variety)
    ]
    
    # Remove duplicates while preserving order
    unique_people = list(dict.fromkeys(famous_people))
    
    try:
        print(f"🚀 Starting Twitter Database Scraper")
        print(f"📋 Will scrape 100 tweets each from {len(unique_people)} famous people")
        print(f"💾 Database: {scraper.db_path}")
        
        total_stored = 0
        
        for i, username in enumerate(unique_people, 1):
            print(f"\n{'='*60}")
            print(f"📱 SCRAPING TWEETS FROM @{username} ({i}/{len(unique_people)})")
            print(f"{'='*60}")
            
            # Scrape and store 100 tweets
            stored_count = scraper.scrape_and_store_tweets(username, target_count=100)
            total_stored += stored_count
            
            # Add delay between requests to be respectful
            if i < len(unique_people):  # Don't delay after the last one
                print("⏳ Waiting 5 seconds before next user...")
                time.sleep(5)
        
        print(f"\n{'='*60}")
        print(f"🎉 SCRAPING COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Total new tweets stored: {total_stored}")
        
        # Show database statistics
        scraper.get_database_stats()
        
        # Show some sample top tweets
        print(f"\n🏆 SAMPLE TOP TWEETS FROM DATABASE:")
        for username in unique_people[:3]:  # Show first 3 users
            top_tweets = scraper.get_top_liked_tweets_from_db(username, count=3)
            if top_tweets:
                print(f"\n📱 @{username} - Top 3 most liked tweets:")
                for i, tweet in enumerate(top_tweets, 1):
                    print(f"   {i}. ❤️ {tweet['likes']} - {tweet['content'][:80]}...")
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 